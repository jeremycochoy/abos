use std::collections::HashMap;

use cda_engine::{LimitOrder, MarketOrder, OrderBook, OrderStatus, Side};

use crate::event::{ExchangeMessage, OrderAction};
use crate::types::{AgentId, MarketSnapshot, Nanos, Symbol};

/// Record of a single trade, collected for Parquet output.
#[derive(Debug, Clone, Copy)]
pub struct TradeRecord {
    pub timestamp: Nanos,
    pub symbol: Symbol,
    pub price: i64,
    pub qty: u64,
    pub aggressor_side: Side,
    pub maker_order_id: u64,
    pub taker_order_id: u64,
}

/// L1 best-bid/ask snapshot, recorded on BBO changes.
#[derive(Debug, Clone, Copy)]
pub struct L1Snapshot {
    pub timestamp: Nanos,
    pub symbol: Symbol,
    pub bid_price: i64,
    pub ask_price: i64,
    pub bid_volume: u64,
    pub ask_volume: u64,
    pub last_trade_price: i64,
}

/// A message routed to a specific agent.
#[derive(Debug, Clone, Copy)]
pub struct RoutedMessage {
    pub agent_id: AgentId,
    pub message: ExchangeMessage,
}

struct OrderInfo {
    agent_id: AgentId,
    remaining_qty: u64,
}

/// Per-symbol exchange wrapping a [`cda_engine::OrderBook`].
pub struct Exchange {
    sym: Symbol,
    book: OrderBook,
    is_open: bool,
    next_order_id: u64,
    resting: HashMap<u64, OrderInfo>,
    last_trade_price: Option<i64>,
    last_trade_time: Option<Nanos>,
    /// Accumulated trade records (drained at end of simulation).
    pub trades: Vec<TradeRecord>,
    /// Accumulated L1 snapshots (drained at end of simulation).
    pub l1_snapshots: Vec<L1Snapshot>,
}

impl Exchange {
    /// Create a new closed exchange for the given symbol.
    #[must_use]
    pub fn new(symbol: Symbol, starting_order_id: u64) -> Self {
        Self {
            sym: symbol,
            book: OrderBook::new(),
            is_open: false,
            next_order_id: starting_order_id,
            resting: HashMap::new(),
            last_trade_price: None,
            last_trade_time: None,
            trades: Vec::with_capacity(1 << 16),
            l1_snapshots: Vec::with_capacity(1 << 16),
        }
    }

    /// The symbol this exchange serves.
    #[must_use]
    pub fn symbol(&self) -> Symbol {
        self.sym
    }

    /// Current top-of-book snapshot.
    #[must_use]
    pub fn snapshot(&self) -> MarketSnapshot {
        let best_bid = self.book.best_bid().map(|p| (p, self.book.volume_at(p, Side::Bid)));
        let best_ask = self.book.best_ask().map(|p| (p, self.book.volume_at(p, Side::Ask)));
        MarketSnapshot {
            best_bid,
            best_ask,
            last_trade_price: self.last_trade_price,
            last_trade_time: self.last_trade_time,
        }
    }

    /// Open the market for trading.
    pub fn open(&mut self, time: Nanos) {
        self.is_open = true;
        self.record_l1(time);
    }

    /// Close the market, cancelling all resting orders.
    pub fn close_into(&mut self, time: Nanos, out: &mut Vec<RoutedMessage>) {
        self.is_open = false;
        for (&oid, info) in &self.resting {
            self.book.cancel_order(oid);
            out.push(RoutedMessage {
                agent_id: info.agent_id,
                message: ExchangeMessage::OrderCancelled { order_id: oid },
            });
        }
        self.resting.clear();
        self.record_l1(time);
    }

    /// Process an order action from an agent. Writes messages into `out`.
    pub fn process_into(
        &mut self,
        agent_id: AgentId,
        action: OrderAction,
        time: Nanos,
        out: &mut Vec<RoutedMessage>,
    ) {
        if !self.is_open {
            let oid = match action {
                OrderAction::CancelOrder { order_id } => order_id,
                _ => 0,
            };
            out.push(RoutedMessage {
                agent_id,
                message: ExchangeMessage::OrderRejected { order_id: oid },
            });
            return;
        }

        let bbo_before = (self.book.best_bid(), self.book.best_ask());

        match action {
            OrderAction::NewLimitOrder { side, price, qty } => {
                let oid = self.alloc_id();
                let result = self.book.add_limit_order(LimitOrder {
                    id: oid, side, price, qty, timestamp: time,
                });
                self.record_fills(side, &result.fills, time, out);
                self.notify_taker(agent_id, oid, qty, &result, out);
            }
            OrderAction::NewMarketOrder { side, qty } => {
                let oid = self.alloc_id();
                let result = self.book.add_market_order(MarketOrder { id: oid, side, qty });
                self.record_fills(side, &result.fills, time, out);
                self.notify_taker(agent_id, oid, qty, &result, out);
            }
            OrderAction::CancelOrder { order_id } => {
                if self.book.cancel_order(order_id) {
                    self.resting.remove(&order_id);
                    out.push(RoutedMessage {
                        agent_id,
                        message: ExchangeMessage::OrderCancelled { order_id },
                    });
                } else {
                    out.push(RoutedMessage {
                        agent_id,
                        message: ExchangeMessage::OrderRejected { order_id },
                    });
                }
            }
        }

        let bbo_after = (self.book.best_bid(), self.book.best_ask());
        if bbo_after != bbo_before {
            self.record_l1(time);
        }
    }

    /// Total resting orders on the book.
    #[must_use]
    pub fn order_count(&self) -> usize {
        self.book.order_count()
    }

    fn alloc_id(&mut self) -> u64 {
        let id = self.next_order_id;
        self.next_order_id += 1;
        id
    }

    fn record_fills(
        &mut self,
        taker_side: Side,
        fills: &[cda_engine::Fill],
        time: Nanos,
        out: &mut Vec<RoutedMessage>,
    ) {
        for fill in fills {
            self.last_trade_price = Some(fill.price);
            self.last_trade_time = Some(time);

            self.trades.push(TradeRecord {
                timestamp: time,
                symbol: self.sym,
                price: fill.price,
                qty: fill.qty,
                aggressor_side: taker_side,
                maker_order_id: fill.maker_order_id,
                taker_order_id: fill.taker_order_id,
            });

            if let Some(info) = self.resting.get_mut(&fill.maker_order_id) {
                out.push(RoutedMessage {
                    agent_id: info.agent_id,
                    message: ExchangeMessage::OrderFilled {
                        order_id: fill.maker_order_id,
                        price: fill.price,
                        qty: fill.qty,
                    },
                });
                info.remaining_qty = info.remaining_qty.saturating_sub(fill.qty);
                if info.remaining_qty == 0 {
                    self.resting.remove(&fill.maker_order_id);
                }
            }
        }
    }

    fn notify_taker(
        &mut self,
        taker: AgentId,
        oid: u64,
        submitted_qty: u64,
        result: &cda_engine::OrderResult,
        out: &mut Vec<RoutedMessage>,
    ) {
        let last_price = result.fills.last().map_or(0, |f| f.price);
        match result.status {
            OrderStatus::Filled => {
                let total: u64 = result.fills.iter().map(|f| f.qty).sum();
                out.push(RoutedMessage {
                    agent_id: taker,
                    message: ExchangeMessage::OrderFilled {
                        order_id: oid, price: last_price, qty: total,
                    },
                });
            }
            OrderStatus::Placed => {
                self.resting.insert(oid, OrderInfo {
                    agent_id: taker, remaining_qty: submitted_qty,
                });
                out.push(RoutedMessage {
                    agent_id: taker,
                    message: ExchangeMessage::OrderAccepted { order_id: oid },
                });
            }
            OrderStatus::Resting { remaining_qty } => {
                self.resting.insert(oid, OrderInfo {
                    agent_id: taker, remaining_qty,
                });
                let filled: u64 = result.fills.iter().map(|f| f.qty).sum();
                if filled > 0 {
                    out.push(RoutedMessage {
                        agent_id: taker,
                        message: ExchangeMessage::OrderFilled {
                            order_id: oid, price: last_price, qty: filled,
                        },
                    });
                }
                out.push(RoutedMessage {
                    agent_id: taker,
                    message: ExchangeMessage::OrderAccepted { order_id: oid },
                });
            }
            OrderStatus::Cancelled { filled_qty } => {
                if filled_qty > 0 {
                    out.push(RoutedMessage {
                        agent_id: taker,
                        message: ExchangeMessage::OrderFilled {
                            order_id: oid, price: last_price, qty: filled_qty,
                        },
                    });
                }
                out.push(RoutedMessage {
                    agent_id: taker,
                    message: ExchangeMessage::OrderCancelled { order_id: oid },
                });
            }
        }
    }

    fn record_l1(&mut self, time: Nanos) {
        let bid_price = self.book.best_bid().unwrap_or(0);
        let ask_price = self.book.best_ask().unwrap_or(0);
        let bid_volume = self.book.best_bid().map_or(0, |p| self.book.volume_at(p, Side::Bid));
        let ask_volume = self.book.best_ask().map_or(0, |p| self.book.volume_at(p, Side::Ask));
        self.l1_snapshots.push(L1Snapshot {
            timestamp: time,
            symbol: self.sym,
            bid_price,
            ask_price,
            bid_volume,
            ask_volume,
            last_trade_price: self.last_trade_price.unwrap_or(0),
        });
    }
}
