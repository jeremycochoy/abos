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

/// Identity of a just-created order, as echoed to its owner.
#[derive(Clone, Copy)]
struct NewOrder {
    id: u64,
    user_id: u64,
    side: Side,
    qty: u64,
}

struct OrderInfo {
    agent_id: AgentId,
    remaining_qty: u64,
    /// Owner's tag, echoed on every message about the order (0 = unset).
    user_id: u64,
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
                message: ExchangeMessage::OrderCancelled {
                    order_id: oid,
                    user_id: info.user_id,
                    symbol: self.sym,
                },
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
            // A new order never receives an id (order_id 0); its echoed
            // user_id still identifies it. A rejected cancel names its target.
            let (order_id, user_id) = match action {
                OrderAction::CancelOrder { order_id } => (order_id, 0),
                OrderAction::NewLimitOrder { user_id, .. }
                | OrderAction::NewMarketOrder { user_id, .. } => (0, user_id),
            };
            out.push(RoutedMessage {
                agent_id,
                message: ExchangeMessage::OrderRejected { order_id, user_id, symbol: self.sym },
            });
            return;
        }

        let bbo_before = (self.book.best_bid(), self.book.best_ask());

        match action {
            OrderAction::NewLimitOrder { side, price, qty, user_id } => {
                let order = NewOrder { id: self.alloc_id(), user_id, side, qty };
                let result = self.book.add_limit_order(LimitOrder {
                    id: order.id, side, price, qty, timestamp: time,
                });
                self.record_fills(side, &result.fills, time, out);
                self.notify_taker(agent_id, order, &result, out);
            }
            OrderAction::NewMarketOrder { side, qty, user_id } => {
                let order = NewOrder { id: self.alloc_id(), user_id, side, qty };
                let result = self.book.add_market_order(MarketOrder { id: order.id, side, qty });
                self.record_fills(side, &result.fills, time, out);
                self.notify_taker(agent_id, order, &result, out);
            }
            OrderAction::CancelOrder { order_id } => {
                if self.book.cancel_order(order_id) {
                    let user_id = self.resting.remove(&order_id).map_or(0, |info| info.user_id);
                    out.push(RoutedMessage {
                        agent_id,
                        message: ExchangeMessage::OrderCancelled {
                            order_id,
                            user_id,
                            symbol: self.sym,
                        },
                    });
                } else {
                    out.push(RoutedMessage {
                        agent_id,
                        message: ExchangeMessage::OrderRejected {
                            order_id,
                            user_id: 0,
                            symbol: self.sym,
                        },
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

    /// Creation acknowledgment: sent exactly once for every new order,
    /// including one that fully fills at submission, carrying everything the
    /// owner needs to identify the order. Emission order per status
    /// preserves the legacy sequence the existing agents were built against:
    /// full fill → ack THEN fill; partial-fill-and-rest → fill THEN ack;
    /// rest without fill → ack only.
    fn push_accepted(&self, agent_id: AgentId, order: NewOrder, out: &mut Vec<RoutedMessage>) {
        out.push(RoutedMessage {
            agent_id,
            message: ExchangeMessage::OrderAccepted {
                order_id: order.id,
                user_id: order.user_id,
                symbol: self.sym,
                side: order.side,
                qty: order.qty,
            },
        });
    }

    fn record_fills(
        &mut self,
        taker_side: Side,
        fills: &[cda_engine::Fill],
        time: Nanos,
        out: &mut Vec<RoutedMessage>,
    ) {
        let maker_side = match taker_side {
            Side::Bid => Side::Ask,
            Side::Ask => Side::Bid,
        };
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
                info.remaining_qty = info.remaining_qty.saturating_sub(fill.qty);
                out.push(RoutedMessage {
                    agent_id: info.agent_id,
                    message: ExchangeMessage::OrderFilled {
                        order_id: fill.maker_order_id,
                        user_id: info.user_id,
                        symbol: self.sym,
                        side: maker_side,
                        price: fill.price,
                        qty: fill.qty,
                        remaining: info.remaining_qty,
                    },
                });
                if info.remaining_qty == 0 {
                    self.resting.remove(&fill.maker_order_id);
                }
            }
        }
    }

    fn notify_taker(
        &mut self,
        taker: AgentId,
        order: NewOrder,
        result: &cda_engine::OrderResult,
        out: &mut Vec<RoutedMessage>,
    ) {
        let last_price = result.fills.last().map_or(0, |f| f.price);
        match result.status {
            OrderStatus::Filled => {
                self.push_accepted(taker, order, out);
                let total: u64 = result.fills.iter().map(|f| f.qty).sum();
                out.push(RoutedMessage {
                    agent_id: taker,
                    message: ExchangeMessage::OrderFilled {
                        order_id: order.id,
                        user_id: order.user_id,
                        symbol: self.sym,
                        side: order.side,
                        price: last_price,
                        qty: total,
                        remaining: 0,
                    },
                });
            }
            OrderStatus::Placed => {
                self.resting.insert(order.id, OrderInfo {
                    agent_id: taker, remaining_qty: order.qty, user_id: order.user_id,
                });
                self.push_accepted(taker, order, out);
            }
            OrderStatus::Resting { remaining_qty } => {
                self.resting.insert(order.id, OrderInfo {
                    agent_id: taker, remaining_qty, user_id: order.user_id,
                });
                // Legacy order: the submission fill precedes the ack, so an
                // agent that inserts on accept and removes on fill ends up
                // TRACKING the resting remainder, exactly as before.
                let filled: u64 = result.fills.iter().map(|f| f.qty).sum();
                if filled > 0 {
                    out.push(RoutedMessage {
                        agent_id: taker,
                        message: ExchangeMessage::OrderFilled {
                            order_id: order.id,
                            user_id: order.user_id,
                            symbol: self.sym,
                            side: order.side,
                            price: last_price,
                            qty: filled,
                            remaining: remaining_qty,
                        },
                    });
                }
                self.push_accepted(taker, order, out);
            }
            OrderStatus::Cancelled { filled_qty } => {
                if filled_qty > 0 {
                    out.push(RoutedMessage {
                        agent_id: taker,
                        message: ExchangeMessage::OrderFilled {
                            order_id: order.id,
                            user_id: order.user_id,
                            symbol: self.sym,
                            side: order.side,
                            price: last_price,
                            qty: filled_qty,
                            remaining: order.qty - filled_qty,
                        },
                    });
                }
                out.push(RoutedMessage {
                    agent_id: taker,
                    message: ExchangeMessage::OrderCancelled {
                        order_id: order.id,
                        user_id: order.user_id,
                        symbol: self.sym,
                    },
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

#[cfg(test)]
mod tests {
    use super::*;

    const SYM: Symbol = 7;

    fn open_exchange() -> Exchange {
        let mut ex = Exchange::new(SYM, 1_000_000_000);
        ex.open(0);
        ex
    }

    fn limit(side: Side, price: i64, qty: u64, user_id: u64) -> OrderAction {
        OrderAction::NewLimitOrder { side, price, qty, user_id }
    }

    // Every new order gets exactly one creation ack — with symbol, side, qty
    // and the echoed user_id — before any of its fills, INCLUDING an order
    // that fully fills at submission (which previously got no ack at all).
    #[test]
    fn full_fill_still_gets_creation_ack_before_its_fill() {
        let mut ex = open_exchange();
        let mut msgs = Vec::new();
        ex.process_into(0, limit(Side::Ask, 100, 10, 5), 0, &mut msgs);
        msgs.clear();
        ex.process_into(1, limit(Side::Bid, 100, 10, 9), 1, &mut msgs);

        let to_taker: Vec<_> = msgs.iter().filter(|m| m.agent_id == 1).collect();
        assert_eq!(to_taker.len(), 2, "creation ack + fill");
        match to_taker[0].message {
            ExchangeMessage::OrderAccepted { user_id, symbol, side, qty, .. } => {
                assert_eq!((user_id, symbol, side, qty), (9, SYM, Side::Bid, 10));
            }
            other => panic!("first taker message must be the creation ack, got {other:?}"),
        }
        match to_taker[1].message {
            ExchangeMessage::OrderFilled { user_id, symbol, side, qty, remaining, .. } => {
                assert_eq!((user_id, symbol, side, qty, remaining), (9, SYM, Side::Bid, 10, 0));
            }
            other => panic!("second taker message must be the fill, got {other:?}"),
        }
    }

    // An order that partially fills at submission and rests keeps the LEGACY
    // echo order — fill first, then the ack — so an agent that inserts on
    // accept and removes on fill ends up tracking the resting remainder,
    // exactly as it did before this protocol change.
    #[test]
    fn partial_fill_at_submission_emits_fill_then_ack() {
        let mut ex = open_exchange();
        let mut msgs = Vec::new();
        ex.process_into(0, limit(Side::Ask, 100, 4, 5), 0, &mut msgs);
        msgs.clear();
        ex.process_into(1, limit(Side::Bid, 100, 10, 9), 1, &mut msgs);

        let to_taker: Vec<_> = msgs.iter().filter(|m| m.agent_id == 1).collect();
        assert_eq!(to_taker.len(), 2, "submission fill + creation ack");
        match to_taker[0].message {
            ExchangeMessage::OrderFilled { qty, remaining, side, .. } => {
                assert_eq!((qty, remaining, side), (4, 6, Side::Bid));
            }
            other => panic!("first message must be the submission fill, got {other:?}"),
        }
        match to_taker[1].message {
            ExchangeMessage::OrderAccepted { qty, side, .. } => {
                assert_eq!((qty, side), (10, Side::Bid));
            }
            other => panic!("second message must be the creation ack, got {other:?}"),
        }
    }

    // A fill of a resting order carries the RESTING order's side and its
    // owner's user_id, not the aggressor's.
    #[test]
    fn maker_fill_carries_maker_side_and_user_id() {
        let mut ex = open_exchange();
        let mut msgs = Vec::new();
        ex.process_into(0, limit(Side::Ask, 100, 10, 5), 0, &mut msgs);
        msgs.clear();
        ex.process_into(1, limit(Side::Bid, 100, 4, 9), 1, &mut msgs);

        let maker_fill = msgs
            .iter()
            .find(|m| m.agent_id == 0)
            .expect("maker must be notified of the fill");
        match maker_fill.message {
            ExchangeMessage::OrderFilled { user_id, symbol, side, qty, remaining, .. } => {
                assert_eq!((user_id, symbol, side, qty, remaining), (5, SYM, Side::Ask, 4, 6));
            }
            other => panic!("maker must receive a fill, got {other:?}"),
        }
    }

    // A new order rejected on a closed market has no order id, but its
    // echoed user_id and symbol still identify it.
    #[test]
    fn rejected_new_order_echoes_user_id_and_symbol() {
        let mut ex = Exchange::new(SYM, 1_000_000_000); // never opened
        let mut msgs = Vec::new();
        ex.process_into(3, limit(Side::Bid, 100, 10, 42), 0, &mut msgs);
        assert_eq!(msgs.len(), 1);
        match msgs[0].message {
            ExchangeMessage::OrderRejected { order_id, user_id, symbol } => {
                assert_eq!((order_id, user_id, symbol), (0, 42, SYM));
            }
            other => panic!("expected a rejection, got {other:?}"),
        }
    }

    // Cancels (agent-requested and market-close) echo the order's user_id.
    #[test]
    fn cancels_echo_the_orders_user_id() {
        let mut ex = open_exchange();
        let mut msgs = Vec::new();
        ex.process_into(0, limit(Side::Bid, 90, 10, 11), 0, &mut msgs);
        let oid = match msgs[0].message {
            ExchangeMessage::OrderAccepted { order_id, .. } => order_id,
            other => panic!("expected the creation ack, got {other:?}"),
        };

        msgs.clear();
        ex.process_into(0, OrderAction::CancelOrder { order_id: oid }, 1, &mut msgs);
        match msgs[0].message {
            ExchangeMessage::OrderCancelled { order_id, user_id, symbol } => {
                assert_eq!((order_id, user_id, symbol), (oid, 11, SYM));
            }
            other => panic!("expected the cancel echo, got {other:?}"),
        }

        msgs.clear();
        ex.process_into(0, limit(Side::Bid, 90, 10, 12), 2, &mut msgs);
        msgs.clear();
        ex.close_into(3, &mut msgs);
        match msgs[0].message {
            ExchangeMessage::OrderCancelled { user_id, symbol, .. } => {
                assert_eq!((user_id, symbol), (12, SYM));
            }
            other => panic!("expected the close-cancel echo, got {other:?}"),
        }
    }
}
