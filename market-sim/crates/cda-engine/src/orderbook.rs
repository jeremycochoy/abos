use std::collections::{BTreeMap, HashMap, VecDeque};

use crate::fills::{Fill, OrderResult, OrderStatus};
use crate::order::{LimitOrder, MarketOrder, RestingOrder, Side};

/// A continuous double-auction order book with price-time (FIFO) priority.
///
/// Internally uses `BTreeMap<i64, VecDeque<RestingOrder>>` per side and a
/// `HashMap<u64, (Side, i64)>` for O(1) cancel lookups.
pub struct OrderBook {
    bids: BTreeMap<i64, VecDeque<RestingOrder>>,
    asks: BTreeMap<i64, VecDeque<RestingOrder>>,
    /// `order_id` → (side, price) for O(1) cancel.
    orders: HashMap<u64, (Side, i64)>,
    /// Reusable fill buffer to avoid per-call allocation.
    fill_buf: Vec<Fill>,
}

impl OrderBook {
    /// Create an empty order book.
    #[must_use]
    pub fn new() -> Self {
        Self {
            bids: BTreeMap::new(),
            asks: BTreeMap::new(),
            orders: HashMap::new(),
            fill_buf: Vec::new(),
        }
    }

    /// Submit a limit order. Returns fills (if crossing) and the order's final status.
    ///
    /// # Panics
    /// Panics in debug mode if `qty == 0` or `price < 0`.
    pub fn add_limit_order(&mut self, order: LimitOrder) -> OrderResult {
        debug_assert!(order.qty > 0, "limit order qty must be > 0");
        debug_assert!(order.price >= 0, "price must be non-negative");

        self.fill_buf.clear();
        let mut remaining = order.qty;

        match order.side {
            Side::Bid => self.match_against_asks(order.id, order.price, &mut remaining),
            Side::Ask => self.match_against_bids(order.id, order.price, &mut remaining),
        }

        let status = if remaining == 0 {
            OrderStatus::Filled
        } else {
            self.place_resting(order.side, order.price, RestingOrder {
                id: order.id,
                qty: remaining,
                timestamp: order.timestamp,
            });
            if remaining == order.qty {
                OrderStatus::Placed
            } else {
                OrderStatus::Resting { remaining_qty: remaining }
            }
        };

        OrderResult { fills: self.fill_buf.drain(..).collect(), status }
    }

    /// Submit a market order. Returns fills. Unfilled remainder is cancelled.
    ///
    /// # Panics
    /// Panics in debug mode if `qty == 0`.
    pub fn add_market_order(&mut self, order: MarketOrder) -> OrderResult {
        debug_assert!(order.qty > 0, "market order qty must be > 0");

        self.fill_buf.clear();
        let mut remaining = order.qty;

        match order.side {
            Side::Bid => self.match_against_asks(order.id, i64::MAX, &mut remaining),
            Side::Ask => self.match_against_bids(order.id, 0, &mut remaining),
        }

        let filled_qty = order.qty - remaining;
        let status = if remaining == 0 {
            OrderStatus::Filled
        } else {
            OrderStatus::Cancelled { filled_qty }
        };

        OrderResult { fills: self.fill_buf.drain(..).collect(), status }
    }

    /// Cancel a resting order by ID. Returns `true` if the order was found and removed.
    ///
    /// # Panics
    /// Panics if internal invariants are violated (order tracked but price level missing).
    pub fn cancel_order(&mut self, order_id: u64) -> bool {
        let Some((side, price)) = self.orders.remove(&order_id) else {
            return false;
        };
        let book_side = match side {
            Side::Bid => &mut self.bids,
            Side::Ask => &mut self.asks,
        };
        let queue = book_side.get_mut(&price).expect("price level must exist for tracked order");
        queue.retain(|o| o.id != order_id);
        if queue.is_empty() {
            book_side.remove(&price);
        }
        true
    }

    /// Best (highest) bid price, or `None` if the bid side is empty.
    #[must_use]
    pub fn best_bid(&self) -> Option<i64> {
        self.bids.keys().next_back().copied()
    }

    /// Best (lowest) ask price, or `None` if the ask side is empty.
    #[must_use]
    pub fn best_ask(&self) -> Option<i64> {
        self.asks.keys().next().copied()
    }

    /// Spread (best ask − best bid), or `None` if either side is empty.
    #[must_use]
    pub fn spread(&self) -> Option<i64> {
        Some(self.best_ask()? - self.best_bid()?)
    }

    /// Total resting volume at a given price level and side.
    #[must_use]
    pub fn volume_at(&self, price: i64, side: Side) -> u64 {
        let book_side = match side {
            Side::Bid => &self.bids,
            Side::Ask => &self.asks,
        };
        book_side
            .get(&price)
            .map_or(0, |q| q.iter().map(|o| o.qty).sum())
    }

    /// Total number of resting orders on the book.
    #[must_use]
    pub fn order_count(&self) -> usize {
        self.orders.len()
    }

    // ── internal matching ──────────────────────────────────────────────

    /// Match an incoming bid against resting asks at prices ≤ `limit_price`.
    fn match_against_asks(&mut self, taker_id: u64, limit_price: i64, remaining: &mut u64) {
        while *remaining > 0 {
            let Some((&ask_price, _)) = self.asks.first_key_value() else { break };
            if ask_price > limit_price {
                break;
            }
            self.fill_at_level(Side::Bid, taker_id, ask_price, remaining);
        }
    }

    /// Match an incoming ask against resting bids at prices ≥ `limit_price`.
    fn match_against_bids(&mut self, taker_id: u64, limit_price: i64, remaining: &mut u64) {
        while *remaining > 0 {
            let Some((&bid_price, _)) = self.bids.last_key_value() else { break };
            if bid_price < limit_price {
                break;
            }
            self.fill_at_level(Side::Ask, taker_id, bid_price, remaining);
        }
    }

    /// Drain the front of the queue at `price` on the *opposite* side of `taker_side`,
    /// filling as much of `remaining` as possible. Removes the price level if fully consumed.
    fn fill_at_level(&mut self, taker_side: Side, taker_id: u64, price: i64, remaining: &mut u64) {
        let book_side = match taker_side {
            Side::Bid => &mut self.asks,
            Side::Ask => &mut self.bids,
        };
        let queue = book_side.get_mut(&price).expect("price level must exist");

        while *remaining > 0 {
            let Some(front) = queue.front_mut() else { break };
            let fill_qty = (*remaining).min(front.qty);

            self.fill_buf.push(Fill {
                maker_order_id: front.id,
                taker_order_id: taker_id,
                price,
                qty: fill_qty,
                taker_side,
            });

            *remaining -= fill_qty;
            front.qty -= fill_qty;

            if front.qty == 0 {
                let filled_id = front.id;
                queue.pop_front();
                self.orders.remove(&filled_id);
            }
        }

        if queue.is_empty() {
            book_side.remove(&price);
        }
    }

    /// Insert a resting order into the appropriate side and register it in the lookup map.
    fn place_resting(&mut self, side: Side, price: i64, order: RestingOrder) {
        self.orders.insert(order.id, (side, price));
        let book_side = match side {
            Side::Bid => &mut self.bids,
            Side::Ask => &mut self.asks,
        };
        book_side.entry(price).or_default().push_back(order);
    }
}

impl Default for OrderBook {
    fn default() -> Self {
        Self::new()
    }
}
