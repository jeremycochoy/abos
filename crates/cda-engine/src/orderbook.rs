use std::collections::{BTreeMap, HashMap, VecDeque};

use crate::fills::{Fill, OrderResult, OrderStatus};
use crate::order::{LimitOrder, MarketOrder, RestingOrder, Side};

/// Orders queued at a single price level, with cached total volume.
struct PriceLevel {
    orders: VecDeque<RestingOrder>,
    total_qty: u64,
}

/// A continuous double-auction order book with price-time (FIFO) priority.
///
/// Uses lazy (tombstone) cancellation: a cancelled order stays queued at its
/// price level and is skipped when matching reaches it. This gives O(1) cancel
/// instead of O(n). `orders` is the authority on what is live, so a queue entry
/// missing from it *is* the tombstone — no separate set of cancelled ids is
/// kept, which is what bounds the book's memory over a long run.
pub struct OrderBook {
    bids: BTreeMap<i64, PriceLevel>,
    asks: BTreeMap<i64, PriceLevel>,
    /// `order_id` → (side, price, `remaining_qty`) for O(1) cancel.
    /// Holds exactly the live resting orders.
    orders: HashMap<u64, (Side, i64, u64)>,
    /// Reusable fill buffer to avoid per-call allocation.
    fill_buf: Vec<Fill>,
    /// Cached best bid as (price, live volume), kept in step with `bids` so
    /// hot-path BBO reads cost no tree walk.
    best_bid_cache: Option<(i64, u64)>,
    /// Cached best ask as (price, live volume).
    best_ask_cache: Option<(i64, u64)>,
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
            best_bid_cache: None,
            best_ask_cache: None,
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
    /// Dropping the order from `orders` is what marks it cancelled; its queue
    /// entry is left behind as a tombstone for matching to skip.
    pub fn cancel_order(&mut self, order_id: u64) -> bool {
        let Some((side, price, remaining_qty)) = self.orders.remove(&order_id) else {
            return false;
        };
        // Update cached volume at the price level
        let book_side = match side {
            Side::Bid => &mut self.bids,
            Side::Ask => &mut self.asks,
        };
        if let Some(level) = book_side.get_mut(&price) {
            level.total_qty -= remaining_qty;
            // No live volume left: drop the level, discarding its tombstones.
            if level.total_qty == 0 {
                book_side.remove(&price);
            }
        }
        if self.reaches_best(side, price) {
            self.refresh_best(side);
        }
        true
    }

    /// Best (highest) bid price, or `None` if the bid side is empty.
    #[must_use]
    pub fn best_bid(&self) -> Option<i64> {
        self.best_bid_cache.map(|(price, _)| price)
    }

    /// Best (lowest) ask price, or `None` if the ask side is empty.
    #[must_use]
    pub fn best_ask(&self) -> Option<i64> {
        self.best_ask_cache.map(|(price, _)| price)
    }

    /// Best bid as (price, live volume at that price), or `None` if the bid
    /// side is empty. Reads the cache: no tree walk.
    #[must_use]
    pub fn best_bid_level(&self) -> Option<(i64, u64)> {
        self.best_bid_cache
    }

    /// Best ask as (price, live volume at that price), or `None` if the ask
    /// side is empty. Reads the cache: no tree walk.
    #[must_use]
    pub fn best_ask_level(&self) -> Option<(i64, u64)> {
        self.best_ask_cache
    }

    /// Recompute one side's cache from its tree. Called only when a change
    /// touches that side's best level.
    fn refresh_best(&mut self, side: Side) {
        match side {
            Side::Bid => {
                self.best_bid_cache =
                    self.bids.last_key_value().map(|(&price, level)| (price, level.total_qty));
            }
            Side::Ask => {
                self.best_ask_cache =
                    self.asks.first_key_value().map(|(&price, level)| (price, level.total_qty));
            }
        }
    }

    /// True when `price` is at least as good as the side's cached best.
    fn reaches_best(&self, side: Side, price: i64) -> bool {
        match side {
            Side::Bid => self.best_bid_cache.is_none_or(|(best, _)| price >= best),
            Side::Ask => self.best_ask_cache.is_none_or(|(best, _)| price <= best),
        }
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
        book_side.get(&price).map_or(0, |level| level.total_qty)
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

    /// Pop cancelled tombstones from the front of a price level's queue.
    ///
    /// A queued order is a tombstone exactly when `orders` no longer holds it:
    /// a full fill drops it from the queue and from `orders` together, so a
    /// queue entry with no `orders` record can only have been cancelled.
    fn drain_tombstones(level: &mut PriceLevel, orders: &HashMap<u64, (Side, i64, u64)>) {
        while let Some(front) = level.orders.front() {
            if orders.contains_key(&front.id) {
                break;
            }
            level.orders.pop_front();
        }
    }

    /// Drain the front of the queue at `price` on the *opposite* side of `taker_side`,
    /// filling as much of `remaining` as possible. Removes the price level if fully consumed.
    fn fill_at_level(&mut self, taker_side: Side, taker_id: u64, price: i64, remaining: &mut u64) {
        let book_side = match taker_side {
            Side::Bid => &mut self.asks,
            Side::Ask => &mut self.bids,
        };
        let level = book_side.get_mut(&price).expect("price level must exist");

        Self::drain_tombstones(level, &self.orders);

        while *remaining > 0 {
            let Some(front) = level.orders.front_mut() else { break };
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
            level.total_qty -= fill_qty;

            if front.qty == 0 {
                let filled_id = front.id;
                level.orders.pop_front();
                self.orders.remove(&filled_id);
                // Update remaining_qty for partial fills tracked in orders map
            } else if let Some(entry) = self.orders.get_mut(&front.id) {
                entry.2 -= fill_qty;
            }

            Self::drain_tombstones(level, &self.orders);
        }

        if level.orders.is_empty() {
            book_side.remove(&price);
        }
        self.refresh_best(match taker_side {
            Side::Bid => Side::Ask,
            Side::Ask => Side::Bid,
        });
    }

    /// Insert a resting order into the appropriate side and register it in the lookup map.
    fn place_resting(&mut self, side: Side, price: i64, order: RestingOrder) {
        self.orders.insert(order.id, (side, price, order.qty));
        let book_side = match side {
            Side::Bid => &mut self.bids,
            Side::Ask => &mut self.asks,
        };
        let level = book_side.entry(price).or_insert_with(|| PriceLevel {
            orders: VecDeque::new(),
            total_qty: 0,
        });
        level.total_qty += order.qty;
        level.orders.push_back(order);
        let cache = match side {
            Side::Bid => &mut self.best_bid_cache,
            Side::Ask => &mut self.best_ask_cache,
        };
        let better = match (side, *cache) {
            (_, None) => true,
            (Side::Bid, Some((best, _))) => price > best,
            (Side::Ask, Some((best, _))) => price < best,
        };
        if better {
            *cache = Some((price, level.total_qty));
        } else if let Some((best, volume)) = cache {
            if *best == price {
                *volume += order.qty;
            }
        }
    }
}

impl Default for OrderBook {
    fn default() -> Self {
        Self::new()
    }
}
