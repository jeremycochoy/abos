use std::collections::HashMap;

use sim_core::MarketSnapshot;

/// Set supporting O(1) insert, remove-by-value, and random-index selection.
pub(crate) struct IndexedSet {
    vec: Vec<u64>,
    map: HashMap<u64, usize>,
}

#[allow(dead_code)]
impl IndexedSet {
    pub fn new() -> Self {
        Self { vec: Vec::new(), map: HashMap::new() }
    }

    pub fn len(&self) -> usize {
        self.vec.len()
    }

    pub fn is_empty(&self) -> bool {
        self.vec.is_empty()
    }

    pub fn insert(&mut self, val: u64) {
        let idx = self.vec.len();
        self.vec.push(val);
        self.map.insert(val, idx);
    }

    pub fn remove(&mut self, val: u64) -> bool {
        let Some(idx) = self.map.remove(&val) else { return false };
        self.vec.swap_remove(idx);
        if idx < self.vec.len() {
            let swapped = self.vec[idx];
            self.map.insert(swapped, idx);
        }
        true
    }

    pub fn remove_at(&mut self, idx: usize) -> u64 {
        let val = self.vec.swap_remove(idx);
        self.map.remove(&val);
        if idx < self.vec.len() {
            let swapped = self.vec[idx];
            self.map.insert(swapped, idx);
        }
        val
    }

    /// Remove all elements and return their IDs.
    pub fn drain_all(&mut self) -> Vec<u64> {
        self.map.clear();
        std::mem::take(&mut self.vec)
    }
}

/// Compute mid-price from a market snapshot, falling back to `reference`.
///
/// A mid between two ticks rounds down half of the time and up half of the time,
/// so the two rounding errors cancel on average. It rounds to the even tick.
pub(crate) fn mid_price(snap: &MarketSnapshot, reference: i64) -> i64 {
    match (snap.best_bid, snap.best_ask) {
        (Some((bid, _)), Some((ask, _))) => {
            let below = i64::midpoint(bid, ask);
            if (bid + ask) % 2 != 0 && below % 2 != 0 { below + 1 } else { below }
        }
        (Some((bid, _)), None) => bid,
        (None, Some((ask, _))) => ask,
        (None, None) => reference,
    }
}

/// Geometric mid `sqrt(bid * ask)` of a book (as in ABIDES), falling back to the price
/// of the only side, then to `reference`.
///
/// For an odd spread the arithmetic mid is a half tick, and `sqrt(bid * ask)` sits just
/// below it, so a plain round always takes the lower tick: quotes centred on it sit half a
/// tick low, and on a coarse tick grid the price drifts down. When the geometric mean is
/// less than half a tick below that half tick, the mid rounds to the even tick, as
/// `mid_price` does. A wide spread keeps the plain geometric mean.
#[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
pub(crate) fn geometric_mid_price(snap: &MarketSnapshot, reference: i64) -> i64 {
    match (snap.best_bid, snap.best_ask) {
        (Some((bid, _)), Some((ask, _))) => {
            let geometric = (bid as f64 * ask as f64).sqrt();
            let arithmetic_mid = (bid + ask) as f64 / 2.0;
            if (bid + ask) % 2 != 0 && arithmetic_mid - geometric < 0.5 {
                mid_price(snap, reference)
            } else {
                geometric.round() as i64
            }
        }
        (Some((bid, _)), None) => bid,
        (None, Some((ask, _))) => ask,
        (None, None) => reference,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn snapshot(bid: i64, ask: i64) -> MarketSnapshot {
        MarketSnapshot {
            best_bid: Some((bid, 100)),
            best_ask: Some((ask, 100)),
            last_trade_price: Some(bid),
            last_trade_time: Some(0),
        }
    }

    #[test]
    fn mid_price_of_an_even_spread_is_exact() {
        assert_eq!(mid_price(&snapshot(100, 102), 0), 101);
    }

    #[test]
    fn mid_price_of_an_odd_spread_has_no_downward_bias() {
        // A half-tick mid goes to the even neighbour, so the error cancels over
        // many books. Truncation would give 100 and 101, low by half a tick each.
        assert_eq!(mid_price(&snapshot(100, 101), 0), 100);
        assert_eq!(mid_price(&snapshot(101, 102), 0), 102);
    }

    #[test]
    fn geometric_mid_of_an_odd_spread_rounds_to_the_even_tick() {
        // sqrt(bid * ask) of an odd spread sits just below the half tick. A plain
        // round gives 100, 101 and 4381: low by half a tick each time.
        assert_eq!(geometric_mid_price(&snapshot(100, 101), 0), 100);
        assert_eq!(geometric_mid_price(&snapshot(101, 102), 0), 102);
        assert_eq!(geometric_mid_price(&snapshot(4380, 4383), 0), 4382);
    }

    #[test]
    #[allow(clippy::cast_precision_loss)]
    fn geometric_mid_of_odd_spreads_has_no_mean_bias() {
        let errors: Vec<f64> = (4000..4200)
            .flat_map(|bid| [1, 3].map(|spread| (bid, bid + spread)))
            .map(|(bid, ask)| {
                geometric_mid_price(&snapshot(bid, ask), 0) as f64 - (bid + ask) as f64 / 2.0
            })
            .collect();
        let mean = errors.iter().sum::<f64>() / errors.len() as f64;
        assert!(mean.abs() < 0.01, "mean rounding error {mean} ticks");
    }

    #[test]
    fn geometric_mid_of_a_wide_or_even_spread_stays_geometric() {
        assert_eq!(geometric_mid_price(&snapshot(9000, 11000), 0), 9950);
        assert_eq!(geometric_mid_price(&snapshot(100, 102), 0), 101);
    }

    #[test]
    fn geometric_mid_of_a_one_sided_or_empty_book_falls_back() {
        let one_side = MarketSnapshot {
            best_bid: Some((100, 1)),
            best_ask: None,
            last_trade_price: None,
            last_trade_time: None,
        };
        assert_eq!(geometric_mid_price(&one_side, 7), 100);
        let empty = MarketSnapshot {
            best_bid: None,
            best_ask: None,
            last_trade_price: None,
            last_trade_time: None,
        };
        assert_eq!(geometric_mid_price(&empty, 7), 7);
    }
}
