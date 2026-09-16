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
/// A half-tick mid goes to the even neighbour, so the rounding error cancels
/// over many books. Truncation would cut the half tick on every odd spread and
/// push the price down, by an amount that matters when a tick is a large
/// fraction of the price.
pub(crate) fn mid_price(snap: &MarketSnapshot, reference: i64) -> i64 {
    match (snap.best_bid, snap.best_ask) {
        (Some((bid, _)), Some((ask, _))) => {
            let half = i64::midpoint(bid, ask); // Rounds down: prices are positive
            let odd_spread = (bid ^ ask) & 1;
            half + odd_spread * half.rem_euclid(2)
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
}
