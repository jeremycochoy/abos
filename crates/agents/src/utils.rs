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
pub(crate) fn mid_price(snap: &MarketSnapshot, reference: i64) -> i64 {
    match (snap.best_bid, snap.best_ask) {
        (Some((bid, _)), Some((ask, _))) => bid + (ask - bid) / 2,
        (Some((bid, _)), None) => bid,
        (None, Some((ask, _))) => ask,
        (None, None) => reference,
    }
}
