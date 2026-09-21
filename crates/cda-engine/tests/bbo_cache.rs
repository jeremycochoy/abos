//! The cached best-bid/ask level equals a scan of the price band (issue #10).
//!
//! The book keeps the best level of each side in a cache so hot-path BBO
//! reads cost no tree walk. `volume_at` still reads the tree, so a full-band
//! probe through it is an independent ground truth for the cache.

use cda_engine::{LimitOrder, MarketOrder, OrderBook, Side};

const BAND_LO: i64 = 9_900;
const BAND_HI: i64 = 10_100;

/// Ground truth from the tree: scan the whole band through `volume_at`.
fn scan_best(book: &OrderBook, side: Side) -> Option<(i64, u64)> {
    let prices: Box<dyn Iterator<Item = i64>> = match side {
        Side::Bid => Box::new((BAND_LO..=BAND_HI).rev()),
        Side::Ask => Box::new(BAND_LO..=BAND_HI),
    };
    for p in prices {
        let v = book.volume_at(p, side);
        if v > 0 {
            return Some((p, v));
        }
    }
    None
}

fn assert_cache_matches(book: &OrderBook, context: &str) {
    let bid = scan_best(book, Side::Bid);
    let ask = scan_best(book, Side::Ask);
    assert_eq!(book.best_bid_level(), bid, "bid cache diverged {context}");
    assert_eq!(book.best_ask_level(), ask, "ask cache diverged {context}");
    assert_eq!(book.best_bid(), bid.map(|(p, _)| p), "best_bid diverged {context}");
    assert_eq!(book.best_ask(), ask.map(|(p, _)| p), "best_ask diverged {context}");
}

struct XorShift(u64);

impl XorShift {
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
}

#[test]
fn cache_survives_a_random_op_stream() {
    for seed in [3, 77, 20_260_921] {
        let mut rng = XorShift(seed);
        let mut book = OrderBook::new();
        let mut live_ids: Vec<u64> = Vec::new();
        let mut next_id = 1;
        for step in 0..4_000 {
            let roll = rng.next() % 100;
            if roll < 55 {
                let side = if rng.next().is_multiple_of(2) { Side::Bid } else { Side::Ask };
                let price = BAND_LO + 40 + i64::try_from(rng.next() % 121).unwrap();
                let qty = 1 + rng.next() % 9;
                let id = next_id;
                next_id += 1;
                book.add_limit_order(LimitOrder { id, side, price, qty, timestamp: step });
                live_ids.push(id);
            } else if roll < 85 {
                if !live_ids.is_empty() {
                    let pick = rng.next() as usize % live_ids.len();
                    let id = live_ids.swap_remove(pick);
                    book.cancel_order(id);
                }
            } else {
                let side = if rng.next().is_multiple_of(2) { Side::Bid } else { Side::Ask };
                let qty = 1 + rng.next() % 30;
                let id = next_id;
                next_id += 1;
                book.add_market_order(MarketOrder { id, side, qty });
            }
            assert_cache_matches(&book, &format!("after step {step} of seed {seed}"));
        }
    }
}

#[test]
fn cache_follows_the_hand_written_edges() {
    let mut book = OrderBook::new();
    assert_cache_matches(&book, "on the empty book");

    // Build two bid levels and one ask level.
    book.add_limit_order(LimitOrder { id: 1, side: Side::Bid, price: 10_000, qty: 5, timestamp: 0 });
    book.add_limit_order(LimitOrder { id: 2, side: Side::Bid, price: 10_000, qty: 3, timestamp: 1 });
    book.add_limit_order(LimitOrder { id: 3, side: Side::Bid, price: 9_990, qty: 7, timestamp: 2 });
    book.add_limit_order(LimitOrder { id: 4, side: Side::Ask, price: 10_010, qty: 4, timestamp: 3 });
    assert_cache_matches(&book, "after the initial build");

    // A better bid moves the cache; an equal bid adds volume.
    book.add_limit_order(LimitOrder { id: 5, side: Side::Bid, price: 10_005, qty: 2, timestamp: 4 });
    assert_cache_matches(&book, "after a better bid");

    // Cancel the lone best bid: the cache falls back to 10_000.
    book.cancel_order(5);
    assert_cache_matches(&book, "after cancelling the best bid");

    // A partial fill at the best level shrinks its volume.
    book.add_limit_order(LimitOrder { id: 6, side: Side::Ask, price: 10_000, qty: 2, timestamp: 5 });
    assert_cache_matches(&book, "after a partial fill of the best bid");

    // A tombstone at the front of the best level, then a fill through it.
    book.cancel_order(1);
    assert_cache_matches(&book, "after a tombstone at the best bid");
    book.add_market_order(MarketOrder { id: 7, side: Side::Ask, qty: 10 });
    assert_cache_matches(&book, "after a market order through two levels");

    // Drain the ask side to empty.
    book.add_market_order(MarketOrder { id: 8, side: Side::Bid, qty: 99 });
    assert_cache_matches(&book, "after draining the ask side");
}
