//! The book must not grow with the number of *cancelled* orders.
//!
//! Agents cancel their resting orders on every wakeup, so run length translates
//! directly into cancel count: a one-year simulation cancels billions of orders.
//! Any record the book keeps per cancelled order therefore turns run length into
//! memory. A previous design kept a `HashSet` of cancelled ids and dropped a
//! price level (with the queue entries that set referred to) whenever a cancel
//! emptied it, so the ids were never reclaimed; a one-year run died trying to
//! grow that set to 309 GB.
//!
//! This lives in its own test binary so the counting allocator observes only
//! this test's allocation traffic.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};

use cda_engine::{LimitOrder, OrderBook, Side};

static LIVE_BYTES: AtomicUsize = AtomicUsize::new(0);

/// Wraps the system allocator to track currently-allocated bytes.
struct Counting;

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        LIVE_BYTES.fetch_add(layout.size(), Ordering::Relaxed);
        System.alloc(layout)
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        LIVE_BYTES.fetch_sub(layout.size(), Ordering::Relaxed);
        System.dealloc(ptr, layout);
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        LIVE_BYTES.fetch_add(new_size, Ordering::Relaxed);
        LIVE_BYTES.fetch_sub(layout.size(), Ordering::Relaxed);
        System.realloc(ptr, layout, new_size)
    }
}

#[global_allocator]
static ALLOC: Counting = Counting;

/// Place then immediately cancel one order per id — the dominant flow in a run.
fn churn(book: &mut OrderBook, ids: std::ops::Range<u64>) {
    for id in ids {
        book.add_limit_order(LimitOrder {
            id,
            side: Side::Bid,
            price: 100,
            qty: 10,
            timestamp: id,
        });
        assert!(book.cancel_order(id), "order {id} should still be cancellable");
    }
}

#[test]
fn place_cancel_churn_does_not_grow_the_book() {
    let mut book = OrderBook::new();

    churn(&mut book, 0..10_000); // let every structure reach its steady capacity
    let after_warmup = LIVE_BYTES.load(Ordering::Relaxed);

    churn(&mut book, 10_000..110_000); // 10x as many cancels again
    let after_churn = LIVE_BYTES.load(Ordering::Relaxed);

    assert_eq!(book.order_count(), 0, "no order should be left resting");
    assert!(
        after_churn <= after_warmup,
        "book grew by {} bytes across 100k further cancels — cancelled orders are being retained",
        after_churn.saturating_sub(after_warmup),
    );
}

/// The tombstones behind a live order are drained once matching reaches them,
/// and never produce fills of their own.
#[test]
fn tombstones_are_skipped_by_matching_and_then_reclaimed() {
    let mut book = OrderBook::new();
    let live = LimitOrder { id: 1, side: Side::Ask, price: 100, qty: 10, timestamp: 0 };
    book.add_limit_order(live);

    for id in 2..1_000 {
        book.add_limit_order(LimitOrder { id, side: Side::Ask, price: 100, qty: 5, timestamp: id });
        assert!(book.cancel_order(id));
    }
    assert_eq!(book.order_count(), 1, "only the uncancelled order is live");
    assert_eq!(book.volume_at(100, Side::Ask), 10, "tombstones carry no volume");

    let taker = LimitOrder { id: 9_999, side: Side::Bid, price: 100, qty: 10, timestamp: 1_000 };
    let result = book.add_limit_order(taker);

    assert_eq!(result.fills.len(), 1, "tombstones must not fill");
    assert_eq!(result.fills[0].maker_order_id, 1);
    assert_eq!(book.order_count(), 0);
    assert_eq!(book.best_ask(), None, "the emptied level is gone");
}
