use cda_engine::{LimitOrder, MarketOrder, OrderBook, OrderStatus, Side};

fn limit(id: u64, side: Side, price: i64, qty: u64, ts: u64) -> LimitOrder {
    LimitOrder { id, side, price, qty, timestamp: ts }
}

fn market(id: u64, side: Side, qty: u64) -> MarketOrder {
    MarketOrder { id, side, qty }
}

// ── 1. Zero quantity ───────────────────────────────────────────────
// qty=0 triggers a debug_assert panic. In release mode it's a no-op.

#[test]
#[cfg(debug_assertions)]
#[should_panic(expected = "limit order qty must be > 0")]
fn zero_qty_limit_order_panics_in_debug() {
    let mut book = OrderBook::new();
    book.add_limit_order(limit(1, Side::Bid, 100, 0, 1));
}

#[test]
#[cfg(debug_assertions)]
#[should_panic(expected = "market order qty must be > 0")]
fn zero_qty_market_order_panics_in_debug() {
    let mut book = OrderBook::new();
    book.add_market_order(market(1, Side::Bid, 0));
}

// ── 2. Duplicate order ID ──────────────────────────────────────────
// Submitting a limit order with an ID already on the book overwrites the
// lookup entry. The old order remains in the queue but becomes un-cancellable
// via the ID (the new order's location wins in the map). This is documented
// as caller responsibility — the simulator must assign unique IDs.

#[test]
fn duplicate_id_overwrites_lookup() {
    let mut book = OrderBook::new();
    book.add_limit_order(limit(1, Side::Bid, 100, 10, 1));
    book.add_limit_order(limit(1, Side::Ask, 200, 5, 2));

    // Cancel hits the *new* mapping (ask side)
    assert!(book.cancel_order(1));
    // The old bid-side order is still in the queue but orphaned from the map.
    // It will be matched normally if a crossing order arrives.
    assert_eq!(book.volume_at(100, Side::Bid), 10);
}

// ── 3. Cancel an order that was just fully filled ──────────────────

#[test]
fn cancel_fully_filled_order_returns_false() {
    let mut book = OrderBook::new();
    book.add_limit_order(limit(1, Side::Ask, 100, 10, 1));
    book.add_limit_order(limit(2, Side::Bid, 100, 10, 2)); // fills order 1

    assert!(!book.cancel_order(1)); // already gone
}

// ── 4. Single tick book ────────────────────────────────────────────

#[test]
fn all_orders_at_same_price() {
    let mut book = OrderBook::new();
    book.add_limit_order(limit(1, Side::Bid, 100, 5, 1));
    book.add_limit_order(limit(2, Side::Bid, 100, 5, 2));
    book.add_limit_order(limit(3, Side::Ask, 100, 5, 3));
    book.add_limit_order(limit(4, Side::Ask, 100, 5, 4));

    // Bids and asks at same price: orders 3 and 4 should have matched against 1 and 2
    // Order 3 (ask@100) crosses bid@100 → matches order 1
    // Order 4 (ask@100) crosses bid@100 → matches order 2
    assert_eq!(book.order_count(), 0);
}

// ── 5. Sequential fill-cancel-add cycles ───────────────────────────

#[test]
fn rapid_fill_cancel_add_cycles() {
    let mut book = OrderBook::new();

    for i in 0..1000_u64 {
        // Add ask
        book.add_limit_order(limit(i * 3, Side::Ask, 100, 1, i));
        // Add bid that crosses
        let res = book.add_limit_order(limit(i * 3 + 1, Side::Bid, 100, 1, i));
        assert_eq!(res.status, OrderStatus::Filled);

        // Add and cancel
        book.add_limit_order(limit(i * 3 + 2, Side::Bid, 99, 1, i));
        assert!(book.cancel_order(i * 3 + 2));
    }

    assert_eq!(book.order_count(), 0);
    assert_eq!(book.best_bid(), None);
    assert_eq!(book.best_ask(), None);
}

// ── 6. Book state after full drain ─────────────────────────────────

#[test]
fn book_empty_after_full_drain() {
    let mut book = OrderBook::new();

    // Populate both sides
    for i in 0..10_u64 {
        book.add_limit_order(limit(i, Side::Bid, 90 + i as i64, 10, i));
        book.add_limit_order(limit(100 + i, Side::Ask, 100 + i as i64, 10, i));
    }
    assert_eq!(book.order_count(), 20);

    // Cancel all bids
    for i in 0..10_u64 {
        book.cancel_order(i);
    }
    // Market sell to drain asks
    let res = book.add_market_order(market(999, Side::Bid, 100));
    assert_eq!(res.status, OrderStatus::Filled);

    assert_eq!(book.order_count(), 0);
    assert_eq!(book.best_bid(), None);
    assert_eq!(book.best_ask(), None);
    assert_eq!(book.spread(), None);
}
