use cda_engine::{Fill, LimitOrder, MarketOrder, OrderBook, OrderStatus, Side};

fn limit(id: u64, side: Side, price: i64, qty: u64, ts: u64) -> LimitOrder {
    LimitOrder { id, side, price, qty, timestamp: ts }
}

fn market(id: u64, side: Side, qty: u64) -> MarketOrder {
    MarketOrder { id, side, qty }
}

// ── 1. Basic placement ─────────────────────────────────────────────

#[test]
fn limit_order_on_empty_book_rests() {
    let mut book = OrderBook::new();
    let res = book.add_limit_order(limit(1, Side::Bid, 100, 10, 1));
    assert_eq!(res.status, OrderStatus::Placed);
    assert!(res.fills.is_empty());
    assert_eq!(book.best_bid(), Some(100));
    assert_eq!(book.order_count(), 1);
}

// ── 2. Basic match ─────────────────────────────────────────────────

#[test]
fn limit_buy_crosses_resting_ask() {
    let mut book = OrderBook::new();
    book.add_limit_order(limit(1, Side::Ask, 100, 10, 1));

    let res = book.add_limit_order(limit(2, Side::Bid, 100, 10, 2));
    assert_eq!(res.status, OrderStatus::Filled);
    assert_eq!(res.fills.len(), 1);
    assert_eq!(res.fills[0], Fill {
        maker_order_id: 1,
        taker_order_id: 2,
        price: 100,
        qty: 10,
        taker_side: Side::Bid,
    });
    assert_eq!(book.order_count(), 0);
}

// ── 3. Partial fill (aggressor larger) ─────────────────────────────

#[test]
fn aggressor_larger_than_resting() {
    let mut book = OrderBook::new();
    book.add_limit_order(limit(1, Side::Ask, 100, 5, 1));

    let res = book.add_limit_order(limit(2, Side::Bid, 100, 10, 2));
    assert_eq!(res.status, OrderStatus::Resting { remaining_qty: 5 });
    assert_eq!(res.fills.len(), 1);
    assert_eq!(res.fills[0].qty, 5);
    assert_eq!(book.best_bid(), Some(100));
    assert_eq!(book.volume_at(100, Side::Bid), 5);
}

// ── 4. Partial fill (aggressor smaller) ────────────────────────────

#[test]
fn aggressor_smaller_than_resting() {
    let mut book = OrderBook::new();
    book.add_limit_order(limit(1, Side::Ask, 100, 10, 1));

    let res = book.add_limit_order(limit(2, Side::Bid, 100, 3, 2));
    assert_eq!(res.status, OrderStatus::Filled);
    assert_eq!(res.fills[0].qty, 3);
    assert_eq!(book.volume_at(100, Side::Ask), 7);
    assert_eq!(book.order_count(), 1);
}

// ── 5. Multi-level sweep ───────────────────────────────────────────

#[test]
fn aggressive_limit_sweeps_multiple_levels() {
    let mut book = OrderBook::new();
    book.add_limit_order(limit(1, Side::Ask, 100, 5, 1));
    book.add_limit_order(limit(2, Side::Ask, 101, 5, 2));
    book.add_limit_order(limit(3, Side::Ask, 102, 5, 3));

    let res = book.add_limit_order(limit(10, Side::Bid, 101, 8, 4));
    assert_eq!(res.status, OrderStatus::Filled);
    assert_eq!(res.fills.len(), 2);
    assert_eq!(res.fills[0], Fill {
        maker_order_id: 1, taker_order_id: 10, price: 100, qty: 5, taker_side: Side::Bid,
    });
    assert_eq!(res.fills[1], Fill {
        maker_order_id: 2, taker_order_id: 10, price: 101, qty: 3, taker_side: Side::Bid,
    });
    // Ask at 101 has 2 remaining, ask at 102 untouched
    assert_eq!(book.volume_at(101, Side::Ask), 2);
    assert_eq!(book.volume_at(102, Side::Ask), 5);
    assert_eq!(book.best_ask(), Some(101));
}

// ── 6. Price-time priority ─────────────────────────────────────────

#[test]
fn earlier_order_fills_first_at_same_price() {
    let mut book = OrderBook::new();
    book.add_limit_order(limit(1, Side::Ask, 100, 5, 10));
    book.add_limit_order(limit(2, Side::Ask, 100, 5, 20));

    let res = book.add_limit_order(limit(3, Side::Bid, 100, 5, 30));
    assert_eq!(res.fills.len(), 1);
    assert_eq!(res.fills[0].maker_order_id, 1); // earlier timestamp wins
    assert_eq!(book.volume_at(100, Side::Ask), 5); // order 2 remains
}

// ── 7. Market order full fill ──────────────────────────────────────

#[test]
fn market_buy_fills_across_levels() {
    let mut book = OrderBook::new();
    book.add_limit_order(limit(1, Side::Ask, 100, 3, 1));
    book.add_limit_order(limit(2, Side::Ask, 101, 3, 2));
    book.add_limit_order(limit(3, Side::Ask, 102, 4, 3));

    let res = book.add_market_order(market(10, Side::Bid, 10));
    assert_eq!(res.status, OrderStatus::Filled);
    assert_eq!(res.fills.len(), 3);
    assert_eq!(res.fills[0].qty, 3);
    assert_eq!(res.fills[1].qty, 3);
    assert_eq!(res.fills[2].qty, 4);
    assert_eq!(book.order_count(), 0);
}

// ── 8. Market order partial fill ───────────────────────────────────

#[test]
fn market_buy_partial_fill_remainder_cancelled() {
    let mut book = OrderBook::new();
    book.add_limit_order(limit(1, Side::Ask, 100, 5, 1));

    let res = book.add_market_order(market(10, Side::Bid, 8));
    assert_eq!(res.status, OrderStatus::Cancelled { filled_qty: 5 });
    assert_eq!(res.fills.len(), 1);
    assert_eq!(res.fills[0].qty, 5);
    assert_eq!(book.order_count(), 0);
    // Must NOT rest on the book
    assert_eq!(book.best_bid(), None);
}

// ── 9. Market order on empty book ──────────────────────────────────

#[test]
fn market_order_on_empty_book() {
    let mut book = OrderBook::new();
    let res = book.add_market_order(market(1, Side::Bid, 100));
    assert_eq!(res.status, OrderStatus::Cancelled { filled_qty: 0 });
    assert!(res.fills.is_empty());
}

// ── 10. Cancel existing order ──────────────────────────────────────

#[test]
fn cancel_existing_order() {
    let mut book = OrderBook::new();
    book.add_limit_order(limit(1, Side::Bid, 100, 10, 1));

    assert!(book.cancel_order(1));
    assert_eq!(book.order_count(), 0);
    assert_eq!(book.best_bid(), None);

    // Order should no longer be matchable
    book.add_limit_order(limit(2, Side::Ask, 100, 10, 2));
    assert_eq!(book.best_ask(), Some(100));
    assert_eq!(book.order_count(), 1);
}

// ── 11. Cancel nonexistent order ───────────────────────────────────

#[test]
fn cancel_nonexistent_order_returns_false() {
    let mut book = OrderBook::new();
    assert!(!book.cancel_order(999));
}

// ── 12. Cancel then re-add ─────────────────────────────────────────

#[test]
fn cancel_then_readd_at_same_level() {
    let mut book = OrderBook::new();
    book.add_limit_order(limit(1, Side::Bid, 100, 10, 1));
    book.cancel_order(1);

    let res = book.add_limit_order(limit(2, Side::Bid, 100, 5, 2));
    assert_eq!(res.status, OrderStatus::Placed);
    assert_eq!(book.volume_at(100, Side::Bid), 5);
    assert_eq!(book.order_count(), 1);
}

// ── 13. Self-trade (different IDs, same price) ─────────────────────

#[test]
fn orders_with_different_ids_at_same_price_match() {
    let mut book = OrderBook::new();
    book.add_limit_order(limit(1, Side::Bid, 100, 10, 1));

    let res = book.add_limit_order(limit(2, Side::Ask, 100, 10, 2));
    assert_eq!(res.status, OrderStatus::Filled);
    assert_eq!(res.fills.len(), 1);
    assert_eq!(book.order_count(), 0);
}

// ── 14. Best bid/ask updates correctly ─────────────────────────────

#[test]
fn bbo_updates_after_adds_fills_and_cancels() {
    let mut book = OrderBook::new();
    assert_eq!(book.best_bid(), None);
    assert_eq!(book.best_ask(), None);

    book.add_limit_order(limit(1, Side::Bid, 99, 5, 1));
    book.add_limit_order(limit(2, Side::Bid, 100, 5, 2));
    assert_eq!(book.best_bid(), Some(100));

    // Cancel best bid → next level becomes best
    book.cancel_order(2);
    assert_eq!(book.best_bid(), Some(99));

    // Fill the remaining bid
    book.add_limit_order(limit(3, Side::Ask, 99, 5, 3));
    assert_eq!(book.best_bid(), None);

    // Ask side
    book.add_limit_order(limit(4, Side::Ask, 200, 5, 4));
    book.add_limit_order(limit(5, Side::Ask, 201, 5, 5));
    assert_eq!(book.best_ask(), Some(200));

    book.cancel_order(4);
    assert_eq!(book.best_ask(), Some(201));
}

// ── 15. Spread calculation ─────────────────────────────────────────

#[test]
fn spread_correct_when_both_sides_exist() {
    let mut book = OrderBook::new();
    assert_eq!(book.spread(), None);

    book.add_limit_order(limit(1, Side::Bid, 99, 5, 1));
    assert_eq!(book.spread(), None); // no ask yet

    book.add_limit_order(limit(2, Side::Ask, 101, 5, 2));
    assert_eq!(book.spread(), Some(2));

    book.add_limit_order(limit(3, Side::Ask, 100, 5, 3));
    assert_eq!(book.spread(), Some(1));
}

// ── 16. Order at same price rests behind existing (FIFO) ───────────

#[test]
fn second_order_at_same_price_rests_behind_first() {
    let mut book = OrderBook::new();
    book.add_limit_order(limit(1, Side::Ask, 100, 5, 1));
    book.add_limit_order(limit(2, Side::Ask, 100, 5, 2));

    // Partial fill: only first order should be touched
    let res = book.add_limit_order(limit(3, Side::Bid, 100, 3, 3));
    assert_eq!(res.fills[0].maker_order_id, 1);
    assert_eq!(book.volume_at(100, Side::Ask), 7); // 2 from order 1, 5 from order 2
}

// ── 17. Exact price cross ──────────────────────────────────────────

#[test]
fn limit_buy_at_exactly_best_ask_matches() {
    let mut book = OrderBook::new();
    book.add_limit_order(limit(1, Side::Ask, 100, 10, 1));

    let res = book.add_limit_order(limit(2, Side::Bid, 100, 10, 2));
    assert_eq!(res.status, OrderStatus::Filled);
    assert_eq!(res.fills[0].price, 100);
}

// ── 18. Bid-ask inversion sweep ────────────────────────────────────

#[test]
fn limit_buy_well_above_ask_sweeps_all_affordable_levels() {
    let mut book = OrderBook::new();
    book.add_limit_order(limit(1, Side::Ask, 100, 2, 1));
    book.add_limit_order(limit(2, Side::Ask, 105, 2, 2));
    book.add_limit_order(limit(3, Side::Ask, 110, 2, 3));
    book.add_limit_order(limit(4, Side::Ask, 200, 2, 4));

    let res = book.add_limit_order(limit(10, Side::Bid, 150, 10, 5));
    // Should sweep 100, 105, 110 (total 6), skip 200, rest with 4
    assert_eq!(res.fills.len(), 3);
    assert_eq!(res.status, OrderStatus::Resting { remaining_qty: 4 });
    assert_eq!(book.best_ask(), Some(200));
    assert_eq!(book.best_bid(), Some(150));
}

// ── 19. Large quantity order ───────────────────────────────────────

#[test]
fn large_quantity_no_overflow() {
    let mut book = OrderBook::new();
    let large_qty = 1_000_000_000_u64;
    book.add_limit_order(limit(1, Side::Ask, 100, large_qty, 1));

    let res = book.add_limit_order(limit(2, Side::Bid, 100, large_qty, 2));
    assert_eq!(res.status, OrderStatus::Filled);
    assert_eq!(res.fills[0].qty, large_qty);
    assert_eq!(book.order_count(), 0);
}

// ── 20. Volume tracking ────────────────────────────────────────────

#[test]
fn volume_at_correct_after_adds_partial_fills_cancels() {
    let mut book = OrderBook::new();

    book.add_limit_order(limit(1, Side::Ask, 100, 10, 1));
    book.add_limit_order(limit(2, Side::Ask, 100, 20, 2));
    assert_eq!(book.volume_at(100, Side::Ask), 30);

    // Partial fill: consume 15
    book.add_limit_order(limit(3, Side::Bid, 100, 15, 3));
    // Order 1 fully filled (10), order 2 partially filled (5 of 20)
    assert_eq!(book.volume_at(100, Side::Ask), 15);

    // Cancel remaining order 2
    book.cancel_order(2);
    assert_eq!(book.volume_at(100, Side::Ask), 0);

    // Empty level returns 0
    assert_eq!(book.volume_at(999, Side::Bid), 0);
}
