use cda_engine::{LimitOrder, OrderBook, Side};

fn limit(id: u64, side: Side, price: i64, qty: u64) -> LimitOrder {
    LimitOrder {
        id,
        side,
        price,
        qty,
        timestamp: 0,
    }
}

fn book_with_asks_and_bids() -> OrderBook {
    let mut book = OrderBook::new();
    book.add_limit_order(limit(1, Side::Ask, 101, 10));
    book.add_limit_order(limit(2, Side::Ask, 102, 20));
    book.add_limit_order(limit(3, Side::Ask, 105, 40));
    book.add_limit_order(limit(4, Side::Bid, 99, 5));
    book.add_limit_order(limit(5, Side::Bid, 98, 15));
    book.add_limit_order(limit(6, Side::Bid, 95, 25));
    book
}

#[test]
fn a_market_order_sees_the_whole_opposing_side() {
    let book = book_with_asks_and_bids();
    assert_eq!(
        book.executable_depth(Side::Bid, None),
        (70, 101 * 10 + 102 * 20 + 105 * 40)
    );
    assert_eq!(
        book.executable_depth(Side::Ask, None),
        (45, 99 * 5 + 98 * 15 + 95 * 25)
    );
}

#[test]
fn a_limit_order_sees_the_levels_inside_its_price_boundary_included() {
    let book = book_with_asks_and_bids();
    assert_eq!(
        book.executable_depth(Side::Bid, Some(102)),
        (30, 101 * 10 + 102 * 20)
    );
    assert_eq!(book.executable_depth(Side::Bid, Some(101)), (10, 101 * 10));
    assert_eq!(book.executable_depth(Side::Bid, Some(100)), (0, 0));
    assert_eq!(
        book.executable_depth(Side::Ask, Some(98)),
        (20, 99 * 5 + 98 * 15)
    );
    assert_eq!(book.executable_depth(Side::Ask, Some(99)), (5, 99 * 5));
    assert_eq!(book.executable_depth(Side::Ask, Some(100)), (0, 0));
}

#[test]
fn an_empty_opposing_side_gives_zero_depth() {
    let mut book = OrderBook::new();
    book.add_limit_order(limit(1, Side::Bid, 99, 5));
    assert_eq!(book.executable_depth(Side::Bid, None), (0, 0));
    assert_eq!(book.executable_depth(Side::Bid, Some(1_000)), (0, 0));
}

#[test]
fn a_cancelled_order_leaves_the_executable_depth() {
    let mut book = book_with_asks_and_bids();
    assert!(book.cancel_order(2));
    assert_eq!(book.executable_depth(Side::Bid, Some(102)), (10, 101 * 10));
    assert_eq!(
        book.executable_depth(Side::Bid, None),
        (50, 101 * 10 + 105 * 40)
    );
    assert!(book.cancel_order(4));
    assert_eq!(
        book.executable_depth(Side::Ask, Some(98)),
        (15, 98 * 15)
    );
}

#[test]
fn a_partial_fill_leaves_the_remainder_in_the_depth() {
    let mut book = book_with_asks_and_bids();
    book.add_limit_order(limit(7, Side::Bid, 101, 4));
    assert_eq!(book.executable_depth(Side::Bid, Some(101)), (6, 101 * 6));
    assert_eq!(
        book.executable_depth(Side::Bid, None),
        (66, 101 * 6 + 102 * 20 + 105 * 40)
    );
}
