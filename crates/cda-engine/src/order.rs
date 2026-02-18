/// Which side of the book an order is on.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Side {
    Bid,
    Ask,
}

/// A limit order submitted to the book.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LimitOrder {
    pub id: u64,
    pub side: Side,
    pub price: i64,
    pub qty: u64,
    pub timestamp: u64,
}

/// A market order that executes immediately and never rests.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MarketOrder {
    pub id: u64,
    pub side: Side,
    pub qty: u64,
}

/// An order resting on the book (always originated as a limit order).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct RestingOrder {
    pub id: u64,
    pub qty: u64,
    pub timestamp: u64,
}
