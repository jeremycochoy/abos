use crate::order::Side;

/// A single fill (trade execution) between a taker and a resting maker order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Fill {
    pub maker_order_id: u64,
    pub taker_order_id: u64,
    pub price: i64,
    pub qty: u64,
    /// Side of the taker (aggressor).
    pub taker_side: Side,
}

/// What happened to the incoming order after submission.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OrderStatus {
    /// Fully filled — no quantity remains.
    Filled,
    /// Partially filled then resting on the book with `remaining_qty`.
    Resting { remaining_qty: u64 },
    /// Placed on the book with no fills (limit order that didn't cross).
    Placed,
    /// Unfilled or partially filled market order — remainder was cancelled.
    Cancelled { filled_qty: u64 },
}

/// Result of submitting an order to the book.
///
/// Fills are stored in an internal buffer owned by the `OrderBook`. This struct
/// provides owned access to the fills via a drained `Vec`. The `OrderBook`
/// retains the allocation for reuse on the next call.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OrderResult {
    pub fills: Vec<Fill>,
    pub status: OrderStatus,
}
