/// Agent identifier (index into the kernel's agent vector).
pub type AgentId = usize;

/// Symbol identifier.
pub type Symbol = u32;

/// Nanosecond timestamp.
pub type Nanos = u64;

/// Snapshot of the top-of-book for a single symbol.
#[derive(Debug, Clone, Copy)]
pub struct MarketSnapshot {
    pub best_bid: Option<(i64, u64)>,
    pub best_ask: Option<(i64, u64)>,
    pub last_trade_price: Option<i64>,
    pub last_trade_time: Option<Nanos>,
}
