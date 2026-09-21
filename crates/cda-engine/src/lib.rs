#![deny(clippy::all)]
#![warn(clippy::pedantic)]

//! High-performance Continuous Double Auction (CDA) engine.
//!
//! Provides a price-time priority order book supporting limit and market orders
//! with O(log N) placement and O(1) cancellation. Designed as the core matching
//! engine for an agent-based market simulator.

pub mod fasthash;
mod fills;
mod order;
mod orderbook;

pub use fills::{Fill, OrderResult, OrderStatus};
pub use order::{LimitOrder, MarketOrder, Side};
pub use orderbook::OrderBook;
