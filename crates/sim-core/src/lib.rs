#![deny(clippy::all)]
#![warn(clippy::pedantic)]

//! Discrete-event simulation kernel for agent-based market simulation.
//!
//! Provides the event loop, exchange wrapper, agent trait, and latency model
//! needed to run a complete market simulation using `cda-engine` as the
//! matching engine.

pub mod agent;
pub mod config;
pub mod event;
pub mod exchange;
pub mod kernel;
pub mod latency;
pub mod output;
pub mod types;

pub use agent::{Agent, AgentAction};
pub use config::SimulationConfig;
pub use event::{ExchangeMessage, OrderAction};
pub use exchange::{Exchange, FlowBucket, FlowOptions, L1Bucket, L1Snapshot, TradeRecord};
pub use kernel::{Kernel, RunOptions, SimulationResult};
pub use latency::{LatencyConfig, LatencyModel, LatencyModelType};
pub use output::Candle;
pub use types::{AgentId, MarketSnapshot, Nanos, Symbol};
