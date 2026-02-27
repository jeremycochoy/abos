#![deny(clippy::all)]
#![warn(clippy::pedantic)]

//! Agent implementations for agent-based market simulation.

pub mod market_maker_agent;
pub mod samplers;
pub mod trend_following_agent;
mod utils;
pub mod zi_agent;

pub use market_maker_agent::{MarketMakerAgent, MarketMakerConfig};
pub use samplers::{
    FixedIntervalWakeup, LogNormalPriceSampler, LogNormalSizeSampler, OrderSizeSampler,
    PoissonWakeup, PriceSampler, WakeupSampler,
};
pub use trend_following_agent::{TrendFollowingAgent, TrendFollowingConfig};
pub use zi_agent::{ZiAgent, ZiAgentConfig};
