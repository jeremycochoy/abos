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
    // ZI agent traits + defaults
    FixedIntervalWakeup, LogNormalPriceSampler, LogNormalSizeSampler, OrderSizeSampler,
    PoissonWakeup, PriceSampler, WakeupSampler,
    // Trend-following agent traits + defaults
    OffsetPriceSampler, ProportionalSizeSampler, TrendPriceSampler, TrendSizeSampler,
    // Market-maker agent traits + defaults
    LiquidityWeightModel, SymmetricHumpModel,
};
pub use trend_following_agent::{TrendFollowingAgent, TrendFollowingConfig};
pub use zi_agent::{ZiAgent, ZiAgentConfig};
