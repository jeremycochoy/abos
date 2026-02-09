#![deny(clippy::all)]
#![warn(clippy::pedantic)]

//! Agent implementations for agent-based market simulation.

pub mod zi_agent;

pub use zi_agent::{ZiAgent, ZiAgentConfig};
