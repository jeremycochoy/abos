use crate::latency::LatencyConfig;
use crate::types::Symbol;

/// Top-level simulation configuration.
#[derive(Debug, Clone)]
pub struct SimulationConfig {
    pub seed: u64,
    pub start_time: u64,
    pub end_time: u64,
    pub symbols: Vec<Symbol>,
    pub latency: LatencyConfig,
    pub tick_size: i64,
    pub lot_size: u64,
}
