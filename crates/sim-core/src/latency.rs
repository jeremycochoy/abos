use rand::Rng;
use rand_distr::{Distribution, LogNormal};

use crate::types::AgentId;

/// Configuration for the latency model.
#[derive(Debug, Clone)]
pub struct LatencyConfig {
    pub default_base_ns: u64,
    pub jitter_mu: f64,
    pub jitter_sigma: f64,
}

/// Latency model with per-agent base latency and log-normal jitter.
pub struct LatencyModel {
    base_latencies: Vec<u64>,
    jitter: LogNormal<f64>,
}

impl LatencyModel {
    /// Build a latency model from config and agent count.
    ///
    /// # Panics
    /// Panics if `jitter_mu` / `jitter_sigma` produce invalid log-normal params.
    #[must_use]
    pub fn new(config: &LatencyConfig, num_agents: usize) -> Self {
        Self {
            base_latencies: vec![config.default_base_ns; num_agents],
            jitter: LogNormal::new(config.jitter_mu, config.jitter_sigma)
                .expect("invalid log-normal params"),
        }
    }

    /// Build with per-agent base latencies.
    ///
    /// # Panics
    /// Panics if `jitter_mu` / `jitter_sigma` produce invalid log-normal params.
    #[must_use]
    pub fn with_bases(bases: Vec<u64>, config: &LatencyConfig) -> Self {
        Self {
            base_latencies: bases,
            jitter: LogNormal::new(config.jitter_mu, config.jitter_sigma)
                .expect("invalid log-normal params"),
        }
    }

    /// Latency from agent to exchange (one-way).
    pub fn agent_to_exchange(&self, agent_id: AgentId, rng: &mut impl Rng) -> u64 {
        self.sample(agent_id, rng)
    }

    /// Latency from exchange to agent (one-way, independently sampled).
    pub fn exchange_to_agent(&self, agent_id: AgentId, rng: &mut impl Rng) -> u64 {
        self.sample(agent_id, rng)
    }

    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss, clippy::cast_precision_loss)]
    fn sample(&self, agent_id: AgentId, rng: &mut impl Rng) -> u64 {
        let base = self.base_latencies[agent_id];
        let jitter: f64 = self.jitter.sample(rng);
        let total = (base as f64 * jitter).round() as u64;
        total.max(1)
    }
}
