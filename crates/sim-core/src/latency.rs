use rand::Rng;
use rand_distr::{Distribution, LogNormal};

use crate::types::AgentId;

/// Speed of light in m/ns (approximately 0.2998 m/ns).
const LIGHT_SPEED_M_PER_NS: f64 = 0.299_792_458;

/// NYC-to-Seattle distance in meters (~3,867 km).
const NYC_SEATTLE_METERS: f64 = 3_866_660.0;

/// Configuration for the latency model.
#[derive(Debug, Clone)]
pub struct LatencyConfig {
    /// Default one-way base latency in nanoseconds (used when `model` is
    /// `Uniform`).
    pub default_base_ns: u64,
    /// Log-normal jitter mean parameter (mu).
    pub jitter_mu: f64,
    /// Log-normal jitter standard deviation parameter (sigma).
    pub jitter_sigma: f64,
    /// Which latency model to use.
    pub model: LatencyModelType,
}

/// Type of latency model.
#[derive(Debug, Clone, Default)]
pub enum LatencyModelType {
    /// Every agent has the same base latency (the default).
    #[default]
    Uniform,
    /// Agents placed uniformly on a NYC-Seattle line. Base latency is
    /// proportional to distance divided by the speed of light.
    NycSeattle { seed: u64 },
    /// No latency at all (zero delay).
    NoLatency,
    /// Cubic jitter model matching ABIDES.
    /// `jitter` (a), `jitter_clip`, `jitter_unit` are the cubic params.
    Cubic { jitter: f64, jitter_clip: f64, jitter_unit: f64 },
}

/// Latency model with per-agent base latency and configurable jitter.
pub struct LatencyModel {
    base_latencies: Vec<u64>,
    jitter_mode: JitterMode,
}

enum JitterMode {
    LogNormal(LogNormal<f64>),
    Cubic { jitter: f64, jitter_clip: f64, jitter_unit: f64 },
    None,
}

impl LatencyModel {
    /// Build a latency model from config and agent count.
    ///
    /// # Panics
    /// Panics if log-normal parameters are invalid.
    #[must_use]
    pub fn new(config: &LatencyConfig, num_agents: usize) -> Self {
        let base_latencies = match &config.model {
            LatencyModelType::NycSeattle { seed } => {
                generate_nyc_seattle_bases(*seed, num_agents)
            }
            LatencyModelType::NoLatency => vec![0; num_agents],
            LatencyModelType::Uniform | LatencyModelType::Cubic { .. } => {
                vec![config.default_base_ns; num_agents]
            }
        };
        let jitter_mode = match &config.model {
            LatencyModelType::NoLatency => JitterMode::None,
            LatencyModelType::Cubic { jitter, jitter_clip, jitter_unit } => {
                JitterMode::Cubic {
                    jitter: *jitter,
                    jitter_clip: *jitter_clip,
                    jitter_unit: *jitter_unit,
                }
            }
            _ => {
                if config.jitter_sigma.abs() < 1e-15 {
                    JitterMode::None
                } else {
                    JitterMode::LogNormal(
                        LogNormal::new(config.jitter_mu, config.jitter_sigma)
                            .expect("invalid log-normal params"),
                    )
                }
            }
        };
        Self { base_latencies, jitter_mode }
    }

    /// Build with per-agent base latencies.
    ///
    /// # Panics
    /// Panics if log-normal parameters are invalid.
    #[must_use]
    pub fn with_bases(bases: Vec<u64>, config: &LatencyConfig) -> Self {
        let jitter_mode = match &config.model {
            LatencyModelType::NoLatency => JitterMode::None,
            LatencyModelType::Cubic { jitter, jitter_clip, jitter_unit } => {
                JitterMode::Cubic {
                    jitter: *jitter,
                    jitter_clip: *jitter_clip,
                    jitter_unit: *jitter_unit,
                }
            }
            _ => {
                JitterMode::LogNormal(
                    LogNormal::new(config.jitter_mu, config.jitter_sigma)
                        .expect("invalid log-normal params"),
                )
            }
        };
        Self { base_latencies: bases, jitter_mode }
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
        match &self.jitter_mode {
            JitterMode::None => base,
            JitterMode::LogNormal(dist) => {
                let j: f64 = dist.sample(rng);
                let total = (base as f64 * j).round() as u64;
                total.max(1)
            }
            JitterMode::Cubic { jitter, jitter_clip, jitter_unit } => {
                // ABIDES cubic model: latency = base + (a / x^3) * (base / unit)
                let x: f64 = rng.gen_range(*jitter_clip..1.0_f64.max(*jitter_clip + 1e-12));
                let cubic = jitter / (x * x * x);
                let extra = cubic * (base as f64 / jitter_unit);
                ((base as f64 + extra).round() as u64).max(1)
            }
        }
    }
}

/// Generate per-agent base latencies by placing agents uniformly on a
/// NYC-Seattle line and computing light-speed one-way delays.
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss, clippy::cast_precision_loss)]
fn generate_nyc_seattle_bases(seed: u64, num_agents: usize) -> Vec<u64> {
    use rand::SeedableRng;
    use rand::rngs::SmallRng;

    let mut rng = SmallRng::seed_from_u64(seed);
    // Each agent gets a random position on the line [0, NYC_SEATTLE_METERS].
    // The exchange is at position 0 (NYC). Latency = distance / c.
    let positions: Vec<f64> = (0..num_agents)
        .map(|_| rng.gen::<f64>() * NYC_SEATTLE_METERS)
        .collect();
    positions
        .iter()
        .map(|&pos| {
            let ns = pos / LIGHT_SPEED_M_PER_NS;
            (ns.round() as u64).max(1)
        })
        .collect()
}
