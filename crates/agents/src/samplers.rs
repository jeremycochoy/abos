//! Pluggable sampling strategies for agent order placement and wakeup timing.
//!
//! Three traits define the sampling interfaces:
//! - [`PriceSampler`] — how order prices are chosen around the mid-price
//! - [`OrderSizeSampler`] — how order quantities are determined
//! - [`WakeupSampler`] — how inter-wakeup delays are drawn
//!
//! Each trait receives `&mut SmallRng` from the agent, keeping a single RNG
//! stream per agent for reproducibility.

use rand::rngs::SmallRng;
use rand_distr::{Distribution, Exp, Normal};

/// Samples an order price given the current mid-price.
pub trait PriceSampler {
    fn sample_price(&mut self, mid: i64, rng: &mut SmallRng) -> i64;
}

/// Samples an order quantity.
pub trait OrderSizeSampler {
    fn sample_order_size(&mut self, rng: &mut SmallRng) -> u64;
}

/// Samples a wakeup delay in nanoseconds.
pub trait WakeupSampler {
    fn sample_wakeup_delay(&mut self, rng: &mut SmallRng) -> u64;
}

// ── Default implementations ─────────────────────────────────────

/// Log-normal price sampler: `price = mid * exp(N(-σ²/2, σ))` where `σ = ln(1 + price_std)`.
pub struct LogNormalPriceSampler {
    normal: Normal<f64>,
}

impl LogNormalPriceSampler {
    #[must_use]
    pub fn new(price_std: f64) -> Self {
        let log_std = (1.0 + price_std).ln();
        Self {
            normal: Normal::new(-0.5 * log_std * log_std, log_std)
                .expect("invalid price distribution params"),
        }
    }
}

impl PriceSampler for LogNormalPriceSampler {
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        clippy::cast_precision_loss
    )]
    fn sample_price(&mut self, mid: i64, rng: &mut SmallRng) -> i64 {
        let log_return: f64 = self.normal.sample(rng);
        ((mid as f64 * log_return.exp()).round() as i64).max(1)
    }
}

/// Log-normal order size sampler: `size = exp(N(0, std)) * scale`.
pub struct LogNormalSizeSampler {
    normal: Normal<f64>,
    scale: f64,
}

impl LogNormalSizeSampler {
    #[must_use]
    pub fn new(scale: f64, std: f64) -> Self {
        Self {
            normal: Normal::new(0.0, std).expect("invalid order size std"),
            scale,
        }
    }
}

impl OrderSizeSampler for LogNormalSizeSampler {
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    fn sample_order_size(&mut self, rng: &mut SmallRng) -> u64 {
        let normal: f64 = self.normal.sample(rng);
        ((normal.exp() * self.scale).round() as u64).max(1)
    }
}

/// Fixed-interval wakeup sampler (deterministic, ignores RNG).
pub struct FixedIntervalWakeup {
    interval_ns: u64,
}

impl FixedIntervalWakeup {
    #[must_use]
    pub const fn new(interval_ns: u64) -> Self {
        Self { interval_ns }
    }
}

impl WakeupSampler for FixedIntervalWakeup {
    fn sample_wakeup_delay(&mut self, _rng: &mut SmallRng) -> u64 {
        self.interval_ns
    }
}

/// Poisson (exponential inter-arrival) wakeup sampler.
pub struct PoissonWakeup {
    dist: Exp<f64>,
}

impl PoissonWakeup {
    #[must_use]
    pub fn new(mean_interval_ns: u64) -> Self {
        #[allow(clippy::cast_precision_loss)]
        let dist =
            Exp::new(1.0 / mean_interval_ns as f64).expect("invalid mean wakeup interval");
        Self { dist }
    }
}

impl WakeupSampler for PoissonWakeup {
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    fn sample_wakeup_delay(&mut self, rng: &mut SmallRng) -> u64 {
        let delay: f64 = self.dist.sample(rng);
        (delay.round() as u64).max(1)
    }
}

#[cfg(test)]
#[allow(clippy::cast_precision_loss, clippy::cast_lossless)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    #[test]
    fn lognormal_price_centered_on_mid() {
        let mut sampler = LogNormalPriceSampler::new(0.03 / 100.0);
        let mut rng = SmallRng::seed_from_u64(42);
        let mid: i64 = 10_000_000;
        let n = 50_000;
        let sum: f64 = (0..n)
            .map(|_| sampler.sample_price(mid, &mut rng) as f64)
            .sum();
        let mean = sum / n as f64;
        let pct_deviation = ((mean - mid as f64) / mid as f64).abs();
        assert!(
            pct_deviation < 0.005,
            "mean price {mean:.0} deviates {:.3}% from mid {mid} (expected < 0.5%)",
            pct_deviation * 100.0
        );
    }

    #[test]
    fn lognormal_price_std_matches_config() {
        let mut sampler = LogNormalPriceSampler::new(0.03 / 100.0);
        let mut rng = SmallRng::seed_from_u64(42);
        let mid: i64 = 10_000_000;
        let n = 50_000;
        let samples: Vec<f64> = (0..n)
            .map(|_| sampler.sample_price(mid, &mut rng) as f64)
            .collect();
        let mean: f64 = samples.iter().sum::<f64>() / n as f64;
        let variance: f64 =
            samples.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / n as f64;
        let relative_std = variance.sqrt() / mean;
        assert!(
            relative_std < 0.001,
            "relative std {relative_std:.6} too large for price_std=0.0003"
        );
        assert!(
            relative_std > 0.0001,
            "relative std {relative_std:.6} too small for price_std=0.0003"
        );
    }

    #[test]
    fn lognormal_size_distribution_mean() {
        let mut sampler = LogNormalSizeSampler::new(12_000.0, 1.7);
        let mut rng = SmallRng::seed_from_u64(42);
        let n = 50_000;
        let sum: f64 = (0..n)
            .map(|_| sampler.sample_order_size(&mut rng) as f64)
            .sum();
        let mean = sum / n as f64;
        let expected = 12_000.0 * (1.7_f64.powi(2) / 2.0).exp();
        let ratio = mean / expected;
        assert!(
            (0.7..1.4).contains(&ratio),
            "mean size {mean:.0} vs expected {expected:.0} (ratio {ratio:.2})"
        );
    }

    #[test]
    fn lognormal_size_always_at_least_one() {
        let mut sampler = LogNormalSizeSampler::new(12_000.0, 1.7);
        let mut rng = SmallRng::seed_from_u64(42);
        for _ in 0..10_000 {
            assert!(sampler.sample_order_size(&mut rng) >= 1);
        }
    }

    #[test]
    fn fixed_interval_returns_constant() {
        let mut sampler = FixedIntervalWakeup::new(1_000_000_000);
        let mut rng = SmallRng::seed_from_u64(42);
        for _ in 0..100 {
            assert_eq!(sampler.sample_wakeup_delay(&mut rng), 1_000_000_000);
        }
    }

    #[test]
    fn poisson_wakeup_mean_reasonable() {
        let mean_ns: u64 = 1_000_000_000;
        let mut sampler = PoissonWakeup::new(mean_ns);
        let mut rng = SmallRng::seed_from_u64(42);
        let n = 50_000;
        let sum: f64 = (0..n)
            .map(|_| sampler.sample_wakeup_delay(&mut rng) as f64)
            .sum();
        let mean = sum / n as f64;
        let ratio = mean / mean_ns as f64;
        assert!(
            (0.9..1.1).contains(&ratio),
            "mean delay {mean:.0} vs expected {mean_ns} (ratio {ratio:.2})"
        );
    }
}
