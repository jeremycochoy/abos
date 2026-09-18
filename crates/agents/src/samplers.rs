//! Pluggable sampling strategies for agent order placement and wakeup timing.
//!
//! ## ZI agent traits
//! - [`PriceSampler`] — order prices around mid-price
//! - [`OrderSizeSampler`] — order quantities
//! - [`WakeupSampler`] — inter-wakeup delays (shared with all agent types)
//!
//! ## Trend-following / contrarian agent traits
//! - [`TrendPriceSampler`] — side-aware order pricing
//! - [`TrendSizeSampler`] — signal-proportional order sizing
//!
//! ## Market-maker agent traits
//! - [`LiquidityWeightModel`] — per-level liquidity weight function
//!
//! Each trait receives `&mut SmallRng` from the agent, keeping a single RNG
//! stream per agent for reproducibility.

use rand::rngs::SmallRng;
use rand_distr::{Distribution, Exp, Normal};

use cda_engine::Side;

// ═══════════════════════════════════════════════════════════════════
// ZI agent traits
// ═══════════════════════════════════════════════════════════════════

/// Samples an order price given the current mid-price.
pub trait PriceSampler {
    fn sample_price(&mut self, mid: i64, rng: &mut SmallRng) -> i64;
}

/// Samples an order quantity.
pub trait OrderSizeSampler {
    fn sample_order_size(&mut self, rng: &mut SmallRng) -> u64;
}

/// Samples a wakeup delay in nanoseconds. Used by all agent types.
pub trait WakeupSampler {
    fn sample_wakeup_delay(&mut self, rng: &mut SmallRng) -> u64;
}

// ═══════════════════════════════════════════════════════════════════
// Trend-following / contrarian agent traits
// ═══════════════════════════════════════════════════════════════════

/// Side-aware price sampler for trend agents.
/// Given a mid-price and trade side, returns the limit order price.
pub trait TrendPriceSampler {
    fn sample_price(&mut self, mid: i64, side: Side, rng: &mut SmallRng) -> i64;
}

/// Signal-proportional order size sampler for trend agents.
/// `ma_diff` is `ln(short_ma) - ln(long_ma)`.
pub trait TrendSizeSampler {
    fn sample_order_size(&mut self, ma_diff: f64, rng: &mut SmallRng) -> u64;
}

// ═══════════════════════════════════════════════════════════════════
// Market-maker agent traits
// ═══════════════════════════════════════════════════════════════════

/// Per-level liquidity weight function for market-maker agents.
/// Returns the unnormalized weight at a given log-distance from mid-price.
pub trait LiquidityWeightModel {
    fn weight(&mut self, log_distance: f64, rng: &mut SmallRng) -> f64;
}

// ═══════════════════════════════════════════════════════════════════
// Default implementations — ZI
// ═══════════════════════════════════════════════════════════════════

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

// ═══════════════════════════════════════════════════════════════════
// Default implementations — Trend-following
// ═══════════════════════════════════════════════════════════════════

/// Fixed-offset price sampler for trend agents.
/// Bid: `mid * (1 + offset)`, Ask: `mid * (1 - offset)`.
pub struct OffsetPriceSampler {
    offset: f64,
}

impl OffsetPriceSampler {
    #[must_use]
    pub fn new(offset: f64) -> Self {
        Self { offset }
    }
}

impl TrendPriceSampler for OffsetPriceSampler {
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss, clippy::cast_precision_loss)]
    fn sample_price(&mut self, mid: i64, side: Side, _rng: &mut SmallRng) -> i64 {
        match side {
            Side::Bid => ((mid as f64 * (1.0 + self.offset)).round() as i64).max(1),
            Side::Ask => ((mid as f64 * (1.0 - self.offset)).round() as i64).max(1),
        }
    }
}

/// Proportional order size sampler for trend agents.
/// `size = factor * |ma_diff| + boost`, clamped to minimum 1.
pub struct ProportionalSizeSampler {
    factor: f64,
    boost: f64,
}

impl ProportionalSizeSampler {
    #[must_use]
    pub fn new(factor: f64, boost: f64) -> Self {
        Self { factor, boost }
    }
}

impl TrendSizeSampler for ProportionalSizeSampler {
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    fn sample_order_size(&mut self, ma_diff: f64, _rng: &mut SmallRng) -> u64 {
        ((self.factor * ma_diff.abs() + self.boost).round() as u64).max(1)
    }
}

// ═══════════════════════════════════════════════════════════════════
// Default implementations — Market-maker
// ═══════════════════════════════════════════════════════════════════

/// Symmetric-hump liquidity weight model.
/// `w = (d + ε)^exponent * exp(-decay * d)` where
/// `decay = exponent / (ln(1 + peak_distance_ratio) + ε)`.
pub struct SymmetricHumpModel {
    peak_distance_ratio: f64,
    shape_exponent: f64,
}

impl SymmetricHumpModel {
    #[must_use]
    pub fn new(peak_distance_ratio: f64, shape_exponent: f64) -> Self {
        Self {
            peak_distance_ratio,
            shape_exponent,
        }
    }
}

impl LiquidityWeightModel for SymmetricHumpModel {
    fn weight(&mut self, log_distance: f64, _rng: &mut SmallRng) -> f64 {
        let eps = 4e-4;
        let peak_fraction = (1.0 + self.peak_distance_ratio).ln();
        let decay_rate = self.shape_exponent / (peak_fraction + eps);
        let d = log_distance.abs() + eps;
        d.powf(self.shape_exponent) * (-decay_rate * log_distance.abs()).exp()
    }
}

#[cfg(test)]
#[allow(clippy::cast_precision_loss, clippy::cast_lossless)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    // ── ZI sampler tests ────────────────────────────────────────────

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
    #[allow(clippy::float_cmp)]
    fn lognormal_price_draw_has_zero_mean() {
        for price_std in [0.0003, 0.003, 0.1, 1.0] {
            let sampler = LogNormalPriceSampler::new(price_std);
            assert_eq!(sampler.normal.mean(), 0.0, "price_std {price_std}");
            assert_eq!(sampler.normal.std_dev(), (1.0 + price_std).ln(), "price_std {price_std}");
        }
    }

    /// Half of the quotes land below the mid. The old mean of the draw,
    /// -σ²/2, put 64 % of them below the mid at σ = ln 2.
    #[test]
    fn lognormal_price_log_centers_on_mid() {
        let mut sampler = LogNormalPriceSampler::new(1.0);
        let mut rng = SmallRng::seed_from_u64(42);
        let mid: i64 = 10_000_000;
        let n = 50_000;
        let below = (0..n).filter(|_| sampler.sample_price(mid, &mut rng) < mid).count();
        let share = below as f64 / f64::from(n);
        assert!((share - 0.5).abs() < 0.01, "{share:.3} of the quotes land below the mid");
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

    // ── Trend sampler tests ─────────────────────────────────────────

    #[test]
    fn offset_price_correct() {
        let mut sampler = OffsetPriceSampler::new(0.02);
        let mut rng = SmallRng::seed_from_u64(42);
        let mid = 10_000;
        assert_eq!(sampler.sample_price(mid, Side::Bid, &mut rng), 10_200);
        assert_eq!(sampler.sample_price(mid, Side::Ask, &mut rng), 9_800);
    }

    #[test]
    fn proportional_size_correct() {
        let mut sampler = ProportionalSizeSampler::new(1000.0, 100.0);
        let mut rng = SmallRng::seed_from_u64(42);
        // 1000 * 0.005 + 100 = 105
        assert_eq!(sampler.sample_order_size(0.005, &mut rng), 105);
        // 1000 * 0.02 + 100 = 120
        assert_eq!(sampler.sample_order_size(0.02, &mut rng), 120);
    }

    #[test]
    fn proportional_size_minimum_one() {
        let mut sampler = ProportionalSizeSampler::new(1.0, 0.0);
        let mut rng = SmallRng::seed_from_u64(42);
        // 1.0 * 0.0001 + 0.0 = 0.0001 → rounds to 0 → clamped to 1
        assert_eq!(sampler.sample_order_size(0.0001, &mut rng), 1);
    }

    // ── Market-maker weight model tests ─────────────────────────────

    #[test]
    fn symmetric_hump_positive_for_nonzero_distance() {
        let mut model = SymmetricHumpModel::new(0.05, 1.2);
        let mut rng = SmallRng::seed_from_u64(42);
        for i in 1..=10 {
            let d = 0.001 * f64::from(i);
            let w = model.weight(d, &mut rng);
            assert!(w > 0.0, "weight({d}) = {w}, expected > 0");
        }
    }

    #[test]
    fn symmetric_hump_peaks_then_decays() {
        let mut model = SymmetricHumpModel::new(0.05, 1.2);
        let mut rng = SmallRng::seed_from_u64(42);
        let near = model.weight(0.001, &mut rng);
        let mid_d = model.weight(0.05, &mut rng);
        let far = model.weight(0.5, &mut rng);
        assert!(
            mid_d > near,
            "weight at peak distance ({mid_d:.6}) should exceed weight near zero ({near:.6})"
        );
        assert!(
            mid_d > far,
            "weight at peak ({mid_d:.6}) should exceed weight far out ({far:.6})"
        );
    }

    #[test]
    fn symmetric_hump_is_symmetric() {
        let mut model = SymmetricHumpModel::new(0.05, 1.2);
        let mut rng = SmallRng::seed_from_u64(42);
        let positive = model.weight(0.02, &mut rng);
        let negative = model.weight(-0.02, &mut rng);
        assert!(
            (positive - negative).abs() < 1e-15,
            "hump weight should be symmetric: {positive} vs {negative}"
        );
    }
}
