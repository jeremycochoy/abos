use rand::rngs::SmallRng;
use rand::Rng;
use rand::SeedableRng;
use rand_distr::{Distribution, Normal};

use cda_engine::Side;
use sim_core::{Agent, AgentAction, AgentId, ExchangeMessage, MarketSnapshot, Nanos, OrderAction};

use crate::utils::{IndexedSet, mid_price};

/// Configuration for a zero-intelligence agent (ABIDES-compatible).
#[derive(Debug, Clone)]
pub struct ZiAgentConfig {
    /// Wakeup interval in nanoseconds (fixed, not exponential).
    pub wake_up_interval_ns: u64,
    /// Log-normal price std (relative to mid-price, e.g. 0.0003 = 0.03%).
    pub price_std: f64,
    /// Scale for lognormal order size distribution.
    pub order_size_scale: f64,
    /// Std for lognormal order size distribution.
    pub order_size_std: f64,
    /// Reference price when book is empty (internal integer units).
    pub reference_price: i64,
    /// Symbol index to trade (0-based).
    pub symbol: u32,
}

/// Zero-intelligence agent matching the ABIDES implementation.
///
/// On each wakeup the agent:
/// 1. Cancels all outstanding orders
/// 2. Picks a random side (buy/sell 50/50)
/// 3. Samples order size from a lognormal distribution
/// 4. Samples price from a lognormal distribution around the mid-price
/// 5. Places a single limit order
/// 6. Schedules next wakeup at a fixed interval
pub struct ZiAgent {
    cfg: ZiAgentConfig,
    rng: SmallRng,
    resting_orders: IndexedSet,
    price_normal: Normal<f64>,
    size_normal: Normal<f64>,
}

impl ZiAgent {
    /// Create a new ZI agent with the given config and RNG seed.
    ///
    /// # Panics
    /// Panics if `price_std` produces invalid distribution params.
    #[must_use]
    pub fn new(config: ZiAgentConfig, seed: u64) -> Self {
        let log_std = (1.0 + config.price_std).ln();
        let price_normal = Normal::new(-0.5 * log_std * log_std, log_std)
            .expect("invalid price distribution params");
        let size_normal = Normal::new(0.0, config.order_size_std)
            .expect("invalid order size std");
        Self {
            cfg: config,
            rng: SmallRng::seed_from_u64(seed),
            resting_orders: IndexedSet::new(),
            price_normal,
            size_normal,
        }
    }

    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss, clippy::cast_precision_loss)]
    fn sample_price(&mut self, mid: i64) -> i64 {
        let log_return: f64 = self.price_normal.sample(&mut self.rng);
        let price = (mid as f64 * log_return.exp()).round() as i64;
        price.max(1)
    }

    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    fn sample_order_size(&mut self) -> u64 {
        let normal: f64 = self.size_normal.sample(&mut self.rng);
        let amount = (normal.exp() * self.cfg.order_size_scale).round() as u64;
        amount.max(1)
    }
}

impl Agent for ZiAgent {
    fn wakeup_into(
        &mut self,
        _time: Nanos,
        _agent_id: AgentId,
        snapshots: &[MarketSnapshot],
        actions: &mut Vec<AgentAction>,
    ) {
        let snap = &snapshots[self.cfg.symbol as usize];

        // Cancel all outstanding orders
        for oid in self.resting_orders.drain_all() {
            actions.push(AgentAction::CancelOrder {
                symbol: self.cfg.symbol,
                order_id: oid,
            });
        }

        // Pick random side
        let side = if self.rng.gen::<bool>() { Side::Bid } else { Side::Ask };

        // Sample order size and price
        let mid = mid_price(snap, self.cfg.reference_price);
        let qty = self.sample_order_size();
        let price = self.sample_price(mid);

        actions.push(AgentAction::SubmitOrder {
            symbol: self.cfg.symbol,
            order: OrderAction::NewLimitOrder { side, price, qty },
        });

        // Schedule next wakeup (fixed interval)
        actions.push(AgentAction::ScheduleWakeUp {
            delay_ns: self.cfg.wake_up_interval_ns,
        });
    }

    fn on_exchange_message(
        &mut self,
        _time: Nanos,
        _agent_id: AgentId,
        message: ExchangeMessage,
    ) {
        match message {
            ExchangeMessage::OrderAccepted { order_id } => {
                self.resting_orders.insert(order_id);
            }
            ExchangeMessage::OrderFilled { order_id, .. }
            | ExchangeMessage::OrderCancelled { order_id } => {
                self.resting_orders.remove(order_id);
            }
            ExchangeMessage::OrderRejected { .. } => {}
        }
    }
}

#[cfg(test)]
#[allow(clippy::cast_precision_loss, clippy::cast_lossless)]
mod tests {
    use super::*;

    fn default_cfg() -> ZiAgentConfig {
        ZiAgentConfig {
            wake_up_interval_ns: 30_000_000_000,
            price_std: 0.03 / 100.0, // 0.03%
            order_size_scale: 12_000.0,
            order_size_std: 1.7,
            reference_price: 10_000_000,
            symbol: 0,
        }
    }

    fn snapshot_with_mid(bid: i64, ask: i64) -> Vec<MarketSnapshot> {
        vec![MarketSnapshot {
            best_bid: Some((bid, 100)),
            best_ask: Some((ask, 100)),
            last_trade_price: Some(i64::midpoint(bid, ask)),
            last_trade_time: Some(0),
        }]
    }

    fn empty_snapshot() -> Vec<MarketSnapshot> {
        vec![MarketSnapshot {
            best_bid: None,
            best_ask: None,
            last_trade_price: None,
            last_trade_time: None,
        }]
    }

    // ── Price distribution matches ABIDES lognormal formula ──────────

    #[test]
    fn price_distribution_is_centered_on_mid() {
        // ABIDES: log_std = ln(1 + price_std)
        //         log_return = N(-0.5*log_std^2, log_std)
        //         price = round(mid * exp(log_return))
        // E[exp(log_return)] = 1.0 (mean-corrected), so mean price ≈ mid
        let mut agent = ZiAgent::new(default_cfg(), 42);
        let mid: i64 = 10_000_000;
        let n = 50_000;
        let sum: f64 = (0..n).map(|_| agent.sample_price(mid) as f64).sum();
        let mean = sum / n as f64;
        let pct_deviation = ((mean - mid as f64) / mid as f64).abs();
        assert!(
            pct_deviation < 0.005,
            "mean price {mean:.0} deviates {:.3}% from mid {mid} (expected < 0.5%)",
            pct_deviation * 100.0
        );
    }

    #[test]
    fn price_distribution_std_matches_config() {
        // With price_std = 0.03% (0.0003), log_std ≈ 0.0003
        // The relative std of sampled prices should be close to price_std
        let mut agent = ZiAgent::new(default_cfg(), 42);
        let mid: i64 = 10_000_000;
        let n = 50_000;
        let samples: Vec<f64> = (0..n).map(|_| agent.sample_price(mid) as f64).collect();
        let mean: f64 = samples.iter().sum::<f64>() / n as f64;
        let variance: f64 =
            samples.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / n as f64;
        let relative_std = variance.sqrt() / mean;
        // Should be close to 0.0003
        assert!(
            relative_std < 0.001,
            "relative std {relative_std:.6} too large for price_std=0.0003"
        );
        assert!(
            relative_std > 0.0001,
            "relative std {relative_std:.6} too small for price_std=0.0003"
        );
    }

    // ── Order size distribution matches ABIDES lognormal formula ─────

    #[test]
    fn order_size_distribution_mean() {
        // ABIDES: size = max(1, round(exp(N(0, std)) * scale))
        // E[exp(N(0,s))] = exp(s^2/2)
        // So expected mean ≈ scale * exp(std^2/2) = 12000 * exp(1.7^2/2) ≈ 12000 * 4.26 ≈ 51k
        let mut agent = ZiAgent::new(default_cfg(), 42);
        let n = 50_000;
        let sum: f64 = (0..n).map(|_| agent.sample_order_size() as f64).sum();
        let mean = sum / n as f64;
        let expected = 12_000.0 * (1.7_f64.powi(2) / 2.0).exp();
        let ratio = mean / expected;
        assert!(
            (0.7..1.4).contains(&ratio),
            "mean size {mean:.0} vs expected {expected:.0} (ratio {ratio:.2})"
        );
    }

    #[test]
    fn order_size_always_at_least_one() {
        let mut agent = ZiAgent::new(default_cfg(), 42);
        for _ in 0..10_000 {
            assert!(agent.sample_order_size() >= 1);
        }
    }

    // ── Cancel-all behavior ─────────────────────────────────────────

    #[test]
    fn wakeup_cancels_all_resting_orders() {
        let mut agent = ZiAgent::new(default_cfg(), 42);
        // Simulate 3 accepted orders
        agent.on_exchange_message(0, 0, ExchangeMessage::OrderAccepted { order_id: 100 });
        agent.on_exchange_message(0, 0, ExchangeMessage::OrderAccepted { order_id: 200 });
        agent.on_exchange_message(0, 0, ExchangeMessage::OrderAccepted { order_id: 300 });

        let snaps = snapshot_with_mid(9_999_000, 10_001_000);
        let mut actions = Vec::new();
        agent.wakeup_into(1_000, 0, &snaps, &mut actions);

        // Should have 3 cancels + 1 submit + 1 schedule = 5 actions
        let cancels: Vec<_> = actions
            .iter()
            .filter(|a| matches!(a, AgentAction::CancelOrder { .. }))
            .collect();
        assert_eq!(cancels.len(), 3, "should cancel all 3 resting orders");

        let mut cancelled_ids: Vec<u64> = cancels
            .iter()
            .map(|a| match a {
                AgentAction::CancelOrder { order_id, .. } => *order_id,
                _ => unreachable!(),
            })
            .collect();
        cancelled_ids.sort_unstable();
        assert_eq!(cancelled_ids, vec![100, 200, 300]);
    }

    // ── Exactly one limit order per wakeup ──────────────────────────

    #[test]
    fn wakeup_places_exactly_one_limit_order() {
        let mut agent = ZiAgent::new(default_cfg(), 42);
        let snaps = snapshot_with_mid(9_999_000, 10_001_000);
        for _ in 0..100 {
            let mut actions = Vec::new();
            agent.wakeup_into(0, 0, &snaps, &mut actions);
            let submits: Vec<_> = actions
                .iter()
                .filter(|a| matches!(a, AgentAction::SubmitOrder { .. }))
                .collect();
            assert_eq!(submits.len(), 1, "should place exactly 1 order");
            // Verify it's always a limit order (never market)
            for s in &submits {
                if let AgentAction::SubmitOrder { order, .. } = s {
                    assert!(
                        matches!(order, OrderAction::NewLimitOrder { .. }),
                        "ZI should only place limit orders"
                    );
                }
            }
        }
    }

    // ── Fixed-interval wakeup (not exponential) ─────────────────────

    #[test]
    fn wakeup_schedules_fixed_interval() {
        let mut agent = ZiAgent::new(default_cfg(), 42);
        let snaps = snapshot_with_mid(9_999_000, 10_001_000);
        for _ in 0..20 {
            let mut actions = Vec::new();
            agent.wakeup_into(0, 0, &snaps, &mut actions);
            let wakeups: Vec<_> = actions
                .iter()
                .filter_map(|a| match a {
                    AgentAction::ScheduleWakeUp { delay_ns } => Some(*delay_ns),
                    _ => None,
                })
                .collect();
            assert_eq!(wakeups.len(), 1);
            assert_eq!(
                wakeups[0], 30_000_000_000,
                "ZI should use fixed 30s interval, not exponential"
            );
        }
    }

    // ── Side is approximately 50/50 ─────────────────────────────────

    #[test]
    fn side_is_approximately_balanced() {
        let mut agent = ZiAgent::new(default_cfg(), 42);
        let snaps = snapshot_with_mid(9_999_000, 10_001_000);
        let n = 10_000;
        let mut bids = 0u32;
        for _ in 0..n {
            let mut actions = Vec::new();
            agent.wakeup_into(0, 0, &snaps, &mut actions);
            for a in &actions {
                if let AgentAction::SubmitOrder {
                    order: OrderAction::NewLimitOrder { side, .. },
                    ..
                } = a
                {
                    if matches!(side, Side::Bid) {
                        bids += 1;
                    }
                }
            }
        }
        let ratio = bids as f64 / n as f64;
        assert!(
            (0.45..0.55).contains(&ratio),
            "bid ratio {ratio:.3} should be ~0.5"
        );
    }

    // ── Uses reference price when book is empty ─────────────────────

    #[test]
    fn uses_reference_price_on_empty_book() {
        let mut agent = ZiAgent::new(default_cfg(), 42);
        let snaps = empty_snapshot();
        let n = 1000;
        let mut prices = Vec::with_capacity(n);
        for _ in 0..n {
            let mut actions = Vec::new();
            agent.wakeup_into(0, 0, &snaps, &mut actions);
            for a in &actions {
                if let AgentAction::SubmitOrder {
                    order: OrderAction::NewLimitOrder { price, .. },
                    ..
                } = a
                {
                    prices.push(*price);
                }
            }
        }
        let mean = prices.iter().sum::<i64>() as f64 / prices.len() as f64;
        let ref_price = default_cfg().reference_price as f64;
        let pct = ((mean - ref_price) / ref_price).abs();
        assert!(
            pct < 0.01,
            "mean price {mean:.0} should be near reference {ref_price:.0}"
        );
    }
}
