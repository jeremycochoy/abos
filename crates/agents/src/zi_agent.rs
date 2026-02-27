use rand::rngs::SmallRng;
use rand::Rng;
use rand::SeedableRng;

use cda_engine::Side;
use sim_core::{Agent, AgentAction, AgentId, ExchangeMessage, MarketSnapshot, Nanos, OrderAction};

use crate::samplers::{
    FixedIntervalWakeup, LogNormalPriceSampler, LogNormalSizeSampler, OrderSizeSampler,
    PriceSampler, WakeupSampler,
};
use crate::utils::{IndexedSet, mid_price};

/// Configuration for a zero-intelligence agent with default samplers.
#[derive(Debug, Clone, serde::Deserialize)]
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

/// Zero-intelligence agent with pluggable sampling strategies.
///
/// On each wakeup the agent:
/// 1. Cancels all outstanding orders
/// 2. Picks a random side (buy/sell 50/50)
/// 3. Samples order size via [`OrderSizeSampler`]
/// 4. Samples price via [`PriceSampler`]
/// 5. Places a single limit order
/// 6. Schedules next wakeup via [`WakeupSampler`]
pub struct ZiAgent {
    reference_price: i64,
    symbol: u32,
    rng: SmallRng,
    resting_orders: IndexedSet,
    price_sampler: Box<dyn PriceSampler>,
    size_sampler: Box<dyn OrderSizeSampler>,
    wakeup_sampler: Box<dyn WakeupSampler>,
}

impl ZiAgent {
    /// Create a ZI agent with default log-normal samplers matching the ABIDES implementation.
    #[must_use]
    pub fn new(config: ZiAgentConfig, seed: u64) -> Self {
        Self::with_samplers(
            config.reference_price,
            config.symbol,
            seed,
            Box::new(LogNormalPriceSampler::new(config.price_std)),
            Box::new(LogNormalSizeSampler::new(
                config.order_size_scale,
                config.order_size_std,
            )),
            Box::new(FixedIntervalWakeup::new(config.wake_up_interval_ns)),
        )
    }

    /// Create a ZI agent with custom sampling strategies.
    #[must_use]
    pub fn with_samplers(
        reference_price: i64,
        symbol: u32,
        seed: u64,
        price_sampler: Box<dyn PriceSampler>,
        size_sampler: Box<dyn OrderSizeSampler>,
        wakeup_sampler: Box<dyn WakeupSampler>,
    ) -> Self {
        Self {
            reference_price,
            symbol,
            rng: SmallRng::seed_from_u64(seed),
            resting_orders: IndexedSet::new(),
            price_sampler,
            size_sampler,
            wakeup_sampler,
        }
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
        let snap = &snapshots[self.symbol as usize];

        for oid in self.resting_orders.drain_all() {
            actions.push(AgentAction::CancelOrder {
                symbol: self.symbol,
                order_id: oid,
            });
        }

        let side = if self.rng.gen::<bool>() {
            Side::Bid
        } else {
            Side::Ask
        };

        let mid = mid_price(snap, self.reference_price);
        let qty = self.size_sampler.sample_order_size(&mut self.rng);
        let price = self.price_sampler.sample_price(mid, &mut self.rng);

        actions.push(AgentAction::SubmitOrder {
            symbol: self.symbol,
            order: OrderAction::NewLimitOrder { side, price, qty },
        });

        actions.push(AgentAction::ScheduleWakeUp {
            delay_ns: self.wakeup_sampler.sample_wakeup_delay(&mut self.rng),
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

    // ── Cancel-all behavior ─────────────────────────────────────────

    #[test]
    fn wakeup_cancels_all_resting_orders() {
        let mut agent = ZiAgent::new(default_cfg(), 42);
        agent.on_exchange_message(0, 0, ExchangeMessage::OrderAccepted { order_id: 100 });
        agent.on_exchange_message(0, 0, ExchangeMessage::OrderAccepted { order_id: 200 });
        agent.on_exchange_message(0, 0, ExchangeMessage::OrderAccepted { order_id: 300 });

        let snaps = snapshot_with_mid(9_999_000, 10_001_000);
        let mut actions = Vec::new();
        agent.wakeup_into(1_000, 0, &snaps, &mut actions);

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

    // ── Fixed-interval wakeup (default sampler) ─────────────────────

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
                "ZI should use fixed 30s interval with default sampler"
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

    // ── Custom samplers via with_samplers ────────────────────────────

    #[test]
    fn custom_samplers_are_used() {
        use crate::samplers::PoissonWakeup;

        let mut agent = ZiAgent::with_samplers(
            10_000_000,
            0,
            42,
            Box::new(LogNormalPriceSampler::new(0.03 / 100.0)),
            Box::new(LogNormalSizeSampler::new(12_000.0, 1.7)),
            Box::new(PoissonWakeup::new(5_000_000_000)),
        );

        let snaps = snapshot_with_mid(9_999_000, 10_001_000);
        let mut delays = Vec::new();
        for _ in 0..100 {
            let mut actions = Vec::new();
            agent.wakeup_into(0, 0, &snaps, &mut actions);
            for a in &actions {
                if let AgentAction::ScheduleWakeUp { delay_ns } = a {
                    delays.push(*delay_ns);
                }
            }
        }
        // Poisson wakeup should produce varying delays (not all the same)
        let all_same = delays.windows(2).all(|w| w[0] == w[1]);
        assert!(
            !all_same,
            "Poisson wakeup should produce varying delays"
        );
    }
}
