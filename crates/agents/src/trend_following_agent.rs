use std::collections::VecDeque;

use rand::rngs::SmallRng;
use rand::SeedableRng;

use cda_engine::Side;
use sim_core::{Agent, AgentAction, AgentId, ExchangeMessage, MarketSnapshot, Nanos, OrderAction};

use crate::samplers::{
    OffsetPriceSampler, PoissonWakeup, ProportionalSizeSampler, TrendPriceSampler,
    TrendSizeSampler, WakeupSampler,
};
use crate::utils::{IndexedSet, mid_price};

/// Configuration for a trend-following agent.
#[derive(Debug, Clone, serde::Deserialize)]
pub struct TrendFollowingConfig {
    /// Short moving-average window (in candles).
    pub short_window: usize,
    /// Long moving-average window (in candles).
    pub long_window: usize,
    /// Log-return threshold to trigger a trade (e.g. 0.0005 = 0.05%).
    pub threshold: f64,
    /// Price offset from mid as fraction (e.g. 0.02 = 2%).
    pub price_offset: f64,
    /// Order size factor (multiplied by `|ln(short_ma/long_ma)|`).
    pub order_size_factor: f64,
    /// Order size boost (added to factor * `log_diff`).
    pub order_size_boost: f64,
    /// Mean inter-arrival time for Poisson wakeups (nanoseconds).
    pub mean_wakeup_interval_ns: u64,
    /// Sampling frequency for price candles (nanoseconds).
    pub sampling_freq_ns: u64,
    /// Reference price when book is empty.
    pub reference_price: i64,
    /// Symbol index to trade (0-based).
    pub symbol: u32,
    /// If true, invert the trading direction (contrarian/mean-reversion).
    pub contrarian: bool,
}

/// Trend-following (or contrarian) agent based on moving-average crossover.
///
/// Collects raw mid-price candles at `sampling_freq_ns` intervals, computes
/// short and long moving averages, then uses the log-ratio
/// `ln(short_ma) - ln(long_ma)` as the signal. Trades when the signal
/// exceeds `threshold`.
///
/// In contrarian mode, the `should_trade` condition is inverted (trades when
/// `|signal| < threshold`) and the trade direction is flipped.
pub struct TrendFollowingAgent {
    cfg: TrendFollowingConfig,
    rng: SmallRng,
    resting_orders: IndexedSet,
    price_sampler: Box<dyn TrendPriceSampler>,
    size_sampler: Box<dyn TrendSizeSampler>,
    wakeup_sampler: Box<dyn WakeupSampler>,
    /// Price history (raw mid-prices as f64, most recent at back).
    price_history: VecDeque<f64>,
    /// Time of last price sample.
    last_sample_time: u64,
}

impl TrendFollowingAgent {
    /// Create a new trend-following agent with default samplers derived from the config.
    ///
    /// # Panics
    /// Panics if `mean_wakeup_interval_ns` is 0.
    #[must_use]
    pub fn new(config: TrendFollowingConfig, seed: u64) -> Self {
        let price_sampler = Box::new(OffsetPriceSampler::new(config.price_offset));
        let size_sampler = Box::new(ProportionalSizeSampler::new(
            config.order_size_factor,
            config.order_size_boost,
        ));
        let wakeup_sampler = Box::new(PoissonWakeup::new(config.mean_wakeup_interval_ns));
        Self::with_samplers(config, seed, price_sampler, size_sampler, wakeup_sampler)
    }

    /// Create a trend-following agent with custom sampling strategies.
    #[must_use]
    pub fn with_samplers(
        config: TrendFollowingConfig,
        seed: u64,
        price_sampler: Box<dyn TrendPriceSampler>,
        size_sampler: Box<dyn TrendSizeSampler>,
        wakeup_sampler: Box<dyn WakeupSampler>,
    ) -> Self {
        Self {
            cfg: config,
            rng: SmallRng::seed_from_u64(seed),
            resting_orders: IndexedSet::new(),
            price_sampler,
            size_sampler,
            wakeup_sampler,
            price_history: VecDeque::new(),
            last_sample_time: 0,
        }
    }

    #[allow(clippy::cast_precision_loss)]
    fn compute_ma(&self, window: usize) -> Option<f64> {
        if self.price_history.len() < window {
            return None;
        }
        let start = self.price_history.len() - window;
        let sum: f64 = self.price_history.iter().skip(start).sum();
        Some(sum / window as f64)
    }

    fn should_trade(&self, ma_diff: f64) -> bool {
        if self.cfg.contrarian {
            ma_diff.abs() < self.cfg.threshold
        } else {
            ma_diff.abs() > self.cfg.threshold
        }
    }

    fn determine_side(&self, ma_diff: f64) -> Side {
        if self.cfg.contrarian {
            if ma_diff > 0.0 { Side::Ask } else { Side::Bid }
        } else {
            if ma_diff > 0.0 { Side::Bid } else { Side::Ask }
        }
    }
}

impl Agent for TrendFollowingAgent {
    fn wakeup_into(
        &mut self,
        time: Nanos,
        _agent_id: AgentId,
        snapshots: &[MarketSnapshot],
        actions: &mut Vec<AgentAction>,
    ) {
        let snap = &snapshots[self.cfg.symbol as usize];
        let mid = mid_price(snap, self.cfg.reference_price);

        // Sample price candle if enough time has elapsed
        if time >= self.last_sample_time + self.cfg.sampling_freq_ns || self.price_history.is_empty() {
            #[allow(clippy::cast_precision_loss)]
            self.price_history.push_back(mid as f64);
            self.last_sample_time = time;
            let max_len = self.cfg.long_window + 10;
            while self.price_history.len() > max_len {
                self.price_history.pop_front();
            }
        }

        // Compute MAs and decide whether to trade
        if let (Some(short_ma), Some(long_ma)) = (
            self.compute_ma(self.cfg.short_window),
            self.compute_ma(self.cfg.long_window),
        ) {
            if short_ma <= 0.0 || long_ma <= 0.0 {
                let delay = self.wakeup_sampler.sample_wakeup_delay(&mut self.rng);
                actions.push(AgentAction::ScheduleWakeUp { delay_ns: delay });
                return;
            }
            let ma_diff = short_ma.ln() - long_ma.ln();

            if self.should_trade(ma_diff) {
                for oid in self.resting_orders.drain_all() {
                    actions.push(AgentAction::CancelOrder {
                        symbol: self.cfg.symbol,
                        order_id: oid,
                    });
                }

                let side = self.determine_side(ma_diff);
                let qty = self.size_sampler.sample_order_size(ma_diff, &mut self.rng);
                let price = self.price_sampler.sample_price(mid, side, &mut self.rng);

                actions.push(AgentAction::SubmitOrder {
                    symbol: self.cfg.symbol,
                    order: OrderAction::NewLimitOrder { side, price, qty },
                });
            }
        }

        // Schedule next wakeup
        let delay = self.wakeup_sampler.sample_wakeup_delay(&mut self.rng);
        actions.push(AgentAction::ScheduleWakeUp { delay_ns: delay });
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
mod tests {
    use super::*;

    fn default_cfg() -> TrendFollowingConfig {
        TrendFollowingConfig {
            short_window: 3,
            long_window: 5,
            threshold: 0.001,
            price_offset: 0.02,
            order_size_factor: 1000.0,
            order_size_boost: 100.0,
            mean_wakeup_interval_ns: 5_000_000_000,
            sampling_freq_ns: 1_000_000_000,
            reference_price: 10_000,
            symbol: 0,
            contrarian: false,
        }
    }

    fn snap_at_price(mid: i64) -> Vec<MarketSnapshot> {
        vec![MarketSnapshot {
            best_bid: Some((mid - 1, 100)),
            best_ask: Some((mid + 1, 100)),
            last_trade_price: Some(mid),
            last_trade_time: Some(0),
        }]
    }

    // ── Raw prices stored (not log-prices) ────────────────────────

    #[test]
    fn stores_raw_prices_not_log() {
        let mut agent = TrendFollowingAgent::new(default_cfg(), 42);
        let snaps = snap_at_price(10_000);
        let mut actions = Vec::new();
        agent.wakeup_into(0, 0, &snaps, &mut actions);
        let stored = agent.price_history.back().unwrap();
        assert!(
            *stored > 100.0,
            "price_history should store raw prices, got {stored} (log would be ~9.2)"
        );
    }

    // ── Signal uses log-ratio of MAs ────────────────────────────────

    #[test]
    fn signal_is_log_ratio_of_raw_mas() {
        let mut cfg = default_cfg();
        cfg.short_window = 3;
        cfg.long_window = 5;
        cfg.threshold = 0.0; // always trade
        let mut agent = TrendFollowingAgent::new(cfg, 42);
        agent.price_history.push_back(100.0);
        agent.price_history.push_back(200.0);
        agent.price_history.push_back(300.0);
        agent.price_history.push_back(400.0);
        agent.price_history.push_back(500.0);
        // short_ma(3) = 400, long_ma(5) = 300
        // signal = ln(400/300) ≈ 0.2877 > 0 → should trade, side = Bid
        let signal = (400.0_f64 / 300.0).ln();
        assert!(agent.should_trade(signal));
        assert_eq!(agent.determine_side(signal), Side::Bid);
    }

    // ── MA computation ──────────────────────────────────────────────

    #[test]
    fn ma_computation_correct() {
        let mut agent = TrendFollowingAgent::new(default_cfg(), 42);
        agent.price_history.push_back(100.0);
        agent.price_history.push_back(200.0);
        agent.price_history.push_back(300.0);
        agent.price_history.push_back(400.0);
        agent.price_history.push_back(500.0);

        let short_ma = agent.compute_ma(3).unwrap();
        let long_ma = agent.compute_ma(5).unwrap();
        assert!((short_ma - 400.0).abs() < 1e-10, "short MA = {short_ma}");
        assert!((long_ma - 300.0).abs() < 1e-10, "long MA = {long_ma}");
    }

    #[test]
    fn ma_returns_none_with_insufficient_history() {
        let mut agent = TrendFollowingAgent::new(default_cfg(), 42);
        agent.price_history.push_back(1.0);
        agent.price_history.push_back(2.0);
        assert!(agent.compute_ma(3).is_none());
        assert!(agent.compute_ma(5).is_none());
    }

    // ── Trade direction: trend-following ─────────────────────────────

    #[test]
    fn trend_following_buys_on_uptrend() {
        let agent = TrendFollowingAgent::new(default_cfg(), 42);
        assert!(agent.should_trade(0.01));
        assert_eq!(agent.determine_side(0.01), Side::Bid);
    }

    #[test]
    fn trend_following_sells_on_downtrend() {
        let agent = TrendFollowingAgent::new(default_cfg(), 42);
        assert!(agent.should_trade(-0.01));
        assert_eq!(agent.determine_side(-0.01), Side::Ask);
    }

    #[test]
    fn trend_following_no_trade_below_threshold() {
        let agent = TrendFollowingAgent::new(default_cfg(), 42);
        assert!(!agent.should_trade(0.0005));
        assert!(!agent.should_trade(-0.0005));
    }

    // ── Trade direction: contrarian ─────────────────────────────────

    #[test]
    fn contrarian_sells_on_uptrend() {
        let mut cfg = default_cfg();
        cfg.contrarian = true;
        let agent = TrendFollowingAgent::new(cfg, 42);
        assert!(agent.should_trade(0.0005));
        assert_eq!(agent.determine_side(0.01), Side::Ask);
    }

    #[test]
    fn contrarian_buys_on_downtrend() {
        let mut cfg = default_cfg();
        cfg.contrarian = true;
        let agent = TrendFollowingAgent::new(cfg, 42);
        assert_eq!(agent.determine_side(-0.01), Side::Bid);
    }

    #[test]
    fn contrarian_no_trade_above_threshold() {
        let mut cfg = default_cfg();
        cfg.contrarian = true;
        let agent = TrendFollowingAgent::new(cfg, 42);
        assert!(!agent.should_trade(0.01));
    }

    // ── No trade before enough candles ───────────────────────────────

    #[test]
    fn no_trade_before_enough_history() {
        let mut agent = TrendFollowingAgent::new(default_cfg(), 42);
        let snaps = snap_at_price(10_000);
        let mut actions = Vec::new();
        agent.wakeup_into(0, 0, &snaps, &mut actions);
        let submits: Vec<_> = actions
            .iter()
            .filter(|a| matches!(a, AgentAction::SubmitOrder { .. }))
            .collect();
        assert_eq!(submits.len(), 0, "should not trade with only 1 candle");
    }

    // ── Candle sampling respects frequency ──────────────────────────

    #[test]
    fn candle_sampling_respects_frequency() {
        let mut agent = TrendFollowingAgent::new(default_cfg(), 42);
        let snaps = snap_at_price(10_000);

        let mut actions = Vec::new();
        agent.wakeup_into(0, 0, &snaps, &mut actions);
        assert_eq!(agent.price_history.len(), 1);

        actions.clear();
        agent.wakeup_into(500_000_000, 0, &snaps, &mut actions);
        assert_eq!(agent.price_history.len(), 1);

        actions.clear();
        agent.wakeup_into(1_000_000_000, 0, &snaps, &mut actions);
        assert_eq!(agent.price_history.len(), 2);
    }

    // ── Cancel-all before placing ───────────────────────────────────

    #[test]
    fn cancels_all_resting_before_trading() {
        let mut cfg = default_cfg();
        cfg.short_window = 2;
        cfg.long_window = 3;
        cfg.threshold = 0.0;
        let mut agent = TrendFollowingAgent::new(cfg, 42);

        agent.on_exchange_message(0, 0, ExchangeMessage::OrderAccepted { order_id: 10 });
        agent.on_exchange_message(0, 0, ExchangeMessage::OrderAccepted { order_id: 20 });

        agent.price_history.push_back(9000.0);
        agent.price_history.push_back(9100.0);
        agent.price_history.push_back(9200.0);
        agent.last_sample_time = 0;

        let snaps = snap_at_price(10_000);
        let mut actions = Vec::new();
        agent.wakeup_into(2_000_000_000, 0, &snaps, &mut actions);

        let cancels: Vec<_> = actions
            .iter()
            .filter(|a| matches!(a, AgentAction::CancelOrder { .. }))
            .collect();
        assert_eq!(cancels.len(), 2, "should cancel both resting orders");
    }

    // ── Custom samplers via with_samplers ────────────────────────────

    #[test]
    fn custom_samplers_are_used() {
        use crate::samplers::FixedIntervalWakeup;

        let cfg = default_cfg();
        let mut agent = TrendFollowingAgent::with_samplers(
            cfg,
            42,
            Box::new(OffsetPriceSampler::new(0.05)),
            Box::new(ProportionalSizeSampler::new(500.0, 50.0)),
            Box::new(FixedIntervalWakeup::new(999_000_000)),
        );

        let snaps = snap_at_price(10_000);
        let mut actions = Vec::new();
        agent.wakeup_into(0, 0, &snaps, &mut actions);

        // Check that the fixed wakeup interval is used
        let wakeups: Vec<_> = actions
            .iter()
            .filter_map(|a| match a {
                AgentAction::ScheduleWakeUp { delay_ns } => Some(*delay_ns),
                _ => None,
            })
            .collect();
        assert_eq!(wakeups.len(), 1);
        assert_eq!(wakeups[0], 999_000_000, "custom fixed wakeup should be used");
    }
}
