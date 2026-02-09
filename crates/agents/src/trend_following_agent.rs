use std::collections::VecDeque;

use rand::rngs::SmallRng;
use rand::SeedableRng;
use rand_distr::{Distribution, Exp};

use cda_engine::Side;
use sim_core::{Agent, AgentAction, AgentId, ExchangeMessage, MarketSnapshot, Nanos, OrderAction};

use crate::utils::{IndexedSet, mid_price};

/// Configuration for a trend-following agent.
#[derive(Debug, Clone)]
pub struct TrendFollowingConfig {
    /// Short moving-average window (in candles).
    pub short_window: usize,
    /// Long moving-average window (in candles).
    pub long_window: usize,
    /// Log-return threshold to trigger a trade (e.g. 0.0005 = 0.05%).
    pub threshold: f64,
    /// Price offset from mid as fraction (e.g. 0.02 = 2%).
    pub price_offset: f64,
    /// Order size factor (multiplied by `|ma_diff|`).
    pub order_size_factor: f64,
    /// Order size boost (added to factor*diff).
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
/// Collects price candles at `sampling_freq_ns` intervals, computes short and
/// long moving averages, and trades when the difference exceeds `threshold`.
///
/// In contrarian mode, the `should_trade` condition is inverted (trades when
/// `|ma_diff| < threshold`) and the trade direction is flipped.
pub struct TrendFollowingAgent {
    cfg: TrendFollowingConfig,
    rng: SmallRng,
    resting_orders: IndexedSet,
    wakeup_dist: Exp<f64>,
    /// Price history (log-prices, most recent at back).
    price_history: VecDeque<f64>,
    /// Time of last price sample.
    last_sample_time: u64,
}

impl TrendFollowingAgent {
    /// Create a new trend-following agent.
    ///
    /// # Panics
    /// Panics if `mean_wakeup_interval_ns` is 0.
    #[must_use]
    pub fn new(config: TrendFollowingConfig, seed: u64) -> Self {
        #[allow(clippy::cast_precision_loss)]
        let wakeup_dist = Exp::new(1.0 / config.mean_wakeup_interval_ns as f64)
            .expect("invalid mean wakeup interval");
        Self {
            cfg: config,
            rng: SmallRng::seed_from_u64(seed),
            resting_orders: IndexedSet::new(),
            wakeup_dist,
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
            // Mean reversion: sell when short > long (overpriced), buy when under
            if ma_diff > 0.0 { Side::Ask } else { Side::Bid }
        } else {
            // Trend following: buy when short > long (uptrend)
            if ma_diff > 0.0 { Side::Bid } else { Side::Ask }
        }
    }

    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    fn compute_order_size(&self, ma_diff: f64) -> u64 {
        let size = self.cfg.order_size_factor * ma_diff.abs() + self.cfg.order_size_boost;
        (size.round() as u64).max(1)
    }

    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss, clippy::cast_precision_loss)]
    fn compute_price(&self, mid: i64, side: Side) -> i64 {
        match side {
            Side::Bid => {
                let p = mid as f64 * (1.0 + self.cfg.price_offset);
                (p.round() as i64).max(1)
            }
            Side::Ask => {
                let p = mid as f64 * (1.0 - self.cfg.price_offset);
                (p.round() as i64).max(1)
            }
        }
    }

    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    fn sample_wakeup_delay(&mut self) -> u64 {
        let delay: f64 = self.wakeup_dist.sample(&mut self.rng);
        (delay.round() as u64).max(1)
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
            let log_price = (mid as f64).ln();
            self.price_history.push_back(log_price);
            self.last_sample_time = time;
            // Keep at most long_window + some buffer
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
            let ma_diff = short_ma - long_ma;

            if self.should_trade(ma_diff) {
                // Cancel all outstanding orders
                for oid in self.resting_orders.drain_all() {
                    actions.push(AgentAction::CancelOrder {
                        symbol: self.cfg.symbol,
                        order_id: oid,
                    });
                }

                let side = self.determine_side(ma_diff);
                let qty = self.compute_order_size(ma_diff);
                let price = self.compute_price(mid, side);

                actions.push(AgentAction::SubmitOrder {
                    symbol: self.cfg.symbol,
                    order: OrderAction::NewLimitOrder { side, price, qty },
                });
            }
        }

        // Schedule next wakeup (Poisson process)
        let delay = self.sample_wakeup_delay();
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
