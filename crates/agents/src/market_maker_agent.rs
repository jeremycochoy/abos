use rand::rngs::SmallRng;
use rand::SeedableRng;
use rand_distr::{Distribution, Exp};

use cda_engine::Side;
use sim_core::{Agent, AgentAction, AgentId, ExchangeMessage, MarketSnapshot, Nanos, OrderAction};

use crate::utils::{IndexedSet, mid_price};

/// Configuration for a liquidity market-maker agent.
#[derive(Debug, Clone)]
pub struct MarketMakerConfig {
    /// Total liquidity budget per update cycle (in qty units).
    pub total_liquidity: f64,
    /// Step size between price levels as fraction of mid (e.g. 0.0001 = 0.01%).
    pub step_size_ratio: f64,
    /// Number of price levels on each side.
    pub max_levels: usize,
    /// Inventory imbalance sensitivity (tanh scaling factor).
    pub imbalance_beta: f64,
    /// Peak distance ratio for the hump model (fraction of mid, e.g. 0.05 = 5%).
    pub peak_distance_ratio: f64,
    /// Shape exponent for the hump model.
    pub shape_exponent: f64,
    /// Mean wakeup interval (Poisson process, nanoseconds).
    pub mean_wakeup_interval_ns: u64,
    /// Reference price when book is empty.
    pub reference_price: i64,
    /// Symbol index to trade (0-based).
    pub symbol: u32,
}

/// Liquidity market-maker agent using a symmetric-hump distribution.
///
/// Places multiple bid and ask orders at different price levels around the
/// mid-price. Adjusts liquidity distribution based on inventory imbalance.
pub struct MarketMakerAgent {
    cfg: MarketMakerConfig,
    rng: SmallRng,
    resting_orders: IndexedSet,
    wakeup_dist: Exp<f64>,
    /// Net inventory (positive = long, negative = short).
    inventory: i64,
}

impl MarketMakerAgent {
    /// Create a new market-maker agent.
    ///
    /// # Panics
    /// Panics if `mean_wakeup_interval_ns` is 0.
    #[must_use]
    pub fn new(config: MarketMakerConfig, seed: u64) -> Self {
        #[allow(clippy::cast_precision_loss)]
        let wakeup_dist = Exp::new(1.0 / config.mean_wakeup_interval_ns as f64)
            .expect("invalid mean wakeup interval");
        Self {
            cfg: config,
            rng: SmallRng::seed_from_u64(seed),
            resting_orders: IndexedSet::new(),
            wakeup_dist,
            inventory: 0,
        }
    }

    /// Compute the symmetric-hump liquidity weight at a given log-distance
    /// from mid-price.
    fn hump_weight(&self, log_distance: f64) -> f64 {
        let eps = 4e-4;
        let peak_fraction = (1.0 + self.cfg.peak_distance_ratio).ln();
        let decay_rate = self.cfg.shape_exponent / (peak_fraction + eps);
        let d = log_distance.abs() + eps;
        d.powf(self.cfg.shape_exponent) * (-decay_rate * log_distance.abs()).exp()
    }

    /// Compute the inventory imbalance factor [-1, 1].
    #[allow(clippy::cast_precision_loss)]
    fn imbalance(&self) -> f64 {
        let raw = (self.inventory as f64 * self.cfg.imbalance_beta
            / self.cfg.total_liquidity)
            .tanh();
        raw.clamp(-1.0, 1.0)
    }

    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    fn sample_wakeup_delay(&mut self) -> u64 {
        let delay: f64 = self.wakeup_dist.sample(&mut self.rng);
        (delay.round() as u64).max(1)
    }
}

impl Agent for MarketMakerAgent {
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss, clippy::cast_precision_loss)]
    fn wakeup_into(
        &mut self,
        _time: Nanos,
        _agent_id: AgentId,
        snapshots: &[MarketSnapshot],
        actions: &mut Vec<AgentAction>,
    ) {
        let snap = &snapshots[self.cfg.symbol as usize];
        let mid = mid_price(snap, self.cfg.reference_price);

        // Cancel all outstanding orders
        for oid in self.resting_orders.drain_all() {
            actions.push(AgentAction::CancelOrder {
                symbol: self.cfg.symbol,
                order_id: oid,
            });
        }

        let imbalance = self.imbalance();
        let log_step = (1.0 + self.cfg.step_size_ratio).ln();

        // Compute weights for all levels to normalize
        let mut bid_weights = Vec::with_capacity(self.cfg.max_levels);
        let mut ask_weights = Vec::with_capacity(self.cfg.max_levels);
        let mut total_weight = 0.0;

        for level in 1..=self.cfg.max_levels {
            let log_dist = log_step * level as f64;
            let base = self.hump_weight(log_dist);
            let bid_w = base * (1.0 - imbalance);
            let ask_w = base * (1.0 + imbalance);
            bid_weights.push((log_dist, bid_w));
            ask_weights.push((log_dist, ask_w));
            total_weight += bid_w + ask_w;
        }

        if total_weight < 1e-15 {
            actions.push(AgentAction::ScheduleWakeUp {
                delay_ns: self.sample_wakeup_delay(),
            });
            return;
        }

        // Place bid orders
        for &(log_dist, weight) in &bid_weights {
            let qty = (self.cfg.total_liquidity * weight / total_weight).round() as u64;
            if qty == 0 { continue; }
            let price = (mid as f64 * (-log_dist).exp()).round() as i64;
            if price < 1 { continue; }
            actions.push(AgentAction::SubmitOrder {
                symbol: self.cfg.symbol,
                order: OrderAction::NewLimitOrder { side: Side::Bid, price, qty },
            });
        }

        // Place ask orders
        for &(log_dist, weight) in &ask_weights {
            let qty = (self.cfg.total_liquidity * weight / total_weight).round() as u64;
            if qty == 0 { continue; }
            let price = (mid as f64 * log_dist.exp()).round() as i64;
            actions.push(AgentAction::SubmitOrder {
                symbol: self.cfg.symbol,
                order: OrderAction::NewLimitOrder { side: Side::Ask, price, qty },
            });
        }

        // Schedule next wakeup
        actions.push(AgentAction::ScheduleWakeUp {
            delay_ns: self.sample_wakeup_delay(),
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
            ExchangeMessage::OrderFilled { order_id, qty, .. } => {
                self.resting_orders.remove(order_id);
                // Track inventory changes (approximate: we don't know side here,
                // but fills alternate our exposure). We track net fills.
                #[allow(clippy::cast_possible_wrap)]
                { self.inventory += qty as i64; }
            }
            ExchangeMessage::OrderCancelled { order_id } => {
                self.resting_orders.remove(order_id);
            }
            ExchangeMessage::OrderRejected { .. } => {}
        }
    }
}
