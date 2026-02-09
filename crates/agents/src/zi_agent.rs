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
