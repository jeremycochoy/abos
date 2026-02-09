use std::collections::HashMap;

use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use rand_distr::{Distribution, Exp};

use cda_engine::Side;
use sim_core::{Agent, AgentAction, AgentId, ExchangeMessage, MarketSnapshot, Nanos, OrderAction};

/// Configuration for a zero-intelligence agent.
#[derive(Debug, Clone)]
pub struct ZiAgentConfig {
    /// Probability of placing a limit order (vs market order).
    pub p_limit: f64,
    /// Probability of cancelling a resting order each wakeup.
    pub p_cancel: f64,
    /// Mean inter-arrival time between wakeups (nanoseconds).
    pub mean_wakeup_interval_ns: u64,
    /// Exponential distribution parameter for limit order price offset (in ticks).
    pub price_offset_lambda: f64,
    /// Fixed order quantity.
    pub default_qty: u64,
    /// Reference price when book is empty (internal integer units).
    pub reference_price: i64,
    /// Minimum price increment.
    pub tick_size: i64,
    /// Symbol index to trade (0-based).
    pub symbol: u32,
}

/// Set supporting O(1) insert, remove-by-value, and random-index selection.
struct IndexedSet {
    vec: Vec<u64>,
    map: HashMap<u64, usize>,
}

impl IndexedSet {
    fn new() -> Self {
        Self { vec: Vec::new(), map: HashMap::new() }
    }

    fn len(&self) -> usize {
        self.vec.len()
    }

    fn is_empty(&self) -> bool {
        self.vec.is_empty()
    }

    fn insert(&mut self, val: u64) {
        let idx = self.vec.len();
        self.vec.push(val);
        self.map.insert(val, idx);
    }

    fn remove(&mut self, val: u64) -> bool {
        let Some(idx) = self.map.remove(&val) else { return false };
        self.vec.swap_remove(idx);
        if idx < self.vec.len() {
            let swapped = self.vec[idx];
            self.map.insert(swapped, idx);
        }
        true
    }

    fn remove_at(&mut self, idx: usize) -> u64 {
        let val = self.vec.swap_remove(idx);
        self.map.remove(&val);
        if idx < self.vec.len() {
            let swapped = self.vec[idx];
            self.map.insert(swapped, idx);
        }
        val
    }
}

/// Zero-intelligence constrained (ZI-C) agent.
pub struct ZiAgent {
    cfg: ZiAgentConfig,
    rng: SmallRng,
    resting_orders: IndexedSet,
    offset_dist: Exp<f64>,
    wakeup_dist: Exp<f64>,
}

impl ZiAgent {
    /// Create a new ZI agent with the given config and RNG seed.
    ///
    /// # Panics
    /// Panics if `price_offset_lambda` or `mean_wakeup_interval_ns` are invalid.
    #[must_use]
    pub fn new(config: ZiAgentConfig, seed: u64) -> Self {
        let offset_dist = Exp::new(config.price_offset_lambda)
            .expect("invalid exponential lambda");
        #[allow(clippy::cast_precision_loss)]
        let wakeup_dist = Exp::new(1.0 / config.mean_wakeup_interval_ns as f64)
            .expect("invalid mean wakeup interval");
        Self {
            cfg: config,
            rng: SmallRng::seed_from_u64(seed),
            resting_orders: IndexedSet::new(),
            offset_dist,
            wakeup_dist,
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
        let snap = &snapshots[self.cfg.symbol as usize];

        // Maybe cancel a resting order
        if !self.resting_orders.is_empty() && self.rng.gen::<f64>() < self.cfg.p_cancel {
            let idx = self.rng.gen_range(0..self.resting_orders.len());
            let oid = self.resting_orders.remove_at(idx);
            actions.push(AgentAction::CancelOrder {
                symbol: self.cfg.symbol,
                order_id: oid,
            });
        }

        // Decide side
        let side = if self.rng.gen::<bool>() { Side::Bid } else { Side::Ask };

        // Decide limit vs market
        let order = if self.rng.gen::<f64>() < self.cfg.p_limit {
            let mid = mid_price(snap, self.cfg.reference_price);
            let offset = self.sample_offset();
            let price = match side {
                Side::Bid => (mid - offset).max(self.cfg.tick_size),
                Side::Ask => mid + offset,
            };
            OrderAction::NewLimitOrder { side, price, qty: self.cfg.default_qty }
        } else {
            OrderAction::NewMarketOrder { side, qty: self.cfg.default_qty }
        };

        actions.push(AgentAction::SubmitOrder {
            symbol: self.cfg.symbol,
            order,
        });

        // Schedule next wakeup (exponential inter-arrival)
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

impl ZiAgent {
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss, clippy::cast_precision_loss)]
    fn sample_offset(&mut self) -> i64 {
        let raw: f64 = self.offset_dist.sample(&mut self.rng);
        let ticks = (raw / self.cfg.tick_size as f64).round() as i64;
        ticks.max(1) * self.cfg.tick_size
    }

    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    fn sample_wakeup_delay(&mut self) -> u64 {
        let delay: f64 = self.wakeup_dist.sample(&mut self.rng);
        (delay.round() as u64).max(1)
    }
}

fn mid_price(snap: &MarketSnapshot, reference: i64) -> i64 {
    match (snap.best_bid, snap.best_ask) {
        (Some((bid, _)), Some((ask, _))) => bid + (ask - bid) / 2,
        (Some((bid, _)), None) => bid,
        (None, Some((ask, _))) => ask,
        (None, None) => reference,
    }
}
