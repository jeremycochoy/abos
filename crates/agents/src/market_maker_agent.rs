use rand::rngs::SmallRng;
use rand::SeedableRng;

use cda_engine::Side;
use sim_core::{Agent, AgentAction, AgentId, ExchangeMessage, MarketSnapshot, Nanos, OrderAction};

use crate::samplers::{LiquidityWeightModel, PoissonWakeup, SymmetricHumpModel, WakeupSampler};
use crate::utils::IndexedSet;

/// Configuration for a liquidity market-maker agent.
#[derive(Debug, Clone, serde::Deserialize)]
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

/// Liquidity market-maker agent using a pluggable weight model.
///
/// Places multiple bid and ask orders at different price levels around the
/// geometric mid-price `sqrt(bid * ask)`. Adjusts liquidity distribution
/// based on inventory imbalance, with per-side normalization ensuring each
/// side receives exactly half the total liquidity budget.
pub struct MarketMakerAgent {
    cfg: MarketMakerConfig,
    rng: SmallRng,
    resting_orders: IndexedSet,
    weight_model: Box<dyn LiquidityWeightModel>,
    wakeup_sampler: Box<dyn WakeupSampler>,
    /// Net inventory (positive = long, negative = short), advanced from the
    /// side carried by each `OrderFilled` message.
    inventory: i64,
}

impl MarketMakerAgent {
    /// Create a new market-maker agent with default samplers derived from the config.
    ///
    /// # Panics
    /// Panics if `mean_wakeup_interval_ns` is 0.
    #[must_use]
    pub fn new(config: MarketMakerConfig, seed: u64) -> Self {
        let weight_model = Box::new(SymmetricHumpModel::new(
            config.peak_distance_ratio,
            config.shape_exponent,
        ));
        let wakeup_sampler = Box::new(PoissonWakeup::new(config.mean_wakeup_interval_ns));
        Self::with_samplers(config, seed, weight_model, wakeup_sampler)
    }

    /// Create a market-maker agent with custom weight model and wakeup sampler.
    #[must_use]
    pub fn with_samplers(
        config: MarketMakerConfig,
        seed: u64,
        weight_model: Box<dyn LiquidityWeightModel>,
        wakeup_sampler: Box<dyn WakeupSampler>,
    ) -> Self {
        Self {
            cfg: config,
            rng: SmallRng::seed_from_u64(seed),
            resting_orders: IndexedSet::new(),
            weight_model,
            wakeup_sampler,
            inventory: 0,
        }
    }

    /// Compute the inventory imbalance factor [-1, 1].
    #[allow(clippy::cast_precision_loss)]
    fn imbalance(&self) -> f64 {
        let raw = (self.inventory as f64 * self.cfg.imbalance_beta
            / self.cfg.total_liquidity)
            .tanh();
        raw.clamp(-1.0, 1.0)
    }

    /// Compute mid-price as geometric mean `sqrt(bid * ask)`, matching ABIDES.
    /// Falls back to one-sided price or reference price when book is empty.
    #[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
    fn geometric_mid(snap: &MarketSnapshot, reference: i64) -> i64 {
        match (snap.best_bid, snap.best_ask) {
            (Some((bid, _)), Some((ask, _))) => {
                ((bid as f64 * ask as f64).sqrt().round()) as i64
            }
            (Some((bid, _)), None) => bid,
            (None, Some((ask, _))) => ask,
            (None, None) => reference,
        }
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
        let mid = Self::geometric_mid(snap, self.cfg.reference_price);

        // Cancel all outstanding orders
        for oid in self.resting_orders.drain_all() {
            actions.push(AgentAction::CancelOrder {
                symbol: self.cfg.symbol,
                order_id: oid,
            });
        }

        let imbalance = self.imbalance();
        let log_step = (1.0 + self.cfg.step_size_ratio).ln();

        // Compute weights for all levels
        let max_levels = self.cfg.max_levels;
        let mut bid_weights = Vec::with_capacity(max_levels);
        let mut ask_weights = Vec::with_capacity(max_levels);

        for level in 1..=max_levels {
            let log_dist = log_step * level as f64;
            let base = self.weight_model.weight(log_dist, &mut self.rng);
            let bid_w = base * (1.0 - imbalance);
            let ask_w = base * (1.0 + imbalance);
            bid_weights.push((log_dist, bid_w));
            ask_weights.push((log_dist, ask_w));
        }

        // Per-side normalization: each side gets exactly half the total liquidity
        let half_liq = self.cfg.total_liquidity * 0.5;
        let bid_total: f64 = bid_weights.iter().map(|(_, w)| w).sum();
        let ask_total: f64 = ask_weights.iter().map(|(_, w)| w).sum();

        // Place bid orders
        if bid_total > 1e-15 {
            for &(log_dist, weight) in &bid_weights {
                let qty = (half_liq * weight / bid_total).round() as u64;
                if qty == 0 { continue; }
                let price = (mid as f64 * (-log_dist).exp()).round() as i64;
                if price < 1 { continue; }
                actions.push(AgentAction::SubmitOrder {
                    symbol: self.cfg.symbol,
                    order: OrderAction::NewLimitOrder { side: Side::Bid, price, qty, user_id: 0 },
                });
            }
        }

        // Place ask orders
        if ask_total > 1e-15 {
            for &(log_dist, weight) in &ask_weights {
                let qty = (half_liq * weight / ask_total).round() as u64;
                if qty == 0 { continue; }
                let price = (mid as f64 * log_dist.exp()).round() as i64;
                actions.push(AgentAction::SubmitOrder {
                    symbol: self.cfg.symbol,
                    order: OrderAction::NewLimitOrder { side: Side::Ask, price, qty, user_id: 0 },
                });
            }
        }

        // Schedule next wakeup
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
        // Every message is self-contained: fills carry the order's side, so
        // inventory needs no submit-time bookkeeping to attribute direction.
        match message {
            ExchangeMessage::OrderAccepted { order_id, .. } => {
                self.resting_orders.insert(order_id);
            }
            ExchangeMessage::OrderFilled { order_id, side, qty, .. } => {
                self.resting_orders.remove(order_id);
                #[allow(clippy::cast_possible_wrap)]
                match side {
                    Side::Bid => self.inventory += qty as i64,
                    Side::Ask => self.inventory -= qty as i64,
                }
            }
            ExchangeMessage::OrderCancelled { order_id, .. } => {
                self.resting_orders.remove(order_id);
            }
            ExchangeMessage::OrderRejected { .. } => {}
        }
    }
}

#[cfg(test)]
#[allow(clippy::cast_precision_loss)]
mod tests {
    use super::*;

    fn default_cfg() -> MarketMakerConfig {
        MarketMakerConfig {
            total_liquidity: 100.0,
            step_size_ratio: 0.001,
            max_levels: 3,
            imbalance_beta: 1.0,
            peak_distance_ratio: 0.05,
            shape_exponent: 1.2,
            mean_wakeup_interval_ns: 5_000_000_000,
            reference_price: 10_000,
            symbol: 0,
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

    // ── Inventory imbalance ──────────────────────────────────────────

    #[test]
    fn zero_inventory_gives_zero_imbalance() {
        let agent = MarketMakerAgent::new(default_cfg(), 42);
        assert!((agent.imbalance()).abs() < 1e-10);
    }

    #[test]
    fn positive_inventory_gives_positive_imbalance() {
        let mut agent = MarketMakerAgent::new(default_cfg(), 42);
        agent.inventory = 50;
        assert!(agent.imbalance() > 0.0);
    }

    #[test]
    fn negative_inventory_gives_negative_imbalance() {
        let mut agent = MarketMakerAgent::new(default_cfg(), 42);
        agent.inventory = -50;
        assert!(agent.imbalance() < 0.0);
    }

    #[test]
    fn imbalance_bounded_by_one() {
        let mut agent = MarketMakerAgent::new(default_cfg(), 42);
        agent.inventory = 1_000_000;
        assert!(agent.imbalance() <= 1.0);
        assert!(agent.imbalance() >= -1.0);
        agent.inventory = -1_000_000;
        assert!(agent.imbalance() <= 1.0);
        assert!(agent.imbalance() >= -1.0);
    }

    // ── Order placement ──────────────────────────────────────────────

    #[test]
    fn places_orders_on_both_sides() {
        let mut agent = MarketMakerAgent::new(default_cfg(), 42);
        let snaps = snap_at_price(10_000);
        let mut actions = Vec::new();
        agent.wakeup_into(0, 0, &snaps, &mut actions);

        let bids: Vec<_> = actions
            .iter()
            .filter(|a| matches!(a, AgentAction::SubmitOrder {
                order: OrderAction::NewLimitOrder { side: Side::Bid, .. }, ..
            }))
            .collect();
        let asks: Vec<_> = actions
            .iter()
            .filter(|a| matches!(a, AgentAction::SubmitOrder {
                order: OrderAction::NewLimitOrder { side: Side::Ask, .. }, ..
            }))
            .collect();

        assert!(!bids.is_empty(), "should place bid orders");
        assert!(!asks.is_empty(), "should place ask orders");
    }

    #[test]
    fn places_correct_number_of_levels() {
        let mut agent = MarketMakerAgent::new(default_cfg(), 42);
        let snaps = snap_at_price(10_000);
        let mut actions = Vec::new();
        agent.wakeup_into(0, 0, &snaps, &mut actions);

        let orders: Vec<_> = actions
            .iter()
            .filter(|a| matches!(a, AgentAction::SubmitOrder { .. }))
            .collect();
        assert!(
            orders.len() <= 6,
            "should place at most 2 * max_levels = 6 orders, got {}",
            orders.len()
        );
        assert!(
            orders.len() >= 2,
            "should place at least 1 bid + 1 ask, got {}",
            orders.len()
        );
    }

    #[test]
    fn bid_prices_below_mid_ask_prices_above() {
        let mut agent = MarketMakerAgent::new(default_cfg(), 42);
        let mid = 10_000;
        let snaps = snap_at_price(mid);
        let mut actions = Vec::new();
        agent.wakeup_into(0, 0, &snaps, &mut actions);

        for a in &actions {
            if let AgentAction::SubmitOrder {
                order: OrderAction::NewLimitOrder { side, price, .. }, ..
            } = a
            {
                match side {
                    Side::Bid => assert!(
                        *price < mid,
                        "bid price {price} should be below mid {mid}"
                    ),
                    Side::Ask => assert!(
                        *price > mid,
                        "ask price {price} should be above mid {mid}"
                    ),
                }
            }
        }
    }

    // ── Total liquidity budget ──────────────────────────────────────

    #[test]
    fn total_qty_approximately_matches_liquidity_budget() {
        let mut agent = MarketMakerAgent::new(default_cfg(), 42);
        let snaps = snap_at_price(10_000);
        let mut actions = Vec::new();
        agent.wakeup_into(0, 0, &snaps, &mut actions);

        let total_qty: u64 = actions
            .iter()
            .filter_map(|a| match a {
                AgentAction::SubmitOrder {
                    order: OrderAction::NewLimitOrder { qty, .. }, ..
                } => Some(*qty),
                _ => None,
            })
            .sum();

        let budget = default_cfg().total_liquidity;
        assert!(
            (total_qty as f64 - budget).abs() / budget < 0.1,
            "total qty {total_qty} should be near budget {budget}"
        );
    }

    // ── Cancel-all before placing ───────────────────────────────────

    #[test]
    fn cancels_all_resting_before_placing() {
        let mut agent = MarketMakerAgent::new(default_cfg(), 42);
        agent.on_exchange_message(0, 0, ExchangeMessage::OrderAccepted { order_id: 10, user_id: 0, symbol: 0, side: Side::Bid, qty: 10 });
        agent.on_exchange_message(0, 0, ExchangeMessage::OrderAccepted { order_id: 20, user_id: 0, symbol: 0, side: Side::Bid, qty: 10 });
        agent.on_exchange_message(0, 0, ExchangeMessage::OrderAccepted { order_id: 30, user_id: 0, symbol: 0, side: Side::Bid, qty: 10 });

        let snaps = snap_at_price(10_000);
        let mut actions = Vec::new();
        agent.wakeup_into(0, 0, &snaps, &mut actions);

        let cancels: Vec<_> = actions
            .iter()
            .filter(|a| matches!(a, AgentAction::CancelOrder { .. }))
            .collect();
        assert_eq!(cancels.len(), 3, "should cancel all 3 resting orders");

        let first_cancel_idx = actions
            .iter()
            .position(|a| matches!(a, AgentAction::CancelOrder { .. }))
            .unwrap();
        let first_submit_idx = actions
            .iter()
            .position(|a| matches!(a, AgentAction::SubmitOrder { .. }))
            .unwrap();
        assert!(
            first_cancel_idx < first_submit_idx,
            "cancels should precede submits"
        );
    }

    // ── Per-side normalization ─────────────────────────────────────

    #[test]
    fn per_side_normalization_balances_sides() {
        let cfg = default_cfg();
        let mut agent = MarketMakerAgent::new(cfg, 42);
        agent.inventory = 200;

        let snaps = snap_at_price(10_000);
        let mut actions = Vec::new();
        agent.wakeup_into(0, 0, &snaps, &mut actions);

        let sum_qty = |actions: &[AgentAction], side: Side| -> u64 {
            actions
                .iter()
                .filter_map(|a| match a {
                    AgentAction::SubmitOrder {
                        order: OrderAction::NewLimitOrder { side: s, qty, .. }, ..
                    } if *s == side => Some(*qty),
                    _ => None,
                })
                .sum()
        };

        let bid_qty = sum_qty(&actions, Side::Bid);
        let ask_qty = sum_qty(&actions, Side::Ask);
        let half = default_cfg().total_liquidity * 0.5;

        assert!(
            (bid_qty as f64 - half).abs() / half < 0.15,
            "bid qty {bid_qty} should be near half budget {half}"
        );
        assert!(
            (ask_qty as f64 - half).abs() / half < 0.15,
            "ask qty {ask_qty} should be near half budget {half}"
        );
    }

    // ── Geometric mid-price ─────────────────────────────────────────

    #[test]
    #[allow(clippy::cast_possible_truncation)]
    fn geometric_mid_price_used() {
        let snap = MarketSnapshot {
            best_bid: Some((9000, 100)),
            best_ask: Some((11000, 100)),
            last_trade_price: Some(10000),
            last_trade_time: Some(0),
        };
        let mid = MarketMakerAgent::geometric_mid(&snap, 10_000);
        let expected = (9000.0_f64 * 11000.0).sqrt().round() as i64;
        assert_eq!(mid, expected, "should use geometric mean");
        assert_ne!(mid, 10000, "should differ from arithmetic mean");
    }

    #[test]
    fn geometric_mid_fallback_on_empty_book() {
        let snap = MarketSnapshot {
            best_bid: None,
            best_ask: None,
            last_trade_price: None,
            last_trade_time: None,
        };
        let mid = MarketMakerAgent::geometric_mid(&snap, 10_000);
        assert_eq!(mid, 10_000, "should fall back to reference price");
    }

    // ── Fill tracking updates inventory by side ─────────────────────

    #[test]
    fn bid_fill_increases_inventory() {
        let mut agent = MarketMakerAgent::new(default_cfg(), 42);
        assert_eq!(agent.inventory, 0);

        agent.on_exchange_message(
            0, 0,
            ExchangeMessage::OrderFilled {
                order_id: 1, user_id: 0, symbol: 0, side: Side::Bid, price: 100, qty: 10, remaining: 0,
            },
        );
        assert_eq!(agent.inventory, 10, "bid fill should increase inventory");
    }

    #[test]
    fn ask_fill_decreases_inventory() {
        let mut agent = MarketMakerAgent::new(default_cfg(), 42);
        assert_eq!(agent.inventory, 0);

        agent.on_exchange_message(
            0, 0,
            ExchangeMessage::OrderFilled {
                order_id: 1, user_id: 0, symbol: 0, side: Side::Ask, price: 100, qty: 10, remaining: 0,
            },
        );
        assert_eq!(agent.inventory, -10, "ask fill should decrease inventory");
    }

    #[test]
    fn mixed_fills_track_net_inventory() {
        let mut agent = MarketMakerAgent::new(default_cfg(), 42);

        agent.on_exchange_message(
            0, 0,
            ExchangeMessage::OrderFilled {
                order_id: 1, user_id: 0, symbol: 0, side: Side::Bid, price: 100, qty: 10, remaining: 0,
            },
        );
        agent.on_exchange_message(
            0, 0,
            ExchangeMessage::OrderFilled {
                order_id: 2, user_id: 0, symbol: 0, side: Side::Ask, price: 101, qty: 7, remaining: 0,
            },
        );
        assert_eq!(agent.inventory, 3, "net inventory should be +10 - 7 = +3");
    }

    // The exchange sends no accept-then-fill correlation requirement any
    // more: a fill that arrives before (or without) its accept still counts,
    // because it carries its own side.
    #[test]
    fn fill_without_prior_accept_still_counts() {
        let mut agent = MarketMakerAgent::new(default_cfg(), 42);
        agent.on_exchange_message(
            0, 0,
            ExchangeMessage::OrderFilled {
                order_id: 999, user_id: 0, symbol: 0, side: Side::Bid, price: 100, qty: 10, remaining: 0,
            },
        );
        assert_eq!(agent.inventory, 10, "self-contained fill must count");
    }

    // ── Custom samplers via with_samplers ────────────────────────────

    #[test]
    fn custom_weight_model_and_wakeup() {
        use crate::samplers::FixedIntervalWakeup;

        let cfg = default_cfg();
        let mut agent = MarketMakerAgent::with_samplers(
            cfg,
            42,
            Box::new(SymmetricHumpModel::new(0.1, 2.0)),
            Box::new(FixedIntervalWakeup::new(777_000_000)),
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
        assert_eq!(wakeups[0], 777_000_000, "custom fixed wakeup should be used");
    }
}
