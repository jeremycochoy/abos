use std::collections::{HashMap, VecDeque};

use rand::rngs::SmallRng;
use rand::SeedableRng;
use rand_distr::{Distribution, Exp};

use cda_engine::Side;
use sim_core::{Agent, AgentAction, AgentId, ExchangeMessage, MarketSnapshot, Nanos, OrderAction};

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

/// Liquidity market-maker agent using a symmetric-hump distribution.
///
/// Places multiple bid and ask orders at different price levels around the
/// geometric mid-price `sqrt(bid * ask)`. Adjusts liquidity distribution
/// based on inventory imbalance, with per-side normalization ensuring each
/// side receives exactly half the total liquidity budget.
pub struct MarketMakerAgent {
    cfg: MarketMakerConfig,
    rng: SmallRng,
    resting_orders: IndexedSet,
    wakeup_dist: Exp<f64>,
    /// Net inventory (positive = long, negative = short).
    inventory: i64,
    /// Maps `order_id` → `Side` so we know the direction when a fill arrives.
    order_sides: HashMap<u64, Side>,
    /// Sides of pending submissions, matched to `OrderAccepted` in FIFO order.
    pending_sides: VecDeque<Side>,
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
            order_sides: HashMap::new(),
            pending_sides: VecDeque::new(),
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
            self.order_sides.remove(&oid);
            actions.push(AgentAction::CancelOrder {
                symbol: self.cfg.symbol,
                order_id: oid,
            });
        }

        let imbalance = self.imbalance();
        let log_step = (1.0 + self.cfg.step_size_ratio).ln();

        // Compute weights for all levels
        let mut bid_weights = Vec::with_capacity(self.cfg.max_levels);
        let mut ask_weights = Vec::with_capacity(self.cfg.max_levels);

        for level in 1..=self.cfg.max_levels {
            let log_dist = log_step * level as f64;
            let base = self.hump_weight(log_dist);
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
                self.pending_sides.push_back(Side::Bid);
                actions.push(AgentAction::SubmitOrder {
                    symbol: self.cfg.symbol,
                    order: OrderAction::NewLimitOrder { side: Side::Bid, price, qty },
                });
            }
        }

        // Place ask orders
        if ask_total > 1e-15 {
            for &(log_dist, weight) in &ask_weights {
                let qty = (half_liq * weight / ask_total).round() as u64;
                if qty == 0 { continue; }
                let price = (mid as f64 * log_dist.exp()).round() as i64;
                self.pending_sides.push_back(Side::Ask);
                actions.push(AgentAction::SubmitOrder {
                    symbol: self.cfg.symbol,
                    order: OrderAction::NewLimitOrder { side: Side::Ask, price, qty },
                });
            }
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
                if let Some(side) = self.pending_sides.pop_front() {
                    self.order_sides.insert(order_id, side);
                }
            }
            ExchangeMessage::OrderFilled { order_id, qty, .. } => {
                self.resting_orders.remove(order_id);
                // Update inventory based on the side of the filled order:
                // bid fill → bought → inventory increases
                // ask fill → sold → inventory decreases
                #[allow(clippy::cast_possible_wrap)]
                if let Some(side) = self.order_sides.remove(&order_id) {
                    match side {
                        Side::Bid => self.inventory += qty as i64,
                        Side::Ask => self.inventory -= qty as i64,
                    }
                }
            }
            ExchangeMessage::OrderCancelled { order_id } => {
                self.resting_orders.remove(order_id);
                self.order_sides.remove(&order_id);
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

    // ── Hump weight function ──────────────────────────────────────────

    #[test]
    fn hump_weight_is_positive_for_nonzero_distance() {
        let agent = MarketMakerAgent::new(default_cfg(), 42);
        for i in 1..=10 {
            let d = 0.001 * f64::from(i);
            let w = agent.hump_weight(d);
            assert!(w > 0.0, "hump_weight({d}) = {w}, expected > 0");
        }
    }

    #[test]
    fn hump_weight_peaks_then_decays() {
        // The hump model: w = (d + eps)^exponent * exp(-decay * d)
        // Should rise from near-zero, peak, then decay at large distances.
        let agent = MarketMakerAgent::new(default_cfg(), 42);
        let near = agent.hump_weight(0.001);
        let mid_d = agent.hump_weight(0.05); // near peak (peak_distance_ratio=0.05)
        let far = agent.hump_weight(0.5);
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
    fn hump_weight_is_symmetric() {
        let agent = MarketMakerAgent::new(default_cfg(), 42);
        let positive = agent.hump_weight(0.02);
        let negative = agent.hump_weight(-0.02);
        assert!(
            (positive - negative).abs() < 1e-15,
            "hump weight should be symmetric: {positive} vs {negative}"
        );
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
        // Extremely large inventory
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
        // max_levels=3 per side → up to 6 orders (some might be 0 qty and skipped)
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

        // Total qty should be approximately total_liquidity (100)
        // Rounding can cause small deviations
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
        // Simulate accepted orders
        agent.on_exchange_message(0, 0, ExchangeMessage::OrderAccepted { order_id: 10 });
        agent.on_exchange_message(0, 0, ExchangeMessage::OrderAccepted { order_id: 20 });
        agent.on_exchange_message(0, 0, ExchangeMessage::OrderAccepted { order_id: 30 });

        let snaps = snap_at_price(10_000);
        let mut actions = Vec::new();
        agent.wakeup_into(0, 0, &snaps, &mut actions);

        let cancels: Vec<_> = actions
            .iter()
            .filter(|a| matches!(a, AgentAction::CancelOrder { .. }))
            .collect();
        assert_eq!(cancels.len(), 3, "should cancel all 3 resting orders");

        // Cancels should come before submits
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
        // With per-side normalization (matching ABIDES), each side should get
        // approximately half the total liquidity, even with inventory imbalance.
        let cfg = default_cfg();
        let mut agent = MarketMakerAgent::new(cfg, 42);
        agent.inventory = 200; // strongly long

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

        // Each side should be approximately half, regardless of imbalance
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
        // geometric mean of 9000 and 11000 = sqrt(9000*11000) ≈ 9949
        // arithmetic mean would be 10000
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

        // Register a bid order's side
        agent.pending_sides.push_back(Side::Bid);
        agent.on_exchange_message(0, 0, ExchangeMessage::OrderAccepted { order_id: 1 });
        agent.on_exchange_message(
            0, 0,
            ExchangeMessage::OrderFilled { order_id: 1, price: 100, qty: 10 },
        );
        assert_eq!(agent.inventory, 10, "bid fill should increase inventory");
    }

    #[test]
    fn ask_fill_decreases_inventory() {
        let mut agent = MarketMakerAgent::new(default_cfg(), 42);
        assert_eq!(agent.inventory, 0);

        // Register an ask order's side
        agent.pending_sides.push_back(Side::Ask);
        agent.on_exchange_message(0, 0, ExchangeMessage::OrderAccepted { order_id: 1 });
        agent.on_exchange_message(
            0, 0,
            ExchangeMessage::OrderFilled { order_id: 1, price: 100, qty: 10 },
        );
        assert_eq!(agent.inventory, -10, "ask fill should decrease inventory");
    }

    #[test]
    fn mixed_fills_track_net_inventory() {
        let mut agent = MarketMakerAgent::new(default_cfg(), 42);

        // Buy 10, sell 7 → net +3
        agent.pending_sides.push_back(Side::Bid);
        agent.pending_sides.push_back(Side::Ask);
        agent.on_exchange_message(0, 0, ExchangeMessage::OrderAccepted { order_id: 1 });
        agent.on_exchange_message(0, 0, ExchangeMessage::OrderAccepted { order_id: 2 });
        agent.on_exchange_message(
            0, 0,
            ExchangeMessage::OrderFilled { order_id: 1, price: 100, qty: 10 },
        );
        agent.on_exchange_message(
            0, 0,
            ExchangeMessage::OrderFilled { order_id: 2, price: 101, qty: 7 },
        );
        assert_eq!(agent.inventory, 3, "net inventory should be +10 - 7 = +3");
    }

    #[test]
    fn unknown_fill_does_not_change_inventory() {
        // If we receive a fill for an order we don't have side info for,
        // inventory should not change (defensive behavior).
        let mut agent = MarketMakerAgent::new(default_cfg(), 42);
        agent.on_exchange_message(
            0, 0,
            ExchangeMessage::OrderFilled { order_id: 999, price: 100, qty: 10 },
        );
        assert_eq!(agent.inventory, 0, "unknown fill should not change inventory");
    }
}
