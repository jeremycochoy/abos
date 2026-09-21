//! Golden fingerprints of full simulation runs (issue #10).
//!
//! Each test folds every output of a seeded run into one u64: the trade log,
//! the L1 log or buckets, the event count, the result end time, and the
//! message and wakeup history of a probe agent. The constants were recorded
//! on master at b98092d. A performance change must keep every fingerprint,
//! which proves the run byte-identical, event order included.

use std::cell::Cell;
use std::rc::Rc;

use agents::{
    MarketMakerAgent, MarketMakerConfig, TrendFollowingAgent, TrendFollowingConfig, ZiAgent,
    ZiAgentConfig,
};
use cda_engine::Side;
use sim_core::{
    Agent, AgentAction, AgentId, ExchangeMessage, Kernel, LatencyConfig, LatencyModelType,
    MarketSnapshot, Nanos, OrderAction, RunOptions, SimulationConfig, SimulationResult,
};

const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

/// FNV-1a fold of one word into a running hash.
fn fold(h: u64, word: u64) -> u64 {
    (h ^ word).wrapping_mul(FNV_PRIME)
}

#[allow(clippy::cast_sign_loss)]
fn fold_i64(h: u64, word: i64) -> u64 {
    fold(h, word as u64)
}

/// A deterministic agent that trades on symbol 0 and records everything it
/// sees: wakeup times, the snapshots of every market, and each exchange
/// message with its delivery time. The hash lives behind an `Rc` so the test
/// reads it after the kernel consumed the agent.
struct ProbeAgent {
    hash: Rc<Cell<u64>>,
    counter: u64,
    open_orders: Vec<u64>,
}

impl ProbeAgent {
    fn new(hash: Rc<Cell<u64>>) -> Self {
        Self { hash, counter: 0, open_orders: Vec::new() }
    }

    fn fold_in(&self, word: u64) {
        self.hash.set(fold(self.hash.get(), word));
    }

    fn fold_snapshot(&self, snap: &MarketSnapshot) {
        let (bp, bv) = snap.best_bid.unwrap_or((-1, 0));
        let (ap, av) = snap.best_ask.unwrap_or((-1, 0));
        #[allow(clippy::cast_sign_loss)]
        {
            self.fold_in(bp as u64);
            self.fold_in(bv);
            self.fold_in(ap as u64);
            self.fold_in(av);
            self.fold_in(snap.last_trade_price.unwrap_or(-1) as u64);
        }
        self.fold_in(snap.last_trade_time.unwrap_or(u64::MAX));
    }
}

impl Agent for ProbeAgent {
    fn wakeup_into(
        &mut self,
        time: Nanos,
        _agent_id: AgentId,
        snapshots: &[MarketSnapshot],
        actions: &mut Vec<AgentAction>,
    ) {
        self.fold_in(0xAAAA);
        self.fold_in(time);
        for snap in snapshots {
            self.fold_snapshot(snap);
        }
        self.counter += 1;
        let mid = snapshots[0]
            .best_bid
            .zip(snapshots[0].best_ask)
            .map_or(10_000, |((b, _), (a, _))| (b + a) / 2);
        let side = if self.counter.is_multiple_of(2) { Side::Bid } else { Side::Ask };
        let offset = 1 + (self.counter % 5) as i64;
        let price = if side == Side::Bid { mid - offset } else { mid + offset };
        actions.push(AgentAction::SubmitOrder {
            symbol: 0,
            order: OrderAction::NewLimitOrder {
                side,
                price,
                qty: 1 + self.counter % 3,
                user_id: self.counter,
            },
        });
        if self.counter.is_multiple_of(3) {
            if let Some(order_id) = self.open_orders.pop() {
                actions.push(AgentAction::CancelOrder { symbol: 0, order_id });
            }
        }
        actions.push(AgentAction::ScheduleWakeUp { delay_ns: 7_777_777 });
    }

    fn on_exchange_message(&mut self, time: Nanos, _agent_id: AgentId, message: ExchangeMessage) {
        self.fold_in(0xBBBB);
        self.fold_in(time);
        match message {
            ExchangeMessage::OrderAccepted { order_id, user_id, symbol, side, qty } => {
                self.fold_in(1);
                self.fold_in(order_id);
                self.fold_in(user_id);
                self.fold_in(u64::from(symbol));
                self.fold_in(u64::from(side == Side::Bid));
                self.fold_in(qty);
                self.open_orders.push(order_id);
            }
            ExchangeMessage::OrderFilled { order_id, user_id, symbol, side, price, qty, remaining } => {
                self.fold_in(2);
                self.fold_in(order_id);
                self.fold_in(user_id);
                self.fold_in(u64::from(symbol));
                self.fold_in(u64::from(side == Side::Bid));
                #[allow(clippy::cast_sign_loss)]
                self.fold_in(price as u64);
                self.fold_in(qty);
                self.fold_in(remaining);
            }
            ExchangeMessage::OrderCancelled { order_id, user_id, symbol } => {
                self.fold_in(3);
                self.fold_in(order_id);
                self.fold_in(user_id);
                self.fold_in(u64::from(symbol));
            }
            ExchangeMessage::OrderRejected { order_id, user_id, symbol } => {
                self.fold_in(4);
                self.fold_in(order_id);
                self.fold_in(user_id);
                self.fold_in(u64::from(symbol));
            }
        }
    }
}

/// Fold every field of the result into one hash.
fn result_fingerprint(result: &SimulationResult, probe_hash: u64) -> u64 {
    let mut h = FNV_OFFSET;
    h = fold(h, result.events_processed);
    h = fold(h, result.end_time);
    h = fold(h, result.trades.len() as u64);
    for t in &result.trades {
        h = fold(h, t.timestamp);
        h = fold(h, u64::from(t.symbol));
        h = fold_i64(h, t.price);
        h = fold(h, t.qty);
        h = fold(h, u64::from(t.aggressor_side == Side::Bid));
        h = fold(h, t.maker_order_id);
        h = fold(h, t.taker_order_id);
    }
    h = fold(h, result.l1_snapshots.len() as u64);
    for s in &result.l1_snapshots {
        h = fold(h, s.timestamp);
        h = fold(h, u64::from(s.symbol));
        h = fold_i64(h, s.bid_price);
        h = fold_i64(h, s.ask_price);
        h = fold(h, s.bid_volume);
        h = fold(h, s.ask_volume);
        h = fold_i64(h, s.last_trade_price);
    }
    h = fold(h, result.l1_buckets.len() as u64);
    for b in &result.l1_buckets {
        h = fold(h, b.bucket_start);
        h = fold(h, u64::from(b.symbol));
        h = fold_i64(h, b.first_bid);
        h = fold_i64(h, b.min_bid);
        h = fold_i64(h, b.last_bid);
        h = fold_i64(h, b.first_ask);
        h = fold_i64(h, b.max_ask);
        h = fold_i64(h, b.last_ask);
        h = fold(h, b.volume);
        h = fold(h, b.quote_volume as u64);
        h = fold(h, (b.quote_volume >> 64) as u64);
    }
    fold(h, probe_hash)
}

fn zi(symbol: u32, seed: u64) -> Box<dyn Agent> {
    Box::new(ZiAgent::new(
        ZiAgentConfig {
            wake_up_interval_ns: 5_000_000,
            price_std: 0.002,
            order_size_scale: 3.0,
            order_size_std: 0.8,
            reference_price: 10_000,
            symbol,
        },
        seed,
    ))
}

fn maker(symbol: u32, seed: u64) -> Box<dyn Agent> {
    Box::new(MarketMakerAgent::new(
        MarketMakerConfig {
            total_liquidity: 60.0,
            step_size_ratio: 0.001,
            max_levels: 3,
            imbalance_beta: 0.4,
            peak_distance_ratio: 0.05,
            shape_exponent: 1.5,
            mean_wakeup_interval_ns: 8_000_000,
            reference_price: 10_000,
            symbol,
        },
        seed,
    ))
}

fn trend(symbol: u32, seed: u64) -> Box<dyn Agent> {
    Box::new(TrendFollowingAgent::new(
        TrendFollowingConfig {
            short_window: 3,
            long_window: 9,
            threshold: 0.0002,
            price_offset: 0.01,
            order_size_factor: 40.0,
            order_size_boost: 2.0,
            mean_wakeup_interval_ns: 15_000_000,
            sampling_freq_ns: 5_000_000,
            reference_price: 10_000,
            symbol,
            contrarian: false,
        },
        seed,
    ))
}

/// Two markets with ZI, maker and trend agents plus the probe.
fn build_agents(probe_hash: &Rc<Cell<u64>>) -> Vec<Box<dyn Agent>> {
    let mut agents: Vec<Box<dyn Agent>> = Vec::new();
    for symbol in 0..2 {
        for k in 0..6 {
            agents.push(zi(symbol, 1000 + u64::from(symbol) * 100 + k));
        }
        agents.push(maker(symbol, 2000 + u64::from(symbol)));
        agents.push(trend(symbol, 3000 + u64::from(symbol)));
    }
    agents.push(Box::new(ProbeAgent::new(Rc::clone(probe_hash))));
    agents
}

fn config(model: LatencyModelType, no_market_hours: bool) -> SimulationConfig {
    SimulationConfig {
        seed: 20260921,
        start_time: 0,
        end_time: 2_000_000_000,
        symbols: vec![0, 1],
        latency: LatencyConfig {
            default_base_ns: 50_000,
            jitter_mu: 0.0,
            jitter_sigma: 0.3,
            model,
        },
        tick_size: 100,
        lot_size: 1,
        no_market_hours,
    }
}

fn run_fingerprint(model: LatencyModelType, no_market_hours: bool, options: &RunOptions) -> u64 {
    let probe_hash = Rc::new(Cell::new(FNV_OFFSET));
    let agents = build_agents(&probe_hash);
    let result = Kernel::run_with(&config(model, no_market_hours), agents, options);
    result_fingerprint(&result, probe_hash.get())
}

#[test]
fn gen88_like_run_keeps_its_fingerprint() {
    let got = run_fingerprint(
        LatencyModelType::NycSeattle { seed: 20260921 },
        true,
        &RunOptions::default(),
    );
    assert_eq!(got, 0x5be1_136c_e8c0_ab09, "fingerprint changed: {got:#018x}");
}

#[test]
fn lean_bucket_run_keeps_its_fingerprint() {
    let got = run_fingerprint(
        LatencyModelType::NycSeattle { seed: 20260921 },
        true,
        &RunOptions { keep_trades: false, l1_bucket_ns: Some(1_000_000_000) },
    );
    assert_eq!(got, 0xaa8d_4841_1311_7022, "fingerprint changed: {got:#018x}");
}

#[test]
fn market_hours_run_keeps_its_fingerprint() {
    let got = run_fingerprint(
        LatencyModelType::Uniform,
        false,
        &RunOptions::default(),
    );
    assert_eq!(got, 0x9c59_ea34_8966_f5cd, "fingerprint changed: {got:#018x}");
}

#[test]
fn no_latency_run_keeps_its_fingerprint() {
    let got = run_fingerprint(LatencyModelType::NoLatency, true, &RunOptions::default());
    assert_eq!(got, 0xe84d_4df1_e7a4_429e, "fingerprint changed: {got:#018x}");
}
