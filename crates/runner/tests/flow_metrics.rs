use std::cell::RefCell;
use std::rc::Rc;

type Seen = Rc<RefCell<Vec<ExchangeMessage>>>;

use cda_engine::Side;
use sim_core::{
    Agent, AgentAction, AgentId, ExchangeMessage, FlowBucket, FlowOptions, Kernel, LatencyConfig,
    LatencyModelType, MarketSnapshot, Nanos, OrderAction, RunOptions, SimulationConfig,
};

const END_TIME: Nanos = 1_000_000;
const WATCHED: AgentId = 2;

struct ScriptAgent {
    script: Vec<(Nanos, OrderAction)>,
    cursor: usize,
    seen: Seen,
}

impl ScriptAgent {
    fn new(script: Vec<(Nanos, OrderAction)>) -> (Self, Seen) {
        let seen = Rc::new(RefCell::new(Vec::new()));
        (
            Self {
                script,
                cursor: 0,
                seen: Rc::clone(&seen),
            },
            seen,
        )
    }
}

impl Agent for ScriptAgent {
    fn wakeup_into(
        &mut self,
        time: Nanos,
        _agent_id: AgentId,
        _snapshots: &[MarketSnapshot],
        actions: &mut Vec<AgentAction>,
    ) {
        while self.cursor < self.script.len() && self.script[self.cursor].0 == time {
            actions.push(AgentAction::SubmitOrder {
                symbol: 0,
                order: self.script[self.cursor].1,
            });
            self.cursor += 1;
        }
        if self.cursor < self.script.len() {
            actions.push(AgentAction::ScheduleWakeUp {
                delay_ns: self.script[self.cursor].0 - time,
            });
        }
    }

    fn on_exchange_message(&mut self, _time: Nanos, _agent_id: AgentId, message: ExchangeMessage) {
        self.seen.borrow_mut().push(message);
    }
}

fn limit(side: Side, price: i64, qty: u64) -> OrderAction {
    OrderAction::NewLimitOrder {
        side,
        price,
        qty,
        user_id: 0,
    }
}

fn config() -> SimulationConfig {
    SimulationConfig {
        seed: 9,
        start_time: 0,
        end_time: END_TIME,
        symbols: vec![0],
        latency: LatencyConfig {
            default_base_ns: 0,
            jitter_mu: 0.0,
            jitter_sigma: 0.0,
            model: LatencyModelType::NoLatency,
        },
        tick_size: 100,
        lot_size: 1,
        no_market_hours: true,
    }
}

fn agents() -> (Vec<Box<dyn Agent>>, Seen) {
    let (maker_a, _) = ScriptAgent::new(vec![
        (0, limit(Side::Ask, 10_100, 10)),
        (0, limit(Side::Ask, 10_200, 20)),
        (0, limit(Side::Bid, 9_900, 10)),
    ]);
    let (maker_b, _) = ScriptAgent::new(vec![
        (0, limit(Side::Ask, 10_100, 5)),
        (0, limit(Side::Bid, 9_800, 5)),
    ]);
    let (watched, watched_seen) = ScriptAgent::new(vec![
        (1_000, limit(Side::Bid, 10_200, 20)),
        (1_000, limit(Side::Ask, 10_150, 8)),
        (3_000, limit(Side::Bid, 10_300, 20)),
    ]);
    let (outsider, _) = ScriptAgent::new(vec![(2_000, limit(Side::Bid, 10_150, 3))]);
    (
        vec![
            Box::new(maker_a),
            Box::new(maker_b),
            Box::new(watched),
            Box::new(outsider),
        ],
        watched_seen,
    )
}

fn flow_options(interval_ns: Nanos) -> FlowOptions {
    FlowOptions {
        agent_id: WATCHED,
        interval_ns,
        horizons_ns: vec![500, 10_000],
        ratio_edges: vec![0.25, 0.5, 0.75, 1.0],
    }
}

fn run(interval_ns: Nanos) -> (Vec<FlowBucket>, Seen) {
    let (agents, seen) = agents();
    let options = RunOptions {
        flow: Some(flow_options(interval_ns)),
        ..RunOptions::default()
    };
    let result = Kernel::run_with(&config(), agents, &options);
    (result.flow_buckets, seen)
}

#[test]
fn one_bucket_counts_market_watched_and_self_flows_exactly() {
    let (buckets, _) = run(END_TIME);
    assert_eq!(buckets.len(), 1);
    let bucket = &buckets[0];
    assert_eq!((bucket.start, bucket.symbol), (0, 0));
    assert_eq!(bucket.market_trades, 5);
    assert_eq!(bucket.market_qty, 38);
    assert_eq!(
        bucket.market_notional,
        10 * 10_100 + 5 * 10_100 + 5 * 10_200 + 3 * 10_150 + 15 * 10_200
    );
    assert_eq!(bucket.self_trades, 1);
    assert_eq!(bucket.self_qty, 5);
    assert_eq!(bucket.self_notional, 5 * 10_150);
    assert_eq!(bucket.taker_qty, 35);
    assert_eq!(bucket.taker_notional, 202_500 + 153_000);
    assert_eq!(bucket.maker_qty, 3);
    assert_eq!(bucket.maker_notional, 3 * 10_150);
    assert_eq!(bucket.orders, 3);
    assert_eq!(bucket.filled_orders, 3);
    assert_eq!(bucket.fill_events, 6);
    assert_eq!(bucket.submitted_qty, 48);
    assert_eq!(bucket.zero_depth_qty, 8);
    assert_eq!(bucket.ratio_qty, vec![0, 0, 20, 0, 20]);
}

#[test]
fn shortfall_sums_signed_cost_times_notional_against_the_pre_trade_mid() {
    let (buckets, _) = run(END_TIME);
    let bucket = &buckets[0];
    let first = (10_100.0 - 10_000.0) / 10_000.0 * 101_000.0
        + (10_100.0 - 10_000.0) / 10_000.0 * 50_500.0
        + (10_200.0 - 10_000.0) / 10_000.0 * 51_000.0;
    let second = (10_200.0 - 10_025.0) / 10_025.0 * 153_000.0;
    assert!((bucket.shortfall - (first + second)).abs() < 1e-9);
    assert_eq!(bucket.shortfall_base, 202_500 + 153_000);
}

#[test]
fn responses_use_the_mid_prevailing_at_each_horizon() {
    let (buckets, _) = run(END_TIME);
    let bucket = &buckets[0];
    let buy_moves = (101_000.0 + 50_500.0 + 51_000.0) * (10_025.0 - 10_000.0) / 10_000.0;
    let flat_sale = -30_450.0 * (10_025.0 - 10_025.0) / 10_025.0;
    let flat_buy = 153_000.0 * (10_025.0 - 10_025.0) / 10_025.0;
    for horizon in 0..2 {
        assert!(
            (bucket.response_num[horizon] - (buy_moves + flat_sale + flat_buy)).abs() < 1e-9,
            "horizon {horizon}: {}",
            bucket.response_num[horizon]
        );
        assert!((bucket.response_den[horizon] - 385_950.0).abs() < 1e-9);
    }
}

#[test]
fn the_taker_fill_message_carries_the_exact_notional_across_levels() {
    let (_, seen) = run(END_TIME);
    let fills: Vec<(u64, i64, u128)> = seen
        .borrow()
        .iter()
        .filter_map(|message| match *message {
            ExchangeMessage::OrderFilled {
                side: Side::Bid,
                price,
                qty,
                notional,
                ..
            } => Some((qty, price, notional)),
            _ => None,
        })
        .collect();
    assert_eq!(fills[0], (20, 10_200, 202_500));
    assert_ne!(202_500, 20 * 10_200);
    assert_eq!(fills[1], (20, 10_200, 203_750));
}

#[test]
fn the_maker_fill_message_carries_its_own_notional() {
    let (_, seen) = run(END_TIME);
    let sales: Vec<(u64, i64, u128)> = seen
        .borrow()
        .iter()
        .filter_map(|message| match *message {
            ExchangeMessage::OrderFilled {
                side: Side::Ask,
                price,
                qty,
                notional,
                ..
            } => Some((qty, price, notional)),
            _ => None,
        })
        .collect();
    assert_eq!(
        sales,
        vec![(3, 10_150, 3 * 10_150), (5, 10_150, 5 * 10_150)]
    );
}

#[test]
fn short_intervals_split_the_flow_and_cover_the_whole_run() {
    let (buckets, _) = run(2_000);
    assert_eq!(buckets.len(), 500);
    assert!(buckets
        .iter()
        .enumerate()
        .all(|(i, b)| b.start == 2_000 * i as u64));
    assert_eq!(buckets[0].market_trades, 3);
    assert_eq!(buckets[0].orders, 2);
    assert_eq!(buckets[1].market_trades, 2);
    assert_eq!(buckets[1].self_trades, 1);
    assert_eq!(buckets[1].orders, 1);
    assert!(buckets[2..]
        .iter()
        .all(|b| b.market_trades == 0 && b.orders == 0));
    let total_num: f64 = buckets.iter().map(|b| b.response_num[0]).sum();
    let single = run(END_TIME).0;
    assert!((total_num - single[0].response_num[0]).abs() < 1e-9);
}

#[test]
fn without_flow_options_the_run_gives_the_same_trades_and_no_buckets() {
    let (with_flow_agents, _) = agents();
    let (plain_agents, _) = agents();
    let options = RunOptions {
        flow: Some(flow_options(END_TIME)),
        ..RunOptions::default()
    };
    let with_flow = Kernel::run_with(&config(), with_flow_agents, &options);
    let plain = Kernel::run_with(&config(), plain_agents, &RunOptions::default());
    assert!(plain.flow_buckets.is_empty());
    assert_eq!(with_flow.trades.len(), plain.trades.len());
    assert_eq!(with_flow.events_processed, plain.events_processed);
    for (a, b) in with_flow.trades.iter().zip(plain.trades.iter()) {
        assert_eq!((a.timestamp, a.price, a.qty), (b.timestamp, b.price, b.qty));
    }
    let notional: u128 = with_flow
        .trades
        .iter()
        .map(|t| u128::from(t.price.unsigned_abs()) * u128::from(t.qty))
        .sum();
    let bucket = &with_flow.flow_buckets[0];
    assert_eq!(notional, bucket.market_notional + bucket.self_notional);
}
