use std::cell::RefCell;
use std::rc::Rc;

use cda_engine::Side;
use sim_core::{
    Agent, AgentAction, AgentId, ExchangeMessage, FlowBucket, FlowOptions, FlowTotals, Kernel,
    LatencyConfig, LatencyModelType, MarketSnapshot, Nanos, OrderAction, RunOptions,
    SimulationConfig,
};

type Seen = Rc<RefCell<Vec<ExchangeMessage>>>;

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
        (Self { script, cursor: 0, seen: Rc::clone(&seen) }, seen)
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
            actions.push(AgentAction::SubmitOrder { symbol: 0, order: self.script[self.cursor].1 });
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
    OrderAction::NewLimitOrder { side, price, qty, user_id: 0 }
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

fn agents(with_self_crosser: bool) -> (Vec<Box<dyn Agent>>, Seen) {
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
    let mut all: Vec<Box<dyn Agent>> =
        vec![Box::new(maker_a), Box::new(maker_b), Box::new(watched), Box::new(outsider)];
    if with_self_crosser {
        let (crosser, _) = ScriptAgent::new(vec![
            (4_000, limit(Side::Ask, 10_400, 7)),
            (4_500, limit(Side::Bid, 10_400, 7)),
        ]);
        all.push(Box::new(crosser));
    }
    (all, watched_seen)
}

fn flow_options(interval_ns: Nanos) -> FlowOptions {
    FlowOptions {
        agent_id: WATCHED,
        interval_ns,
        horizons_ns: vec![500, 10_000],
        ratio_edges: vec![0.25, 0.5, 0.75, 1.0],
    }
}

fn run(interval_ns: Nanos, with_self_crosser: bool) -> (Vec<FlowBucket>, Vec<FlowTotals>, Seen) {
    let (agents, seen) = agents(with_self_crosser);
    let options = RunOptions { flow: Some(flow_options(interval_ns)), ..RunOptions::default() };
    let result = Kernel::run_with(&config(), agents, &options);
    (result.flow_buckets, result.flow_totals, seen)
}

#[test]
fn one_bucket_counts_external_watched_and_self_flows_exactly() {
    let (buckets, totals, _) = run(END_TIME, false);
    assert_eq!(buckets.len(), 1);
    let bucket = &buckets[0];
    assert_eq!((bucket.start, bucket.symbol), (0, 0));
    assert_eq!(bucket.market_trades, 5);
    assert_eq!(bucket.market_qty, 38);
    assert_eq!(
        bucket.market_notional,
        10 * 10_100 + 5 * 10_100 + 5 * 10_200 + 3 * 10_150 + 15 * 10_200
    );
    assert_eq!(
        (bucket.market_self_trades, bucket.market_self_qty, bucket.market_self_notional),
        (1, 5, 5 * 10_150)
    );
    assert_eq!(
        (bucket.self_trades, bucket.self_qty, bucket.self_notional),
        (1, 5, 5 * 10_150)
    );
    assert_eq!((bucket.taker_qty, bucket.taker_notional), (35, 355_500));
    assert_eq!((bucket.maker_qty, bucket.maker_notional), (3, 3 * 10_150));
    assert_eq!(
        (bucket.orders, bucket.rejected_orders, bucket.eligible_orders),
        (3, 0, 3)
    );
    assert_eq!((bucket.filled_orders, bucket.fill_events), (3, 5));
    assert_eq!(bucket.submitted_qty, 48);
    assert_eq!(totals.len(), 1);
    assert_eq!(
        totals[0],
        FlowTotals {
            symbol: 0,
            accepted_orders: 3,
            rejected_orders: 0,
            filled_orders: 3,
            fill_events: 5,
        }
    );
}

#[test]
fn ratios_use_submitted_and_opposing_displayed_notional() {
    let (buckets, _, _) = run(END_TIME, false);
    let bucket = &buckets[0];
    assert_eq!(bucket.ratio_count, vec![0, 0, 1, 0, 1]);
    assert_eq!(bucket.ratio_notional, vec![0.0, 0.0, 204_000.0, 0.0, 206_000.0]);
    assert_eq!(
        (bucket.zero_depth_count, bucket.zero_depth_qty, bucket.zero_depth_notional),
        (1, 8, 81_200.0)
    );
}

#[test]
fn shortfall_splits_taker_and_maker_components_against_the_pre_trade_mid() {
    let (buckets, _, _) = run(END_TIME, false);
    let bucket = &buckets[0];
    let taker = (10_100.0 / 10_000.0 - 1.0) * 101_000.0
        + (10_100.0 / 10_000.0 - 1.0) * 50_500.0
        + (10_200.0 / 10_000.0 - 1.0) * 51_000.0
        + (10_200.0 / 10_025.0 - 1.0) * 153_000.0;
    let maker = -(10_150.0 / 10_025.0 - 1.0) * 30_450.0;
    assert!((bucket.shortfall_taker - taker).abs() < 1e-9, "{}", bucket.shortfall_taker);
    assert_eq!(bucket.shortfall_taker_base, 355_500);
    assert!((bucket.shortfall_maker - maker).abs() < 1e-9, "{}", bucket.shortfall_maker);
    assert_eq!(bucket.shortfall_maker_base, 30_450);
    assert_eq!(
        (bucket.shortfall_excluded_count, bucket.shortfall_excluded_notional),
        (0, 0)
    );
}

#[test]
fn responses_resolve_with_the_prevailing_mid_and_exclude_missing_mids() {
    let (buckets, _, _) = run(END_TIME, false);
    let bucket = &buckets[0];
    let resolved_buys = (101_000.0 + 50_500.0 + 51_000.0) * (10_025.0 / 10_000.0 - 1.0);
    let resolved_sale = -30_450.0 * (10_025.0 / 10_025.0 - 1.0);
    assert!(
        (bucket.response_num[0] - (resolved_buys + resolved_sale)).abs() < 1e-9,
        "{}",
        bucket.response_num[0]
    );
    assert!((bucket.response_den[0] - 232_950.0).abs() < 1e-9);
    assert_eq!(bucket.response_count[0], 4);
    assert!((bucket.response_excluded_notional[0] - 153_000.0).abs() < 1e-9);
    assert_eq!(bucket.response_excluded_count[0], 1);

    assert_eq!(bucket.response_num[1], 0.0);
    assert_eq!(bucket.response_den[1], 0.0);
    assert_eq!(bucket.response_count[1], 0);
    assert!((bucket.response_excluded_notional[1] - 385_950.0).abs() < 1e-9);
    assert_eq!(bucket.response_excluded_count[1], 5);
}

#[test]
fn a_self_trade_of_any_agent_leaves_the_market_denominator() {
    let (buckets, _, _) = run(END_TIME, true);
    let bucket = &buckets[0];
    assert_eq!(bucket.market_trades, 5);
    assert_eq!(bucket.market_notional, 385_950);
    assert_eq!(
        (bucket.market_self_trades, bucket.market_self_qty, bucket.market_self_notional),
        (2, 12, 5 * 10_150 + 7 * 10_400)
    );
    assert_eq!(
        (bucket.self_trades, bucket.self_qty, bucket.self_notional),
        (1, 5, 5 * 10_150)
    );
}

#[test]
fn the_taker_fill_message_carries_the_exact_notional_across_levels() {
    let (_, _, seen) = run(END_TIME, false);
    let fills: Vec<(u64, i64, u128)> = seen
        .borrow()
        .iter()
        .filter_map(|message| match *message {
            ExchangeMessage::OrderFilled { side: Side::Bid, price, qty, notional, .. } => {
                Some((qty, price, notional))
            }
            _ => None,
        })
        .collect();
    assert_eq!(fills[0], (20, 10_200, 202_500));
    assert_ne!(202_500, 20 * 10_200);
    assert_eq!(fills[1], (20, 10_200, 203_750));
}

#[test]
fn the_maker_fill_message_carries_its_own_notional() {
    let (_, _, seen) = run(END_TIME, false);
    let sales: Vec<(u64, i64, u128)> = seen
        .borrow()
        .iter()
        .filter_map(|message| match *message {
            ExchangeMessage::OrderFilled { side: Side::Ask, price, qty, notional, .. } => {
                Some((qty, price, notional))
            }
            _ => None,
        })
        .collect();
    assert_eq!(sales, vec![(3, 10_150, 3 * 10_150), (5, 10_150, 5 * 10_150)]);
}

#[test]
fn short_intervals_carry_live_orders_into_the_next_cohort() {
    let (buckets, _, _) = run(2_000, false);
    assert_eq!(buckets.len(), 500);
    assert!(buckets.iter().enumerate().all(|(i, b)| b.start == 2_000 * i as u64));
    assert_eq!(
        (buckets[0].orders, buckets[0].eligible_orders, buckets[0].filled_orders),
        (2, 2, 1)
    );
    assert_eq!(buckets[0].fill_events, 3);
    assert_eq!(buckets[0].market_trades, 3);
    assert_eq!(
        (buckets[1].orders, buckets[1].eligible_orders, buckets[1].filled_orders),
        (1, 2, 2)
    );
    assert_eq!(buckets[1].fill_events, 2);
    assert_eq!(buckets[1].market_trades, 2);
    assert_eq!(buckets[1].self_trades, 1);
    assert!(buckets[2..]
        .iter()
        .all(|b| b.orders == 0 && b.eligible_orders == 0 && b.market_trades == 0));
    let single = run(END_TIME, false).0;
    let split_num: f64 = buckets.iter().map(|b| b.response_num[0]).sum();
    assert!((split_num - single[0].response_num[0]).abs() < 1e-9);
    let split_excluded: f64 = buckets.iter().map(|b| b.response_excluded_notional[1]).sum();
    assert!((split_excluded - single[0].response_excluded_notional[1]).abs() < 1e-9);
}

#[test]
fn without_flow_options_the_run_gives_the_same_trades_and_no_buckets() {
    let (with_flow_agents, _) = agents(false);
    let (plain_agents, _) = agents(false);
    let options = RunOptions { flow: Some(flow_options(END_TIME)), ..RunOptions::default() };
    let with_flow = Kernel::run_with(&config(), with_flow_agents, &options);
    let plain = Kernel::run_with(&config(), plain_agents, &RunOptions::default());
    assert!(plain.flow_buckets.is_empty());
    assert!(plain.flow_totals.is_empty());
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
    assert_eq!(notional, bucket.market_notional + bucket.market_self_notional);
}
