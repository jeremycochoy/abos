//! Integration tests of `Kernel::run_with` and `RunOptions` (issue #8).
//!
//! The default options reproduce `Kernel::run` exactly. `keep_trades: false`
//! drops the trade log and nothing else. `l1_bucket_ns` replaces the
//! per-event L1 log with one aggregate per k-nanosecond bucket; the tests
//! rebuild those aggregates from the full log of an identical seeded run and
//! demand equality.

use agents::{ZiAgent, ZiAgentConfig};
use sim_core::{
    Agent, Kernel, L1Bucket, L1Snapshot, LatencyConfig, LatencyModelType, RunOptions,
    SimulationConfig, Symbol,
};

const BUCKET_NS: u64 = 10_000_000; // 10 ms buckets over a 100 ms run

fn zi_config(symbol: Symbol) -> ZiAgentConfig {
    ZiAgentConfig {
        wake_up_interval_ns: 1_000_000,
        price_std: 0.000_3,
        order_size_scale: 1.0,
        order_size_std: 0.3,
        reference_price: 10_000,
        symbol,
    }
}

fn sim_config(seed: u64, symbols: Vec<Symbol>) -> SimulationConfig {
    SimulationConfig {
        seed,
        start_time: 0,
        end_time: 100_000_000,
        symbols,
        latency: LatencyConfig {
            default_base_ns: 1_000,
            jitter_mu: 0.0,
            jitter_sigma: 0.0001,
            model: LatencyModelType::Uniform,
        },
        tick_size: 100,
        lot_size: 1,
        no_market_hours: false,
    }
}

fn make_agents(symbols: &[Symbol], per_symbol: usize, seed: u64) -> Vec<Box<dyn Agent>> {
    let mut agents: Vec<Box<dyn Agent>> = Vec::new();
    for &symbol in symbols {
        let cfg = zi_config(symbol);
        for i in 0..per_symbol {
            let agent_seed = seed.wrapping_add(u64::from(symbol) * 1000 + i as u64);
            agents.push(Box::new(ZiAgent::new(cfg.clone(), agent_seed)));
        }
    }
    agents
}

/// Rebuild the expected buckets of one symbol from the full L1 log.
///
/// The rules mirror the kline export: the price fields read the quoted
/// snapshots only (both sides present), the volume and the notional sum over
/// every snapshot, and a bucket exists when at least one snapshot fell in it.
fn reference_buckets(snapshots: &[L1Snapshot], symbol: Symbol, bucket_ns: u64) -> Vec<L1Bucket> {
    let mut buckets: Vec<L1Bucket> = Vec::new();
    for snap in snapshots.iter().filter(|s| s.symbol == symbol) {
        let start = snap.timestamp - snap.timestamp % bucket_ns;
        if buckets.last().map(|b| b.bucket_start) != Some(start) {
            buckets.push(L1Bucket::new(start, symbol));
        }
        let bucket = buckets.last_mut().unwrap();
        bucket.absorb(snap.bid_price, snap.ask_price, snap.bid_volume, snap.ask_volume);
    }
    buckets
}

fn assert_same_snapshots(a: &[L1Snapshot], b: &[L1Snapshot]) {
    assert_eq!(a.len(), b.len());
    for (x, y) in a.iter().zip(b.iter()) {
        assert_eq!(x.timestamp, y.timestamp);
        assert_eq!(x.symbol, y.symbol);
        assert_eq!(x.bid_price, y.bid_price);
        assert_eq!(x.ask_price, y.ask_price);
        assert_eq!(x.bid_volume, y.bid_volume);
        assert_eq!(x.ask_volume, y.ask_volume);
        assert_eq!(x.last_trade_price, y.last_trade_price);
    }
}

#[test]
fn default_options_reproduce_the_default_run() {
    let cfg = sim_config(42, vec![0]);
    let plain = Kernel::run(&cfg, make_agents(&[0], 10, 42));
    let with_default = Kernel::run_with(&cfg, make_agents(&[0], 10, 42), &RunOptions::default());

    assert_eq!(plain.events_processed, with_default.events_processed);
    assert_eq!(plain.trades.len(), with_default.trades.len());
    assert!(!plain.trades.is_empty(), "the run must trade, or the test tests nothing");
    assert_same_snapshots(&plain.l1_snapshots, &with_default.l1_snapshots);
    assert!(with_default.l1_buckets.is_empty());
}

#[test]
fn keep_trades_off_drops_the_trades_and_nothing_else() {
    let cfg = sim_config(42, vec![0]);
    let reference = Kernel::run(&cfg, make_agents(&[0], 10, 42));
    let options = RunOptions { keep_trades: false, ..RunOptions::default() };
    let lean = Kernel::run_with(&cfg, make_agents(&[0], 10, 42), &options);

    assert!(lean.trades.is_empty());
    assert!(!reference.trades.is_empty());
    assert_eq!(reference.events_processed, lean.events_processed);
    assert_same_snapshots(&reference.l1_snapshots, &lean.l1_snapshots);
}

#[test]
fn bucket_mode_replaces_the_snapshot_log() {
    let cfg = sim_config(42, vec![0]);
    let options = RunOptions { l1_bucket_ns: Some(BUCKET_NS), ..RunOptions::default() };
    let lean = Kernel::run_with(&cfg, make_agents(&[0], 10, 42), &options);

    assert!(lean.l1_snapshots.is_empty());
    assert!(!lean.l1_buckets.is_empty());
    for bucket in &lean.l1_buckets {
        assert_eq!(bucket.bucket_start % BUCKET_NS, 0);
    }
}

#[test]
fn bucket_aggregates_equal_a_reference_aggregation_of_the_full_log() {
    let symbols: Vec<Symbol> = vec![0, 1];
    let cfg = sim_config(42, symbols.clone());
    let full = Kernel::run(&cfg, make_agents(&symbols, 6, 42));
    let options = RunOptions { l1_bucket_ns: Some(BUCKET_NS), ..RunOptions::default() };
    let lean = Kernel::run_with(&cfg, make_agents(&symbols, 6, 42), &options);

    assert_eq!(full.events_processed, lean.events_processed);
    for &symbol in &symbols {
        let expected = reference_buckets(&full.l1_snapshots, symbol, BUCKET_NS);
        let got: Vec<L1Bucket> =
            lean.l1_buckets.iter().copied().filter(|b| b.symbol == symbol).collect();
        assert!(!expected.is_empty());
        assert_eq!(expected, got);
    }
}

#[test]
fn both_options_together() {
    let cfg = sim_config(7, vec![0]);
    let full = Kernel::run(&cfg, make_agents(&[0], 10, 7));
    let options = RunOptions { keep_trades: false, l1_bucket_ns: Some(BUCKET_NS), ..RunOptions::default() };
    let lean = Kernel::run_with(&cfg, make_agents(&[0], 10, 7), &options);

    assert!(lean.trades.is_empty());
    assert!(lean.l1_snapshots.is_empty());
    assert_eq!(reference_buckets(&full.l1_snapshots, 0, BUCKET_NS), lean.l1_buckets);
}

#[test]
fn bucket_volumes_count_every_snapshot_and_prices_the_quoted_ones() {
    let mut bucket = L1Bucket::new(0, 0);
    bucket.absorb(0, 0, 0, 0); // empty book: no quote, no volume
    bucket.absorb(100, 0, 5, 0); // one-sided book: volume and notional, no quote
    bucket.absorb(101, 103, 4, 6); // first quote
    bucket.absorb(99, 105, 1, 2); // widens the extremes
    bucket.absorb(100, 104, 3, 3); // last quote

    assert_eq!(bucket.first_bid, 101);
    assert_eq!(bucket.first_ask, 103);
    assert_eq!(bucket.min_bid, 99);
    assert_eq!(bucket.max_ask, 105);
    assert_eq!(bucket.last_bid, 100);
    assert_eq!(bucket.last_ask, 104);
    assert_eq!(bucket.volume, 5 + 4 + 6 + 1 + 2 + 3 + 3);
    let notional = 100 * 5 + (101 * 4 + 103 * 6) + (99 + 105 * 2) + (100 * 3 + 104 * 3);
    assert_eq!(bucket.quote_volume, notional);
}

#[test]
fn a_bucket_with_no_quote_keeps_zero_prices() {
    let mut bucket = L1Bucket::new(0, 3);
    bucket.absorb(0, 0, 0, 0);
    bucket.absorb(50, 0, 2, 0);

    assert_eq!(bucket.first_bid, 0);
    assert_eq!(bucket.last_ask, 0);
    assert_eq!(bucket.volume, 2);
    assert_eq!(bucket.quote_volume, 100);
}
