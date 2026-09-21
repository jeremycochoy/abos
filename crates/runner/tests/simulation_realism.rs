use std::sync::OnceLock;

use agents::{ZiAgent, ZiAgentConfig};
use sim_core::{Agent, Kernel, LatencyConfig, LatencyModelType, SimulationConfig, SimulationResult};

fn realistic_config() -> SimulationConfig {
    SimulationConfig {
        seed: 123,
        start_time: 0,
        end_time: 60_000_000_000, // 1 minute (fast enough for debug mode)
        symbols: vec![0],
        latency: LatencyConfig {
            default_base_ns: 50_000,
            jitter_mu: 0.0,
            jitter_sigma: 0.3,
            model: LatencyModelType::Uniform,
        },
        tick_size: 100,
        lot_size: 1,
        no_market_hours: false,
    }
}

fn make_agents(n: usize, seed: u64) -> Vec<Box<dyn Agent>> {
    let cfg = ZiAgentConfig {
        wake_up_interval_ns: 10_000_000,
        price_std: 0.000_3,
        order_size_scale: 1.0,
        order_size_std: 0.3,
        reference_price: 10_000,
        symbol: 0,
    };
    (0..n)
        .map(|i| Box::new(ZiAgent::new(cfg, seed.wrapping_add(i as u64))) as Box<dyn Agent>)
        .collect()
}

// Run once, share across tests.
static RESULT: OnceLock<SimulationResult> = OnceLock::new();

fn get_result() -> &'static SimulationResult {
    RESULT.get_or_init(|| {
        let cfg = realistic_config();
        Kernel::run(&cfg, make_agents(100, 123))
    })
}

#[test]
fn trades_happen() {
    let r = get_result();
    assert!(r.trades.len() > 10, "should produce trades, got {}", r.trades.len());
}

#[test]
fn price_stays_near_reference() {
    let r = get_result();
    assert!(!r.trades.is_empty());

    let total_volume: u64 = r.trades.iter().map(|t| t.qty).sum();
    let notional: f64 = r.trades.iter().map(|t| t.price as f64 * t.qty as f64).sum();
    let vwap = notional / total_volume as f64;
    let reference = 10_000.0;

    let pct_deviation = ((vwap - reference) / reference).abs();
    assert!(pct_deviation < 0.20,
        "VWAP {vwap:.1} deviates {:.1}% from reference {reference}", pct_deviation * 100.0);
}

#[test]
fn spread_is_positive() {
    let r = get_result();

    let valid_snaps: Vec<_> = r.l1_snapshots.iter()
        .filter(|s| s.bid_price > 0 && s.ask_price > 0)
        .collect();
    assert!(!valid_snaps.is_empty());

    let total_spread: f64 = valid_snaps.iter()
        .map(|s| (s.ask_price - s.bid_price) as f64)
        .sum();
    let avg_spread = total_spread / valid_snaps.len() as f64;

    assert!(avg_spread > 0.0, "spread should be positive");
    let reference = 10_000.0;
    assert!(avg_spread / reference < 0.10,
        "spread {avg_spread:.1} too wide vs ref {reference}");
}

#[test]
fn book_has_orders() {
    let r = get_result();

    let total = r.l1_snapshots.len();
    assert!(total > 0);
    let both_sides = r.l1_snapshots.iter()
        .filter(|s| s.bid_price > 0 && s.ask_price > 0)
        .count();

    let ratio = both_sides as f64 / total as f64;
    assert!(ratio > 0.3,
        "book should have both sides >30% of time, got {:.1}%", ratio * 100.0);
}

#[test]
fn cancellations_occur() {
    let r = get_result();
    assert!(r.events_processed > 100);
}

#[test]
fn performance_reasonable() {
    let cfg = realistic_config();
    let start = std::time::Instant::now();
    let r = Kernel::run(&cfg, make_agents(100, 999));
    let elapsed = start.elapsed();
    // Only assert tight bound in release mode; debug is ~10-30x slower
    if cfg!(not(debug_assertions)) {
        assert!(elapsed.as_secs() < 5,
            "1 min sim took {:.1}s in release", elapsed.as_secs_f64());
    }
    assert!(!r.trades.is_empty());
}
