use std::path::Path;
use std::time::Instant;

use agents::{ZiAgent, ZiAgentConfig};
use sim_core::{Agent, Kernel, LatencyConfig, LatencyModelType, SimulationConfig};

fn main() {
    // ── Reference experiment (ABIDES zero_intelligence_only) ──────────────
    // 20 ZI agents, 30s wakeup interval, lognormal price (0.03% std),
    // lognormal order size (scale=12000, std=1.7).
    // Run for a full year with no market hours (continuous trading).

    let tick_size: i64 = 100_000;          // 1 tick = $0.00001
    let lot_size: u64 = 100_000;           // 1 lot = 100,000 internal units
    let real_price: i64 = 100_000;         // $100,000
    let initial_price = real_price * tick_size; // internal price units

    let one_second_ns: u64 = 1_000_000_000;
    let one_day_ns: u64 = 86_400 * one_second_ns;
    let one_year_ns: u64 = 365 * one_day_ns;

    let config = SimulationConfig {
        seed: 42,
        start_time: 0,
        end_time: one_year_ns,
        symbols: vec![0],
        latency: LatencyConfig {
            default_base_ns: 50_000,       // 50µs
            jitter_mu: 0.0,
            jitter_sigma: 0.3,
            model: LatencyModelType::NycSeattle { seed: 42 },
        },
        tick_size,
        lot_size,
        no_market_hours: true,             // Continuous trading, no open/close
    };

    let zi_cfg = ZiAgentConfig {
        wake_up_interval_ns: 30 * one_second_ns,  // 30 seconds
        price_std: 0.03 / 100.0,                  // 0.03%
        order_size_scale: 12_000.0,
        order_size_std: 1.7,
        reference_price: initial_price,
        symbol: 0,
    };

    let agents: Vec<Box<dyn Agent>> = (0..20)
        .map(|i| Box::new(ZiAgent::new(zi_cfg.clone(), config.seed.wrapping_add(i))) as _)
        .collect();

    let start = Instant::now();
    let result = Kernel::run(&config, agents);
    let elapsed = start.elapsed();

    // Write output
    let _ = sim_core::output::write_trades(
        Path::new("trades.parquet"), &result.trades, config.tick_size, config.lot_size,
    );
    let _ = sim_core::output::write_l1_snapshots(
        Path::new("l1_snapshots.parquet"), &result.l1_snapshots, config.tick_size, config.lot_size,
    );

    // Summary
    let total_volume: u64 = result.trades.iter().map(|t| t.qty).sum();
    #[allow(clippy::cast_precision_loss)]
    let vwap = if total_volume > 0 {
        let notional: f64 = result.trades.iter().map(|t| t.price as f64 * t.qty as f64).sum();
        notional / total_volume as f64 / config.tick_size as f64
    } else {
        0.0
    };

    let (high, low) = result.trades.iter().fold((i64::MIN, i64::MAX), |(h, l), t| {
        (h.max(t.price), l.min(t.price))
    });

    println!("=== ZI Single-Symbol Simulation (1 Year) ===");
    println!("Trades:          {}", result.trades.len());
    println!("Total volume:    {total_volume}");
    #[allow(clippy::cast_precision_loss)]
    {
        println!("VWAP:            ${vwap:.2}");
        if !result.trades.is_empty() {
            println!("Price range:     ${:.2} - ${:.2}",
                low as f64 / config.tick_size as f64,
                high as f64 / config.tick_size as f64);
        }
    }
    println!("L1 snapshots:    {}", result.l1_snapshots.len());
    println!("Events processed:{}", result.events_processed);
    println!("Wall-clock:      {:.3}s", elapsed.as_secs_f64());
    #[allow(clippy::cast_precision_loss)]
    if elapsed.as_secs_f64() > 0.0 {
        println!("Events/sec:      {:.0}", result.events_processed as f64 / elapsed.as_secs_f64());
    }
}
