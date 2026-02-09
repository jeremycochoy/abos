use std::path::Path;
use std::time::Instant;

use agents::{ZiAgent, ZiAgentConfig};
use sim_core::{Agent, Kernel, LatencyConfig, SimulationConfig};

fn main() {
    let config = SimulationConfig {
        seed: 42,
        start_time: 34_200_000_000_000,   // 09:30:00 as ns since midnight
        end_time: 34_800_000_000_000,      // 09:40:00 (10 minutes)
        symbols: vec![0],
        latency: LatencyConfig {
            default_base_ns: 50_000,       // 50µs
            jitter_mu: 0.0,
            jitter_sigma: 0.3,
        },
        tick_size: 100,
        lot_size: 1,
    };

    let zi_cfg = ZiAgentConfig {
        p_limit: 0.70,
        p_cancel: 0.10,
        mean_wakeup_interval_ns: 10_000_000, // 10ms
        price_offset_lambda: 0.2,
        default_qty: 1,
        reference_price: 10_000,
        tick_size: 1,
        symbol: 0,
    };

    let agents: Vec<Box<dyn Agent>> = (0..100)
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
    let vwap = if total_volume > 0 {
        let notional: f64 = result.trades.iter().map(|t| t.price as f64 * t.qty as f64).sum();
        notional / total_volume as f64 / config.tick_size as f64
    } else {
        0.0
    };

    let (high, low) = result.trades.iter().fold((i64::MIN, i64::MAX), |(h, l), t| {
        (h.max(t.price), l.min(t.price))
    });

    println!("=== Simulation Summary ===");
    println!("Trades:          {}", result.trades.len());
    println!("Total volume:    {total_volume}");
    println!("VWAP:            ${vwap:.2}");
    if !result.trades.is_empty() {
        println!("Price range:     ${:.2} - ${:.2}",
            low as f64 / config.tick_size as f64,
            high as f64 / config.tick_size as f64);
    }
    println!("L1 snapshots:    {}", result.l1_snapshots.len());
    println!("Events processed:{}", result.events_processed);
    println!("Wall-clock:      {:.3}s", elapsed.as_secs_f64());
    if elapsed.as_secs_f64() > 0.0 {
        println!("Events/sec:      {:.0}", result.events_processed as f64 / elapsed.as_secs_f64());
    }
}
