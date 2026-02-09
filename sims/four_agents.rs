use std::path::Path;
use std::time::Instant;

use agents::{
    MarketMakerAgent, MarketMakerConfig, TrendFollowingAgent, TrendFollowingConfig, ZiAgent,
    ZiAgentConfig,
};
use sim_core::{Agent, Kernel, LatencyConfig, LatencyModelType, SimulationConfig};

fn main() {
    // ── Reference: simple_4agents_model from evolve_trading ──────────────
    let tick_size: i64 = 100_000;
    let lot_size: u64 = 100_000;
    let real_price: i64 = 100_000;
    let initial_price = real_price * tick_size;

    let one_second_ns: u64 = 1_000_000_000;
    let one_minute_ns: u64 = 60 * one_second_ns;
    let one_day_ns: u64 = 86_400 * one_second_ns;
    let one_year_ns: u64 = 365 * one_day_ns;

    let config = SimulationConfig {
        seed: 42,
        start_time: 0,
        end_time: one_year_ns,
        symbols: vec![0],
        latency: LatencyConfig {
            default_base_ns: 50_000,
            jitter_mu: 0.0,
            jitter_sigma: 0.3,
            model: LatencyModelType::NycSeattle { seed: 42 },
        },
        tick_size,
        lot_size,
        no_market_hours: true,
    };

    let mut agents: Vec<Box<dyn Agent>> = Vec::new();
    let mut next_seed: u64 = config.seed;
    let mut alloc_seed = || { let s = next_seed; next_seed += 1; s };

    // 1) 9 ZI agents
    let zi_cfg = ZiAgentConfig {
        wake_up_interval_ns: 30 * one_second_ns,
        price_std: 0.025 / 100.0,        // 0.025%
        #[allow(clippy::cast_precision_loss)]
        order_size_scale: 0.08 * lot_size as f64,   // 8,000
        order_size_std: 1.0,
        reference_price: initial_price,
        symbol: 0,
    };
    for _ in 0..9 {
        agents.push(Box::new(ZiAgent::new(zi_cfg.clone(), alloc_seed())));
    }

    // 2) 3 Trend-following agents
    let tf_cfg = TrendFollowingConfig {
        short_window: 12,
        long_window: 40,
        threshold: 0.05 / 100.0,          // 0.05%
        price_offset: 2.0 / 100.0,        // 2%
        #[allow(clippy::cast_precision_loss)]
        order_size_factor: 0.07 * lot_size as f64,   // 7,000
        #[allow(clippy::cast_precision_loss)]
        order_size_boost: 0.007 * lot_size as f64,   // 700
        mean_wakeup_interval_ns: 5 * one_minute_ns,
        sampling_freq_ns: 60 * one_second_ns,
        reference_price: initial_price,
        symbol: 0,
        contrarian: false,
    };
    for _ in 0..3 {
        agents.push(Box::new(TrendFollowingAgent::new(tf_cfg.clone(), alloc_seed())));
    }

    // 3) 1 Trend-contrarian agent
    let tc_cfg = TrendFollowingConfig {
        short_window: 3,
        long_window: 60,
        threshold: 0.08 / 100.0,          // 0.08%
        price_offset: 2.5 / 100.0,        // 2.5%
        #[allow(clippy::cast_precision_loss)]
        order_size_factor: 0.07 * lot_size as f64,
        #[allow(clippy::cast_precision_loss)]
        order_size_boost: 0.007 * lot_size as f64,
        mean_wakeup_interval_ns: 5 * one_minute_ns,
        sampling_freq_ns: 60 * one_second_ns,
        reference_price: initial_price,
        symbol: 0,
        contrarian: true,
    };
    agents.push(Box::new(TrendFollowingAgent::new(tc_cfg, alloc_seed())));

    // 4) 1 Liquidity market-maker agent
    let mm_cfg = MarketMakerConfig {
        #[allow(clippy::cast_precision_loss)]
        total_liquidity: 0.005 * lot_size as f64,    // 500 units
        step_size_ratio: 0.01 / 100.0,    // 0.01%
        max_levels: 3,
        imbalance_beta: 1.0,
        peak_distance_ratio: 5_000.0 / 100_000.0,   // 5%
        shape_exponent: 1.2,
        mean_wakeup_interval_ns: 30 * one_second_ns,
        reference_price: initial_price,
        symbol: 0,
    };
    agents.push(Box::new(MarketMakerAgent::new(mm_cfg, alloc_seed())));

    println!("Starting 4-agent-model simulation (1 year, {} agents)...", agents.len());
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

    println!("=== 4-Agent-Model Simulation (1 Year) ===");
    println!("Trades:          {}", result.trades.len());
    println!("Total volume:    {total_volume}");
    #[allow(clippy::cast_precision_loss)]
    {
        println!("VWAP:            ${vwap:.2}");
        if !result.trades.is_empty() {
            let (high, low) = result.trades.iter().fold((i64::MIN, i64::MAX), |(h, l), t| {
                (h.max(t.price), l.min(t.price))
            });
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
