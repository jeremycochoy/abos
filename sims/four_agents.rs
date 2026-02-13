use std::path::Path;
use std::time::Instant;

use agents::{
    MarketMakerAgent, MarketMakerConfig, TrendFollowingAgent, TrendFollowingConfig, ZiAgent,
    ZiAgentConfig,
};
use sim_core::{Agent, Kernel, LatencyConfig, LatencyModelType, SimulationConfig};

// Trading precision constants
const LOT_SIZE: u64 = 100_000;
const TICK_SIZE: i64 = 100_000;
const REAL_PRICE: i64 = 100_000;

// Time constants
const ONE_SECOND_NS: u64 = 1_000_000_000;
const ONE_MINUTE_NS: u64 = 60 * ONE_SECOND_NS;
const ONE_DAY_NS: u64 = 86_400 * ONE_SECOND_NS;
const ONE_YEAR_NS: u64 = 365 * ONE_DAY_NS;

// EVOLVE-BLOCK-START

// Zero Intelligence agent parameters
const ZI_NB_AGENTS: usize = 9;
const ZI_PRICE_STD: f64 = 0.025 / 100.0; // 0.025% standard deviation relative to mid price
const ZI_ORDER_SIZE_LOGNORMAL_STD: f64 = 1.0;
const ZI_ORDER_SIZE_SCALE: f64 = 0.08 * LOT_SIZE as f64; // 8,000
const ZI_WAKE_UP_INTERVAL_NS: u64 = 30 * ONE_SECOND_NS;

// Trend Following agent parameters
const TF_NB_AGENTS: usize = 3;
const TF_SHORT_WINDOW: usize = 12;
const TF_LONG_WINDOW: usize = 40;
const TF_THRESHOLD: f64 = 0.05 / 100.0; // 0.05%
const TF_PRICE_OFFSET: f64 = 2.0 / 100.0; // 2%
const TF_ORDER_SIZE_FACTOR: f64 = 0.07 * LOT_SIZE as f64; // 7,000
const TF_ORDER_SIZE_BOOST: f64 = 0.007 * LOT_SIZE as f64; // 700
const TF_TRADE_ARRIVAL_INTERVAL_NS: u64 = 5 * ONE_MINUTE_NS;
const TF_SAMPLING_FREQ_NS: u64 = 60 * ONE_SECOND_NS;

// Mean reverting (trend contrarian) agent parameters
const TC_NB_AGENTS: usize = 1;
const TC_SHORT_WINDOW: usize = 3;
const TC_LONG_WINDOW: usize = 60;
const TC_THRESHOLD: f64 = 0.08 / 100.0; // 0.08%
const TC_PRICE_OFFSET: f64 = 2.5 / 100.0; // 2.5%
const TC_ORDER_SIZE_FACTOR: f64 = 0.07 * LOT_SIZE as f64;
const TC_ORDER_SIZE_BOOST: f64 = 0.007 * LOT_SIZE as f64;
const TC_TRADE_ARRIVAL_INTERVAL_NS: u64 = 5 * ONE_MINUTE_NS;
const TC_SAMPLING_FREQ_NS: u64 = 60 * ONE_SECOND_NS;

// Liquidity Market Maker parameters
const LMM_NB_AGENTS: usize = 1;
const LMM_TOTAL_LIQUIDITY: f64 = 0.005 * LOT_SIZE as f64; // 500 units
const LMM_STEP_SIZE_RATIO: f64 = 0.01 / 100.0; // 0.01%
const LMM_MAX_LEVELS: usize = 3;
const LMM_IMBALANCE_BETA: f64 = 1.0;
const LMM_PEAK_DISTANCE_RATIO: f64 = 5_000.0 / 100_000.0; // 5%
const LMM_SHAPE_EXPONENT: f64 = 1.2;
const LMM_WAKE_UP_INTERVAL_NS: u64 = 30 * ONE_SECOND_NS;

// EVOLVE-BLOCK-END

fn main() {
    let initial_price = REAL_PRICE * TICK_SIZE;

    let config = SimulationConfig {
        seed: 42,
        start_time: 0,
        end_time: ONE_YEAR_NS,
        symbols: vec![0],
        latency: LatencyConfig {
            default_base_ns: 50_000,
            jitter_mu: 0.0,
            jitter_sigma: 0.3,
            model: LatencyModelType::NycSeattle { seed: 42 },
        },
        tick_size: TICK_SIZE,
        lot_size: LOT_SIZE,
        no_market_hours: true,
    };

    let mut agents: Vec<Box<dyn Agent>> = Vec::new();
    let mut next_seed: u64 = config.seed;
    let mut alloc_seed = || { let s = next_seed; next_seed += 1; s };

    // 1) ZI agents
    let zi_cfg = ZiAgentConfig {
        wake_up_interval_ns: ZI_WAKE_UP_INTERVAL_NS,
        price_std: ZI_PRICE_STD,
        order_size_scale: ZI_ORDER_SIZE_SCALE,
        order_size_std: ZI_ORDER_SIZE_LOGNORMAL_STD,
        reference_price: initial_price,
        symbol: 0,
    };
    for _ in 0..ZI_NB_AGENTS {
        agents.push(Box::new(ZiAgent::new(zi_cfg.clone(), alloc_seed())));
    }

    // 2) Trend-following agents
    let tf_cfg = TrendFollowingConfig {
        short_window: TF_SHORT_WINDOW,
        long_window: TF_LONG_WINDOW,
        threshold: TF_THRESHOLD,
        price_offset: TF_PRICE_OFFSET,
        order_size_factor: TF_ORDER_SIZE_FACTOR,
        order_size_boost: TF_ORDER_SIZE_BOOST,
        mean_wakeup_interval_ns: TF_TRADE_ARRIVAL_INTERVAL_NS,
        sampling_freq_ns: TF_SAMPLING_FREQ_NS,
        reference_price: initial_price,
        symbol: 0,
        contrarian: false,
    };
    for _ in 0..TF_NB_AGENTS {
        agents.push(Box::new(TrendFollowingAgent::new(tf_cfg.clone(), alloc_seed())));
    }

    // 3) Trend-contrarian agent(s)
    let tc_cfg = TrendFollowingConfig {
        short_window: TC_SHORT_WINDOW,
        long_window: TC_LONG_WINDOW,
        threshold: TC_THRESHOLD,
        price_offset: TC_PRICE_OFFSET,
        order_size_factor: TC_ORDER_SIZE_FACTOR,
        order_size_boost: TC_ORDER_SIZE_BOOST,
        mean_wakeup_interval_ns: TC_TRADE_ARRIVAL_INTERVAL_NS,
        sampling_freq_ns: TC_SAMPLING_FREQ_NS,
        reference_price: initial_price,
        symbol: 0,
        contrarian: true,
    };
    for _ in 0..TC_NB_AGENTS {
        agents.push(Box::new(TrendFollowingAgent::new(tc_cfg.clone(), alloc_seed())));
    }

    // 4) Liquidity market-maker agent(s)
    let mm_cfg = MarketMakerConfig {
        total_liquidity: LMM_TOTAL_LIQUIDITY,
        step_size_ratio: LMM_STEP_SIZE_RATIO,
        max_levels: LMM_MAX_LEVELS,
        imbalance_beta: LMM_IMBALANCE_BETA,
        peak_distance_ratio: LMM_PEAK_DISTANCE_RATIO,
        shape_exponent: LMM_SHAPE_EXPONENT,
        mean_wakeup_interval_ns: LMM_WAKE_UP_INTERVAL_NS,
        reference_price: initial_price,
        symbol: 0,
    };
    for _ in 0..LMM_NB_AGENTS {
        agents.push(Box::new(MarketMakerAgent::new(mm_cfg.clone(), alloc_seed())));
    }

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
