use std::collections::HashMap;
use std::path::Path;
use std::thread;
use std::time::Instant;

use agents::{
    MarketMakerAgent, MarketMakerConfig, TrendFollowingAgent, TrendFollowingConfig, ZiAgent,
    ZiAgentConfig,
};
use sim_core::output::{write_klines, Candle};
use sim_core::{Agent, Kernel, LatencyConfig, LatencyModelType, SimulationConfig, SimulationResult};

const NUM_YEARS: usize = 100;
const MAX_PARALLEL: usize = 8;
const ONE_SECOND_NS: u64 = 1_000_000_000;
const ONE_MINUTE_NS: u64 = 60 * ONE_SECOND_NS;
const ONE_DAY_NS: u64 = 86_400 * ONE_SECOND_NS;
const ONE_YEAR_NS: u64 = 365 * ONE_DAY_NS;
const TICK_SIZE: i64 = 100_000;
const LOT_SIZE: u64 = 100_000;
const START_YEAR: i32 = 1919;
const CANDLES_PER_YEAR: usize = (ONE_YEAR_NS / ONE_MINUTE_NS) as usize; // 525_600

/// Per-minute OHLCV before timestamp assignment and normalization.
struct MinuteCandle {
    open: f64,
    high: f64,
    low: f64,
    close: f64,
    volume: f64,
    quote_volume: f64,
}

fn run_one_year(year_idx: usize) -> SimulationResult {
    let real_price: i64 = 100_000;
    let initial_price = real_price * TICK_SIZE;

    #[allow(clippy::cast_possible_truncation)]
    let base_seed = (year_idx as u64 + 1) * 1000;

    let config = SimulationConfig {
        seed: base_seed,
        start_time: 0,
        end_time: ONE_YEAR_NS,
        symbols: vec![0],
        latency: LatencyConfig {
            default_base_ns: 50_000,
            jitter_mu: 0.0,
            jitter_sigma: 0.3,
            model: LatencyModelType::NycSeattle { seed: base_seed },
        },
        tick_size: TICK_SIZE,
        lot_size: LOT_SIZE,
        no_market_hours: true,
    };

    let mut agents: Vec<Box<dyn Agent>> = Vec::new();
    let mut next_seed: u64 = base_seed + 100;
    let mut alloc_seed = || {
        let s = next_seed;
        next_seed += 1;
        s
    };

    // 9 ZI agents
    #[allow(clippy::cast_precision_loss)]
    let zi_cfg = ZiAgentConfig {
        wake_up_interval_ns: 30 * ONE_SECOND_NS,
        price_std: 0.025 / 100.0,
        order_size_scale: 0.08 * LOT_SIZE as f64,
        order_size_std: 1.0,
        reference_price: initial_price,
        symbol: 0,
    };
    for _ in 0..9 {
        agents.push(Box::new(ZiAgent::new(zi_cfg.clone(), alloc_seed())));
    }

    // 3 Trend-following agents
    #[allow(clippy::cast_precision_loss)]
    let tf_cfg = TrendFollowingConfig {
        short_window: 12,
        long_window: 40,
        threshold: 0.05 / 100.0,
        price_offset: 2.0 / 100.0,
        order_size_factor: 0.07 * LOT_SIZE as f64,
        order_size_boost: 0.007 * LOT_SIZE as f64,
        mean_wakeup_interval_ns: 5 * ONE_MINUTE_NS,
        sampling_freq_ns: 60 * ONE_SECOND_NS,
        reference_price: initial_price,
        symbol: 0,
        contrarian: false,
    };
    for _ in 0..3 {
        agents.push(Box::new(TrendFollowingAgent::new(
            tf_cfg.clone(),
            alloc_seed(),
        )));
    }

    // 1 Trend-contrarian agent
    #[allow(clippy::cast_precision_loss)]
    let tc_cfg = TrendFollowingConfig {
        short_window: 3,
        long_window: 60,
        threshold: 0.08 / 100.0,
        price_offset: 2.5 / 100.0,
        order_size_factor: 0.07 * LOT_SIZE as f64,
        order_size_boost: 0.007 * LOT_SIZE as f64,
        mean_wakeup_interval_ns: 5 * ONE_MINUTE_NS,
        sampling_freq_ns: 60 * ONE_SECOND_NS,
        reference_price: initial_price,
        symbol: 0,
        contrarian: true,
    };
    agents.push(Box::new(TrendFollowingAgent::new(tc_cfg, alloc_seed())));

    // 1 Market-maker agent
    #[allow(clippy::cast_precision_loss)]
    let mm_cfg = MarketMakerConfig {
        total_liquidity: 0.005 * LOT_SIZE as f64,
        step_size_ratio: 0.01 / 100.0,
        max_levels: 3,
        imbalance_beta: 1.0,
        peak_distance_ratio: 5_000.0 / 100_000.0,
        shape_exponent: 1.2,
        mean_wakeup_interval_ns: 30 * ONE_SECOND_NS,
        reference_price: initial_price,
        symbol: 0,
    };
    agents.push(Box::new(MarketMakerAgent::new(mm_cfg, alloc_seed())));

    Kernel::run(&config, agents)
}

/// Compute 1-minute OHLCV candles from a year's simulation result.
///
/// Uses mid price = (bid + ask) / 2 / tick_size for OHLC.
/// Forward-fills minutes with no BBO change.
#[allow(clippy::cast_precision_loss)]
fn compute_year_candles(result: &SimulationResult) -> Vec<MinuteCandle> {
    let tick = TICK_SIZE as f64;

    // Build OHLC from sorted snapshots.
    // Each entry: (minute_bin, open, high, low, close)
    let mut ohlc_minutes: Vec<(u64, f64, f64, f64, f64)> = Vec::new();

    for snap in &result.l1_snapshots {
        if snap.bid_price <= 0 || snap.ask_price <= 0 {
            continue;
        }
        let mid = (snap.bid_price as f64 + snap.ask_price as f64) / (2.0 * tick);
        let minute = snap.timestamp / ONE_MINUTE_NS;

        if let Some(last) = ohlc_minutes.last_mut() {
            if last.0 == minute {
                if mid > last.2 {
                    last.2 = mid;
                }
                if mid < last.3 {
                    last.3 = mid;
                }
                last.4 = mid;
                continue;
            }
        }
        ohlc_minutes.push((minute, mid, mid, mid, mid));
    }

    if ohlc_minutes.is_empty() {
        return Vec::new();
    }

    // Build volume / quote_volume from trades, keyed by minute bin.
    let mut vol_map: HashMap<u64, (f64, f64)> = HashMap::new();
    for trade in &result.trades {
        let minute = trade.timestamp / ONE_MINUTE_NS;
        let qty = trade.qty as f64;
        let notional = trade.price as f64 * qty / tick;
        vol_map
            .entry(minute)
            .and_modify(|(v, qv)| {
                *v += qty;
                *qv += notional;
            })
            .or_insert((qty, notional));
    }

    // Forward-fill gaps (minutes with no BBO change get previous close).
    let min_minute = ohlc_minutes[0].0;
    let max_minute = ohlc_minutes.last().expect("non-empty").0;
    let total = (max_minute - min_minute + 1) as usize;
    let mut candles = Vec::with_capacity(total);

    let mut ohlc_idx = 0;
    let mut last_close = ohlc_minutes[0].1; // first open

    for minute in min_minute..=max_minute {
        if ohlc_idx < ohlc_minutes.len() && ohlc_minutes[ohlc_idx].0 == minute {
            let (_, open, high, low, close) = ohlc_minutes[ohlc_idx];
            let (volume, quote_volume) = vol_map.get(&minute).copied().unwrap_or((0.0, 0.0));
            candles.push(MinuteCandle {
                open,
                high,
                low,
                close,
                volume,
                quote_volume,
            });
            last_close = close;
            ohlc_idx += 1;
        } else {
            candles.push(MinuteCandle {
                open: last_close,
                high: last_close,
                low: last_close,
                close: last_close,
                volume: 0.0,
                quote_volume: 0.0,
            });
        }
    }

    candles
}

/// Returns true if the given year is a leap year.
fn is_leap(y: i32) -> bool {
    (y % 4 == 0 && y % 100 != 0) || y % 400 == 0
}

/// Nanoseconds from Unix epoch to January 1st of the given year (UTC).
fn year_start_nanos(year: i32) -> i64 {
    let days: i64 = if year >= 1970 {
        (1970..year)
            .map(|y| if is_leap(y) { 366_i64 } else { 365 })
            .sum()
    } else {
        -(year..1970)
            .map(|y| if is_leap(y) { 366_i64 } else { 365 })
            .sum::<i64>()
    };
    days * 86_400 * 1_000_000_000
}

fn main() {
    let output_path = Path::new("klines_100y.parquet");

    println!(
        "Generating {NUM_YEARS} years of simulation data ({MAX_PARALLEL} threads)..."
    );
    let start = Instant::now();

    let mut all_candles: Vec<Candle> = Vec::with_capacity(CANDLES_PER_YEAR * NUM_YEARS);
    // Start at the simulation's real reference price (real_price = 100_000)
    // so the chained series reflects actual price levels, not arbitrary 100.0.
    let mut running_price = 100_000.0_f64;

    for batch_start in (0..NUM_YEARS).step_by(MAX_PARALLEL) {
        let batch_end = (batch_start + MAX_PARALLEL).min(NUM_YEARS);
        println!(
            "Batch {}-{}...",
            batch_start,
            batch_end.saturating_sub(1)
        );

        // Phase 1: Run simulations in parallel.
        let handles: Vec<_> = (batch_start..batch_end)
            .map(|i| thread::spawn(move || (i, run_one_year(i))))
            .collect();

        // Join in order so candle chaining is sequential.
        let results: Vec<(usize, SimulationResult)> = handles
            .into_iter()
            .map(|h| h.join().expect("simulation thread panicked"))
            .collect();

        // Phase 2: Compute candles, normalize, assign real timestamps.
        for (year_idx, result) in &results {
            let year_candles = compute_year_candles(result);
            if year_candles.is_empty() {
                println!("  Year {year_idx:03}: no candles (skipped)");
                continue;
            }

            let first_open = year_candles[0].open;
            if first_open <= 0.0 {
                println!("  Year {year_idx:03}: non-positive first open (skipped)");
                continue;
            }

            // Normalize: divide by first open, multiply by running_price.
            let scale = running_price / first_open;

            #[allow(clippy::cast_possible_wrap)]
            let real_year = START_YEAR + *year_idx as i32;
            let year_start_ns = year_start_nanos(real_year);
            let one_min_i64: i64 = 60_000_000_000;
            let n = year_candles.len();

            for (i, mc) in year_candles.iter().enumerate() {
                all_candles.push(Candle {
                    ts_nanos: year_start_ns + (i as i64) * one_min_i64,
                    open: mc.open * scale,
                    high: mc.high * scale,
                    low: mc.low * scale,
                    close: mc.close * scale,
                    volume: mc.volume,
                    quote_volume: mc.quote_volume,
                });
            }

            running_price = year_candles.last().expect("non-empty").close * scale;

            println!(
                "  Year {:03} ({}): {} candles, price {:.4} -> {:.4}",
                year_idx,
                real_year,
                n,
                first_open * scale,
                running_price,
            );
        }
    }

    let sim_elapsed = start.elapsed();
    println!(
        "Simulation + candles done in {:.1}s",
        sim_elapsed.as_secs_f64()
    );

    // Phase 3: Write single output parquet.
    println!(
        "Writing {} candles to {}...",
        all_candles.len(),
        output_path.display()
    );
    write_klines(output_path, &all_candles).expect("write klines parquet");

    let total_elapsed = start.elapsed();
    println!(
        "Done! {} candles written in {:.1}s",
        all_candles.len(),
        total_elapsed.as_secs_f64()
    );
    println!("Output: {}", output_path.display());
}
