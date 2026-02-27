use std::path::Path;
use std::time::Instant;

use serde::Deserialize;

use agents::{
    MarketMakerAgent, MarketMakerConfig, TrendFollowingAgent, TrendFollowingConfig, ZiAgent,
    ZiAgentConfig,
};
use sim_core::{Agent, Kernel, LatencyConfig, LatencyModelType, SimulationConfig};

/// Top-level JSON configuration.
#[derive(Debug, Deserialize)]
struct JsonConfig {
    seed: u64,
    /// Simulation duration in nanoseconds.
    end_time_ns: u64,
    tick_size: i64,
    lot_size: u64,
    /// Directory to write output parquet files.
    output_dir: String,
    /// Latency configuration (optional, defaults to NycSeattle).
    #[serde(default)]
    latency: Option<LatencyConfig>,
    /// Zero-intelligence agent groups.
    #[serde(default)]
    zi_agents: Vec<AgentGroup<ZiAgentConfig>>,
    /// Trend-following / contrarian agent groups.
    #[serde(default)]
    trend_following_agents: Vec<TrendFollowingGroup>,
    /// Market-maker agent groups.
    #[serde(default)]
    market_maker_agents: Vec<AgentGroup<MarketMakerConfig>>,
}

#[derive(Debug, Deserialize)]
struct AgentGroup<C> {
    count: usize,
    #[serde(flatten)]
    config: C,
}

#[derive(Debug, Deserialize)]
struct TrendFollowingGroup {
    count: usize,
    #[serde(flatten)]
    config: TrendFollowingConfig,
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 2 {
        eprintln!("Usage: run_json <config.json>");
        std::process::exit(1);
    }

    let json_str = std::fs::read_to_string(&args[1])
        .unwrap_or_else(|e| { eprintln!("Failed to read {}: {e}", args[1]); std::process::exit(1); });
    let cfg: JsonConfig = serde_json::from_str(&json_str)
        .unwrap_or_else(|e| { eprintln!("Failed to parse JSON: {e}"); std::process::exit(1); });

    let initial_price = 100_000_i64 * cfg.tick_size;

    let latency = cfg.latency.unwrap_or(LatencyConfig {
        default_base_ns: 50_000,
        jitter_mu: 0.0,
        jitter_sigma: 0.3,
        model: LatencyModelType::NycSeattle { seed: cfg.seed },
    });

    let sim_config = SimulationConfig {
        seed: cfg.seed,
        start_time: 0,
        end_time: cfg.end_time_ns,
        symbols: vec![0],
        latency,
        tick_size: cfg.tick_size,
        lot_size: cfg.lot_size,
        no_market_hours: true,
    };

    let mut agents: Vec<Box<dyn Agent>> = Vec::new();
    let mut next_seed: u64 = cfg.seed;
    let mut alloc_seed = || { let s = next_seed; next_seed += 1; s };

    // ZI agents
    for group in &cfg.zi_agents {
        let mut agent_cfg = group.config.clone();
        agent_cfg.reference_price = initial_price;
        agent_cfg.symbol = 0;
        for _ in 0..group.count {
            agents.push(Box::new(ZiAgent::new(agent_cfg.clone(), alloc_seed())));
        }
    }

    // Trend-following / contrarian agents
    for group in &cfg.trend_following_agents {
        let mut agent_cfg = group.config.clone();
        agent_cfg.reference_price = initial_price;
        agent_cfg.symbol = 0;
        for _ in 0..group.count {
            agents.push(Box::new(TrendFollowingAgent::new(agent_cfg.clone(), alloc_seed())));
        }
    }

    // Market-maker agents
    for group in &cfg.market_maker_agents {
        let mut agent_cfg = group.config.clone();
        agent_cfg.reference_price = initial_price;
        agent_cfg.symbol = 0;
        for _ in 0..group.count {
            agents.push(Box::new(MarketMakerAgent::new(agent_cfg.clone(), alloc_seed())));
        }
    }

    println!("Starting simulation ({} agents)...", agents.len());
    let start = Instant::now();
    let result = Kernel::run(&sim_config, agents);
    let elapsed = start.elapsed();

    // Write output
    std::fs::create_dir_all(&cfg.output_dir).unwrap_or_else(|e| {
        eprintln!("Failed to create output dir {}: {e}", cfg.output_dir);
        std::process::exit(1);
    });

    let out = Path::new(&cfg.output_dir);
    let _ = sim_core::output::write_l1_snapshots(
        &out.join("l1_snapshots.parquet"), &result.l1_snapshots, sim_config.tick_size, sim_config.lot_size,
    );
    let _ = sim_core::output::write_trades(
        &out.join("trades.parquet"), &result.trades, sim_config.tick_size, sim_config.lot_size,
    );

    println!("Simulation complete: {} events in {:.3}s", result.events_processed, elapsed.as_secs_f64());
    #[allow(clippy::cast_precision_loss)]
    if elapsed.as_secs_f64() > 0.0 {
        println!("Events/sec: {:.0}", result.events_processed as f64 / elapsed.as_secs_f64());
    }
    println!("L1 snapshots: {}", result.l1_snapshots.len());
    println!("Trades: {}", result.trades.len());
    println!("Output: {}", cfg.output_dir);
}
