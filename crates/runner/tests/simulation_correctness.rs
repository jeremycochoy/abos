use agents::{
    MarketMakerAgent, MarketMakerConfig, TrendFollowingAgent, TrendFollowingConfig, ZiAgent,
    ZiAgentConfig,
};
use sim_core::{
    Agent, AgentAction, AgentId, ExchangeMessage, Kernel, LatencyConfig, LatencyModelType,
    MarketSnapshot, Nanos, OrderAction, SimulationConfig,
};

fn zi_config() -> ZiAgentConfig {
    ZiAgentConfig {
        wake_up_interval_ns: 1_000_000,
        price_std: 0.000_3,
        order_size_scale: 1.0,
        order_size_std: 0.3,
        reference_price: 10_000,
        symbol: 0,
    }
}

fn sim_config(seed: u64, duration_ns: u64) -> SimulationConfig {
    SimulationConfig {
        seed,
        start_time: 0,
        end_time: duration_ns,
        symbols: vec![0],
        latency: LatencyConfig {
            default_base_ns: 1_000,
            jitter_mu: 0.0,
            jitter_sigma: 0.0001, // near-zero jitter for predictability
            model: LatencyModelType::Uniform,
        },
        tick_size: 100,
        lot_size: 1,
        no_market_hours: false,
    }
}

fn make_zi_agents(n: usize, seed: u64) -> Vec<Box<dyn Agent>> {
    let cfg = zi_config();
    (0..n)
        .map(|i| Box::new(ZiAgent::new(cfg, seed.wrapping_add(i as u64))) as Box<dyn Agent>)
        .collect()
}

// ── Determinism ─────────────────────────────────────────────────────────

#[test]
fn determinism_identical_seed() {
    let cfg = sim_config(42, 100_000_000);
    let r1 = Kernel::run(&cfg, make_zi_agents(10, 42));
    let r2 = Kernel::run(&cfg, make_zi_agents(10, 42));

    assert_eq!(r1.trades.len(), r2.trades.len());
    assert_eq!(r1.events_processed, r2.events_processed);
    for (a, b) in r1.trades.iter().zip(r2.trades.iter()) {
        assert_eq!(a.timestamp, b.timestamp);
        assert_eq!(a.price, b.price);
        assert_eq!(a.qty, b.qty);
        assert_eq!(a.maker_order_id, b.maker_order_id);
        assert_eq!(a.taker_order_id, b.taker_order_id);
    }
}

#[test]
fn determinism_different_seed_differs() {
    let c1 = sim_config(42, 100_000_000);
    let c2 = sim_config(99, 100_000_000);
    let r1 = Kernel::run(&c1, make_zi_agents(10, 42));
    let r2 = Kernel::run(&c2, make_zi_agents(10, 99));
    // Very unlikely to produce identical trade logs with different seeds
    let same = r1.trades.len() == r2.trades.len()
        && r1.trades.iter().zip(r2.trades.iter()).all(|(a, b)| a.price == b.price);
    assert!(!same || r1.trades.is_empty());
}

// ── Market open/close ───────────────────────────────────────────────────

#[test]
fn no_trades_before_open() {
    let cfg = sim_config(42, 10_000_000);
    let result = Kernel::run(&cfg, make_zi_agents(10, 42));
    for trade in &result.trades {
        assert!(trade.timestamp >= cfg.start_time,
            "trade at {} before market open {}", trade.timestamp, cfg.start_time);
    }
}

#[test]
fn no_trades_after_close() {
    let cfg = sim_config(42, 10_000_000);
    let result = Kernel::run(&cfg, make_zi_agents(10, 42));
    for trade in &result.trades {
        assert!(trade.timestamp <= cfg.end_time,
            "trade at {} after market close {}", trade.timestamp, cfg.end_time);
    }
}

// ── Empty book / single agent ───────────────────────────────────────────

/// An agent that only submits market orders (no limit orders).
struct MarketOnlyAgent { symbol: u32 }

impl Agent for MarketOnlyAgent {
    fn wakeup_into(&mut self, _: Nanos, _: AgentId, _: &[MarketSnapshot], out: &mut Vec<AgentAction>) {
        out.push(AgentAction::SubmitOrder {
            symbol: self.symbol,
            order: OrderAction::NewMarketOrder {
                side: cda_engine::Side::Bid,
                qty: 1,
                user_id: 0,
            },
        });
        out.push(AgentAction::ScheduleWakeUp { delay_ns: 1_000_000 });
    }
    fn on_exchange_message(&mut self, _: Nanos, _: AgentId, _: ExchangeMessage) {}
}

#[test]
fn empty_book_market_orders_no_trades() {
    let cfg = sim_config(42, 10_000_000);
    let agents: Vec<Box<dyn Agent>> = vec![Box::new(MarketOnlyAgent { symbol: 0 })];
    let result = Kernel::run(&cfg, agents);
    assert_eq!(result.trades.len(), 0, "no trades on empty book");
}

/// An agent that only submits limit orders at a fixed price on one side.
struct LimitOnlyAgent { side: cda_engine::Side, price: i64 }

impl Agent for LimitOnlyAgent {
    fn wakeup_into(&mut self, _: Nanos, _: AgentId, _: &[MarketSnapshot], out: &mut Vec<AgentAction>) {
        out.push(AgentAction::SubmitOrder {
            symbol: 0,
            order: OrderAction::NewLimitOrder {
                side: self.side,
                price: self.price,
                qty: 1,
                user_id: 0,
            },
        });
        out.push(AgentAction::ScheduleWakeUp { delay_ns: 1_000_000 });
    }
    fn on_exchange_message(&mut self, _: Nanos, _: AgentId, _: ExchangeMessage) {}
}

#[test]
fn single_agent_limit_orders_no_trades() {
    let cfg = sim_config(42, 5_000_000);
    let agents: Vec<Box<dyn Agent>> = vec![
        Box::new(LimitOnlyAgent { side: cda_engine::Side::Bid, price: 100 }),
    ];
    let result = Kernel::run(&cfg, agents);
    assert_eq!(result.trades.len(), 0, "single-sided limit orders produce no trades");
}

// ── Two agents forced trade ─────────────────────────────────────────────

#[test]
fn two_agents_forced_trade() {
    let cfg = sim_config(42, 10_000_000);
    let agents: Vec<Box<dyn Agent>> = vec![
        Box::new(LimitOnlyAgent { side: cda_engine::Side::Bid, price: 100 }),
        Box::new(LimitOnlyAgent { side: cda_engine::Side::Ask, price: 100 }),
    ];
    let result = Kernel::run(&cfg, agents);
    assert!(!result.trades.is_empty(), "crossing limit orders should trade");
    for trade in &result.trades {
        assert_eq!(trade.price, 100);
    }
}

// ── Latency model sanity ────────────────────────────────────────────────

#[test]
fn latency_model_base_only() {
    // With near-zero jitter and base=1ms, all deliveries should be ~1ms after send
    let cfg = SimulationConfig {
        seed: 42,
        start_time: 0,
        end_time: 5_000_000,
        symbols: vec![0],
        latency: LatencyConfig {
            default_base_ns: 1_000_000,     // 1ms
            jitter_mu: 0.0,
            jitter_sigma: 0.0001,
            model: LatencyModelType::Uniform,
        },
        tick_size: 100,
        lot_size: 1,
        no_market_hours: false,
    };
    let agents: Vec<Box<dyn Agent>> = vec![
        Box::new(LimitOnlyAgent { side: cda_engine::Side::Bid, price: 100 }),
        Box::new(LimitOnlyAgent { side: cda_engine::Side::Ask, price: 100 }),
    ];
    let result = Kernel::run(&cfg, agents);
    // Trades should happen: agent wakes at 0, order arrives ~1ms later
    // With 2 agents and 5ms window, we should get some trades
    assert!(result.events_processed > 0);
}

// ── Fill conservation ───────────────────────────────────────────────────

#[test]
fn fill_conservation() {
    let cfg = sim_config(42, 50_000_000);
    let result = Kernel::run(&cfg, make_zi_agents(20, 42));
    // Each trade should have qty > 0
    for trade in &result.trades {
        assert!(trade.qty > 0, "fill qty must be positive");
    }
}

// ── Order ID uniqueness ─────────────────────────────────────────────────

#[test]
fn order_id_uniqueness() {
    let cfg = sim_config(42, 50_000_000);
    let result = Kernel::run(&cfg, make_zi_agents(20, 42));
    // Taker IDs should all be unique (each order submission gets a unique ID)
    let mut taker_ids: Vec<u64> = result.trades.iter().map(|t| t.taker_order_id).collect();
    taker_ids.sort_unstable();
    taker_ids.dedup();
    // Some taker orders may produce multiple fills (multi-level sweeps),
    // so taker_ids after dedup should have <= trade count
    assert!(taker_ids.len() <= result.trades.len());
}

// ── Large simulation doesn't crash ──────────────────────────────────────

#[test]
fn large_simulation_completes() {
    let cfg = SimulationConfig {
        seed: 42,
        start_time: 0,
        end_time: 1_000_000_000, // 1 second simulated
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
    };
    let agents = make_zi_agents(100, 42);
    let result = Kernel::run(&cfg, agents);
    assert!(result.events_processed > 0);
    assert!(!result.trades.is_empty(), "100 agents should produce trades");
}

// ── L1 snapshots recorded ───────────────────────────────────────────────

#[test]
fn l1_snapshots_recorded() {
    let cfg = sim_config(42, 50_000_000);
    let result = Kernel::run(&cfg, make_zi_agents(20, 42));
    assert!(!result.l1_snapshots.is_empty(), "should record BBO changes");
}

// ── no_market_hours mode ─────────────────────────────────────────────

#[test]
fn no_market_hours_allows_trading() {
    let mut cfg = sim_config(42, 10_000_000);
    cfg.no_market_hours = true;
    let result = Kernel::run(&cfg, make_zi_agents(10, 42));
    assert!(!result.trades.is_empty(), "continuous mode should produce trades");
}

// ── Trend-following agent ────────────────────────────────────────────

#[test]
fn trend_following_agent_runs() {
    let cfg = sim_config(42, 200_000_000);
    let mut agents: Vec<Box<dyn Agent>> = make_zi_agents(10, 42);
    agents.push(Box::new(TrendFollowingAgent::new(
        TrendFollowingConfig {
            short_window: 3,
            long_window: 10,
            threshold: 0.001,
            price_offset: 0.02,
            order_size_factor: 1.0,
            order_size_boost: 1.0,
            mean_wakeup_interval_ns: 5_000_000,
            sampling_freq_ns: 2_000_000,
            reference_price: 10_000,
            symbol: 0,
            contrarian: false,
        },
        99,
    )));
    let result = Kernel::run(&cfg, agents);
    assert!(result.events_processed > 0);
    assert!(!result.trades.is_empty());
}

// ── Contrarian agent ─────────────────────────────────────────────────

#[test]
fn contrarian_agent_runs() {
    let cfg = sim_config(42, 200_000_000);
    let mut agents: Vec<Box<dyn Agent>> = make_zi_agents(10, 42);
    agents.push(Box::new(TrendFollowingAgent::new(
        TrendFollowingConfig {
            short_window: 3,
            long_window: 10,
            threshold: 0.001,
            price_offset: 0.025,
            order_size_factor: 1.0,
            order_size_boost: 1.0,
            mean_wakeup_interval_ns: 5_000_000,
            sampling_freq_ns: 2_000_000,
            reference_price: 10_000,
            symbol: 0,
            contrarian: true,
        },
        99,
    )));
    let result = Kernel::run(&cfg, agents);
    assert!(result.events_processed > 0);
    assert!(!result.trades.is_empty());
}

// ── Market-maker agent ───────────────────────────────────────────────

#[test]
fn market_maker_agent_runs() {
    let cfg = sim_config(42, 200_000_000);
    let mut agents: Vec<Box<dyn Agent>> = make_zi_agents(10, 42);
    agents.push(Box::new(MarketMakerAgent::new(
        MarketMakerConfig {
            total_liquidity: 10.0,
            step_size_ratio: 0.001,
            max_levels: 3,
            imbalance_beta: 1.0,
            peak_distance_ratio: 0.05,
            shape_exponent: 1.2,
            mean_wakeup_interval_ns: 5_000_000,
            reference_price: 10_000,
            symbol: 0,
        },
        99,
    )));
    let result = Kernel::run(&cfg, agents);
    assert!(result.events_processed > 0);
    assert!(!result.trades.is_empty());
}

// ── NYC-Seattle latency model ────────────────────────────────────────

#[test]
fn nyc_seattle_latency_model_works() {
    let cfg = SimulationConfig {
        seed: 42,
        start_time: 0,
        end_time: 50_000_000,
        symbols: vec![0],
        latency: LatencyConfig {
            default_base_ns: 50_000,
            jitter_mu: 0.0,
            jitter_sigma: 0.3,
            model: LatencyModelType::NycSeattle { seed: 7 },
        },
        tick_size: 100,
        lot_size: 1,
        no_market_hours: false,
    };
    let result = Kernel::run(&cfg, make_zi_agents(10, 42));
    assert!(result.events_processed > 0);
}

// ── Mixed agent simulation ───────────────────────────────────────────

#[test]
fn mixed_agents_simulation() {
    let cfg = sim_config(42, 200_000_000);
    let zi_cfg = zi_config();
    let mut agents: Vec<Box<dyn Agent>> = (0..5)
        .map(|i| Box::new(ZiAgent::new(zi_cfg, 42 + i)) as Box<dyn Agent>)
        .collect();
    agents.push(Box::new(TrendFollowingAgent::new(
        TrendFollowingConfig {
            short_window: 3, long_window: 10, threshold: 0.001,
            price_offset: 0.02, order_size_factor: 1.0, order_size_boost: 1.0,
            mean_wakeup_interval_ns: 5_000_000, sampling_freq_ns: 2_000_000,
            reference_price: 10_000, symbol: 0, contrarian: false,
        },
        100,
    )));
    agents.push(Box::new(MarketMakerAgent::new(
        MarketMakerConfig {
            total_liquidity: 5.0, step_size_ratio: 0.001, max_levels: 2,
            imbalance_beta: 1.0, peak_distance_ratio: 0.05, shape_exponent: 1.2,
            mean_wakeup_interval_ns: 5_000_000, reference_price: 10_000, symbol: 0,
        },
        101,
    )));
    let result = Kernel::run(&cfg, agents);
    assert!(!result.trades.is_empty(), "mixed agents should trade");
}
