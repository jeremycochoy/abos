use criterion::{criterion_group, criterion_main, Criterion};

use agents::{ZiAgent, ZiAgentConfig};
use sim_core::{Agent, Kernel, LatencyConfig, SimulationConfig};

fn make_agents(n: usize, seed: u64) -> Vec<Box<dyn Agent>> {
    let cfg = ZiAgentConfig {
        p_limit: 0.70,
        p_cancel: 0.10,
        mean_wakeup_interval_ns: 10_000_000,
        price_offset_lambda: 0.2,
        default_qty: 1,
        reference_price: 10_000,
        tick_size: 1,
        symbol: 0,
    };
    (0..n)
        .map(|i| Box::new(ZiAgent::new(cfg.clone(), seed.wrapping_add(i as u64))) as Box<dyn Agent>)
        .collect()
}

fn sim_config(duration_ns: u64) -> SimulationConfig {
    SimulationConfig {
        seed: 42,
        start_time: 0,
        end_time: duration_ns,
        symbols: vec![0],
        latency: LatencyConfig {
            default_base_ns: 50_000,
            jitter_mu: 0.0,
            jitter_sigma: 0.3,
        },
        tick_size: 100,
        lot_size: 1,
    }
}

fn bench_full_simulation(c: &mut Criterion) {
    let cfg = sim_config(600_000_000_000); // 10 minutes
    c.bench_function("100_agents_10min", |b| {
        b.iter(|| Kernel::run(&cfg, make_agents(100, 42)));
    });
}

fn bench_short_simulation(c: &mut Criterion) {
    let cfg = sim_config(10_000_000_000); // 10 seconds
    c.bench_function("100_agents_10sec", |b| {
        b.iter(|| Kernel::run(&cfg, make_agents(100, 42)));
    });
}

criterion_group!(benches, bench_short_simulation, bench_full_simulation);
criterion_main!(benches);
