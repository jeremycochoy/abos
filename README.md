# Agent-Based Orderbook Simulation

High-performance agent-based market simulation in Rust. Simulates a continuous double auction (CDA) order book with configurable agents, latency models, and market structure. Generates a full year of realistic market data in under 15 seconds.

## Quick Start

```bash
# Run a full-year simulation with 20 ZI agents (~10s)
cargo run --release --bin zi_single_symbol

# Run a full-year simulation with 14 mixed agents (~5.5s)
cargo run --release --bin four_agents

# Visualize the results (requires Python: pandas, matplotlib, pyarrow)
python scripts/visualize.py l1_snapshots.parquet
```

## Architecture

```
├── crates/
│   ├── cda-engine/       # CDA matching engine (zero deps, ~21M ops/sec)
│   ├── sim-core/         # Discrete-event kernel, exchange, latency, Parquet output
│   ├── agents/           # ZI, TrendFollowing, Contrarian, MarketMaker agents
│   └── runner/           # Simulation binaries & integration tests (87 tests)
├── sims/                 # Simulation scenarios (zi_single_symbol, four_agents)
└── scripts/              # Python visualization with 5-min resampling
```

### CDA Engine (`cda-engine`)

Zero-dependency, single-threaded order book with price-time (FIFO) priority.

| Method | Description |
|---|---|
| `add_limit_order(LimitOrder)` | Submit limit order; crosses if possible, rests remainder |
| `add_market_order(MarketOrder)` | Execute immediately; unfilled remainder cancelled |
| `cancel_order(order_id)` | Cancel a resting order by ID (O(1)) |
| `best_bid() / best_ask()` | Best price on each side |
| `spread()` | Ask minus bid |

**Internals:** `BTreeMap<i64, PriceLevel>` per side for price-level ordering, `HashMap` + `HashSet` tombstones for O(1) cancel. Reusable `Vec<Fill>` buffer avoids per-operation allocation.

### Simulation Kernel (`sim-core`)

Discrete-event simulation with `BinaryHeap<Event>` priority queue. Features:

- Per-symbol `Exchange` wrapping the CDA engine
- Exchange-assigned unique order IDs
- Configurable market hours or continuous 24/7 trading (`no_market_hours: true`)
- L1 snapshot recording on every BBO change
- Parquet output for trades and L1 snapshots

### Agent Types

| Agent | Strategy | Key Parameters |
|-------|----------|---------------|
| **ZI** | Cancel-all + random limit order (lognormal price/size) | `price_std`, `order_size_scale/std`, `wake_up_interval_ns` |
| **TrendFollowing** | MA crossover → trade in trend direction | `short/long_window`, `threshold`, `price_offset` |
| **Contrarian** | MA crossover → mean-revert against trend | Same as TF with `contrarian: true` |
| **MarketMaker** | Multi-level symmetric-hump quotes with inventory skew | `total_liquidity`, `max_levels`, `imbalance_beta` |

The ZI agent matches the ABIDES `ZeroIntelligence` implementation (lognormal price distribution, lognormal order size, cancel-all-then-place cycle). The trend-following and contrarian agents match the `TrendFollowingAgent` from `evolve_trading/simple_4agents_model`. The market maker uses the `SymmetricHumpLiquidityModel` from ABIDES.

### Latency Models

| Model | Description |
|-------|-------------|
| **Uniform** | Constant base latency + log-normal jitter (default) |
| **NycSeattle** | Agents on NYC-Seattle line, light-speed proportional delays |
| **Cubic** | ABIDES cubic jitter model with heavy-tailed delays |
| **NoLatency** | Zero delay (for testing) |

## Output Format

Two Parquet files per simulation:

- **`trades.parquet`** — timestamp, symbol, price, qty, aggressor_side, maker/taker order IDs
- **`l1_snapshots.parquet`** — timestamp, symbol, bid/ask price/volume, last_trade_price

Both include `tick_size` and `lot_size` in schema metadata for normalization.

## Visualization

```bash
python scripts/visualize.py l1_snapshots.parquet           # interactive plot
python scripts/visualize.py l1_snapshots.parquet --save     # save PNG
```

Shows 4 panels: mid-price with bid-ask band, spread, volume, and returns (5-min resampling).

## Performance

| Scenario | Wall-clock |
|----------|-----------|
| 20 ZI agents, 1 year | ~10 seconds |
| 14 mixed agents, 1 year | ~5.5 seconds |
| CDA engine throughput | ~21M ops/sec |
| Event throughput | ~9.5M events/sec |

## Development

```bash
cargo test                    # 87 tests across 7 test files
cargo clippy --all-targets    # must pass with zero warnings
cargo bench                   # criterion benchmarks
```

**Type conventions:** Prices `i64` (tick units), quantities `u64`, order IDs `u64` (exchange-assigned), timestamps `u64` (nanoseconds), agent IDs `usize`, symbols `u32`.

**Code conventions:** `#![deny(clippy::all)]` + `#![warn(clippy::pedantic)]`, no `unwrap()` in library code, no async/threads. `cda-engine` has zero runtime dependencies.
