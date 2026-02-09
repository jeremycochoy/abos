# CLAUDE.md

## Commands

```bash
cargo test                    # run all tests (87 tests across 7 test files)
cargo clippy --all-targets    # lint check — must pass with zero warnings
cargo bench                   # run criterion benchmarks

# Run simulations (release mode)
cargo run --release --bin zi_single_symbol    # 20 ZI agents, 1 year
cargo run --release --bin four_agents         # 14 agents (9 ZI + 3 TF + 1 TC + 1 MM), 1 year

# Visualize results
python scripts/visualize.py l1_snapshots.parquet           # interactive plot
python scripts/visualize.py l1_snapshots.parquet --save     # save PNG
```

## Repository Structure

Cargo workspace with four crates under `crates/`, simulation scenarios under `sims/`, and visualization scripts under `scripts/`.

```
├── Cargo.toml                              # workspace root (resolver = "2")
├── CLAUDE.md
├── README.md
├── crates/
│   ├── cda-engine/                         # Continuous Double Auction matching engine
│   │   ├── src/
│   │   │   ├── lib.rs                      # crate root, re-exports public API
│   │   │   ├── order.rs                    # Side, LimitOrder, MarketOrder, RestingOrder
│   │   │   ├── fills.rs                    # Fill, OrderStatus, OrderResult
│   │   │   └── orderbook.rs               # OrderBook — all matching logic
│   │   ├── tests/
│   │   │   ├── correctness.rs             # 20 scenario tests
│   │   │   └── edge_cases.rs              # 7 edge case tests
│   │   └── benches/
│   │       └── orderbook_bench.rs         # 6 criterion benchmarks
│   ├── sim-core/                           # Discrete-event simulation kernel
│   │   └── src/
│   │       ├── lib.rs                      # crate root
│   │       ├── config.rs                   # SimulationConfig (with no_market_hours)
│   │       ├── types.rs                    # AgentId, Symbol, Nanos, MarketSnapshot
│   │       ├── event.rs                    # Event, EventPayload, OrderAction
│   │       ├── agent.rs                    # Agent trait, AgentAction
│   │       ├── kernel.rs                   # Kernel event loop, SimulationResult
│   │       ├── exchange.rs                 # Per-symbol Exchange, TradeRecord, L1Snapshot
│   │       ├── latency.rs                  # LatencyModel (Uniform, NycSeattle, Cubic, NoLatency)
│   │       └── output.rs                   # Parquet export (trades + L1 snapshots)
│   ├── agents/                             # Agent implementations
│   │   └── src/
│   │       ├── lib.rs                      # re-exports all agents
│   │       ├── utils.rs                    # IndexedSet, mid_price helper
│   │       ├── zi_agent.rs                 # ZiAgent (ABIDES-compatible, lognormal)
│   │       ├── trend_following_agent.rs    # TrendFollowingAgent (MA crossover / contrarian)
│   │       └── market_maker_agent.rs       # MarketMakerAgent (symmetric-hump liquidity)
│   └── runner/                             # Simulation runner & integration tests
│       └── tests/
│           ├── simulation_correctness.rs   # 18 tests (agents, latency, market hours)
│           └── simulation_realism.rs       # 6 realism tests
├── sims/
│   ├── zi_single_symbol.rs                 # ZI-only experiment (20 agents, 1 year)
│   └── four_agents.rs                      # 4-type experiment (14 agents, 1 year)
└── scripts/
    └── visualize.py                        # Python visualization with 5-min resampling
```

## Code Conventions

- `#![deny(clippy::all)]` and `#![warn(clippy::pedantic)]` at crate root.
- All public types and functions have doc comments.
- No `unwrap()` in library code. `debug_assert!` for invariants, `expect()` only for internal invariants that indicate bugs.
- `cda-engine`: no runtime dependencies (std only), no async, no threads, no I/O.
- `sim-core`: depends on `arrow`/`parquet` for output, `rand` for RNG. No async.

## Type Conventions

- Prices: `i64` (tick units, non-negative)
- Quantities: `u64` (strictly positive)
- Order IDs: `u64` (exchange-assigned, unique per symbol)
- Timestamps: `u64` (nanoseconds)
- Agent IDs: `usize` (index into kernel's agent vector)
- Symbols: `u32`

## Key Architecture Details

### OrderBook internals (cda-engine/orderbook.rs)
- **Bid side:** `BTreeMap<i64, PriceLevel>` — best bid via `.last_key_value()`
- **Ask side:** `BTreeMap<i64, PriceLevel>` — best ask via `.first_key_value()`
- **Cancel:** `HashMap<u64, (Side, i64, u64)>` for O(1) cancel + `HashSet<u64>` tombstones
- Empty price levels removed immediately after last order is filled/cancelled
- `fill_buf: Vec<Fill>` is reusable, cleared per call, drained into `OrderResult`

### Kernel (sim-core/kernel.rs)
- `BinaryHeap<Event>` min-heap ordered by (delivery_time, seq)
- Sequence numbers ensure FIFO for same-timestamp events
- Reusable buffers: `snap_buf`, `action_buf`, `msg_buf` (zero-alloc hot path)
- `no_market_hours: true` → market always open, no open/close events

### Agent trait (sim-core/agent.rs)
```rust
pub trait Agent {
    fn wakeup_into(&mut self, time, agent_id, snapshots, actions: &mut Vec<AgentAction>);
    fn on_exchange_message(&mut self, time, agent_id, message: ExchangeMessage);
}
```
Object-safe, zero-allocation design (actions written to borrowed buffer).

### Agents (agents/)
| Agent | Behavior | Key Params |
|-------|----------|------------|
| `ZiAgent` | Cancel-all + place random limit order | `price_std`, `order_size_scale/std`, `wake_up_interval_ns` |
| `TrendFollowingAgent` | MA crossover → follow trend | `short/long_window`, `threshold`, `price_offset`, `contrarian` |
| `TrendFollowingAgent` (contrarian) | MA crossover → mean-revert | Same as above with `contrarian: true` |
| `MarketMakerAgent` | Multi-level symmetric-hump quotes | `total_liquidity`, `max_levels`, `imbalance_beta` |

### Latency Model (sim-core/latency.rs)
- **Uniform:** constant base + log-normal jitter (default)
- **NycSeattle:** agents on NYC-Seattle line, latency ∝ light-speed distance
- **Cubic:** ABIDES-compatible cubic jitter model (heavy tail)
- **NoLatency:** zero delay

### Parquet Output (sim-core/output.rs)
- `trades.parquet`: timestamp, symbol, price, qty, aggressor_side, maker/taker IDs
- `l1_snapshots.parquet`: timestamp, symbol, bid/ask price/volume, last_trade_price
- Both include `tick_size` and `lot_size` in schema metadata

## Test Coverage

### cda-engine: correctness.rs (20 tests) + edge_cases.rs (7 tests)
Matching engine: placement, fills, partial fills, sweeps, FIFO, cancels, BBO, spread, volume.

### agents: unit tests (36 tests)
- **ZiAgent** (10 tests): Price distribution center/std matches ABIDES lognormal formula, order size distribution mean/min, cancel-all, exactly one limit order, fixed-interval wakeup, 50/50 side balance, reference price fallback.
- **TrendFollowingAgent** (12 tests): MA computation correctness, insufficient history, TF buy/sell/threshold, contrarian sell/buy/threshold, order size proportional to signal, price offset, no trade before enough candles, candle sampling frequency, cancel-all before trading.
- **MarketMakerAgent** (14 tests): Hump weight positivity/peak-decay/symmetry, inventory imbalance (zero/positive/negative/bounded), both-sides placement, correct number of levels, bid below/ask above mid, total qty matches liquidity budget, cancel-all before placing, imbalance shifts liquidity, fill tracking.

### runner: simulation_correctness.rs (18 tests)
Determinism, market open/close, no_market_hours mode, empty book, forced trade, latency models (uniform, NYC-Seattle), fill conservation, order ID uniqueness, all agent types individually, mixed-agent simulation.

### runner: simulation_realism.rs (6 tests)
Trade generation, price stability, positive spread, two-sided book, cancellations, performance.

## Performance

- CDA engine: ~21M ops/sec individual operations
- Full simulation: ~9.5M events/sec
- **20 ZI agents × 1 year: ~10 seconds** (release mode)
- **14 mixed agents × 1 year: ~5.5 seconds** (release mode)
