# market-sim

Agent-based market simulator built in Rust. The goal is to simulate one year of multi-symbol market data in minutes, with discrete-event scheduling and independent order books per symbol.

## Architecture

```
market-sim/
├── crates/
│   └── cda-engine/       # Continuous Double Auction matching engine (this is what exists today)
│   └── sim-core/         # (planned) Discrete-event simulation loop, SimContext
│   └── agents/           # (planned) Agent traits, zero-intelligence agents, market makers
│   └── bin/runner/       # (planned) CLI runner for simulations
```

### cda-engine

The CDA engine is a single-threaded, zero-dependency order book with price-time (FIFO) priority. It supports limit orders and market orders with partial fills.

**Public API:**

| Method | Description |
|---|---|
| `OrderBook::new()` | Create an empty book |
| `add_limit_order(LimitOrder)` | Submit a limit order; crosses if possible, rests remainder |
| `add_market_order(MarketOrder)` | Execute immediately; unfilled remainder is cancelled |
| `cancel_order(order_id)` | Cancel a resting order by ID |
| `best_bid() / best_ask()` | Best price on each side, or `None` |
| `spread()` | Ask minus bid, or `None` |
| `volume_at(price, side)` | Total resting qty at a price level |
| `order_count()` | Number of resting orders |

**Key types:**

- `Side` — `Bid` or `Ask`
- `LimitOrder` — `{ id: u64, side, price: i64, qty: u64, timestamp: u64 }`
- `MarketOrder` — `{ id: u64, side, qty: u64 }`
- `Fill` — `{ maker_order_id, taker_order_id, price, qty, taker_side }`
- `OrderStatus` — `Filled | Resting { remaining_qty } | Placed | Cancelled { filled_qty }`
- `OrderResult` — `{ fills: Vec<Fill>, status: OrderStatus }`

**Data conventions:**

- Prices are `i64` in smallest tick units (e.g. cents). Non-negative.
- Quantities are `u64`, strictly positive.
- Order IDs are `u64`, assigned by the caller.
- Timestamps are `u64` nanoseconds, provided by the caller for FIFO ordering.

## Quick Start

```bash
# Run all tests
cargo test

# Run benchmarks
cargo bench

# Check lints
cargo clippy --all-targets
```

## Performance

Benchmarked on Apple Silicon (M-series). Mixed workload (30% limit adds, 60% cancels, 10% market orders):

| Benchmark | Result |
|---|---|
| Limit order (no cross) | ~39 ns |
| Mixed workload throughput | ~21M ops/sec |
| 1M operations batch | ~48 ms |

## Design Decisions

**Internals:** `BTreeMap<i64, VecDeque<RestingOrder>>` per side for price-level ordering, plus `HashMap<u64, (Side, i64)>` for O(1) cancel lookup. Empty price levels are removed eagerly.

**Fill buffer:** `OrderBook` owns a reusable `Vec<Fill>` that is cleared and drained per call, avoiding per-operation allocation. The caller receives an owned `Vec<Fill>` via `drain(..).collect()` while the book retains the buffer capacity.

**No self-trade prevention:** The engine matches any crossing orders regardless of origin. Self-trade prevention is the simulator's responsibility.

**Duplicate order IDs:** Submitting an ID already on the book overwrites the cancel-lookup entry. The caller must ensure unique IDs.

**Zero quantity:** `debug_assert!` panics in debug builds; no-op in release. The caller must enforce `qty > 0`.
