# CLAUDE.md

## Commands

```bash
cargo test                    # run all tests (27 tests across 2 test files)
cargo clippy --all-targets    # lint check — must pass with zero warnings
cargo bench                   # run criterion benchmarks (6 benchmarks)
```

## Repository Structure

Cargo workspace with one crate so far. Future crates (`sim-core`, `agents`, `bin/runner`) will be added as siblings under `crates/`.

```
market-sim/
├── Cargo.toml                              # workspace root (resolver = "2")
└── crates/cda-engine/
    ├── Cargo.toml                          # no runtime deps, dev: criterion + rand
    ├── src/
    │   ├── lib.rs                          # crate root, re-exports public API
    │   ├── order.rs                        # Side, LimitOrder, MarketOrder, RestingOrder(pub(crate))
    │   ├── fills.rs                        # Fill, OrderStatus, OrderResult
    │   └── orderbook.rs                    # OrderBook — all matching logic
    ├── tests/
    │   ├── correctness.rs                  # 20 scenario tests
    │   └── edge_cases.rs                   # 7 edge case tests
    └── benches/
        └── orderbook_bench.rs              # 6 criterion benchmarks
```

## Code Conventions

- `#![deny(clippy::all)]` and `#![warn(clippy::pedantic)]` at crate root.
- All public types and functions have doc comments.
- No `unwrap()` in library code. `debug_assert!` for invariants, `expect()` only for internal invariants that indicate bugs.
- No runtime dependencies — std only.
- No async, no threads, no locks.
- No `serde`, no logging, no I/O in the engine crate.

## Type Conventions

- Prices: `i64` (tick units, non-negative)
- Quantities: `u64` (strictly positive)
- Order IDs: `u64` (caller-assigned, must be unique)
- Timestamps: `u64` (nanoseconds, caller-provided for FIFO priority)
- No `f64`, no `Decimal`, no `u128`

## Key Architecture Details

### OrderBook internals (orderbook.rs)
- **Bid side:** `BTreeMap<i64, VecDeque<RestingOrder>>` — iterate with `.last_key_value()` for best bid
- **Ask side:** `BTreeMap<i64, VecDeque<RestingOrder>>` — iterate with `.first_key_value()` for best ask
- **Cancel lookup:** `HashMap<u64, (Side, i64)>` maps order_id to (side, price) for O(1) cancel
- Empty price levels are removed immediately after last order is filled/cancelled
- `fill_buf: Vec<Fill>` is an internal reusable buffer, cleared per call, drained into `OrderResult`

### Matching flow
1. `add_limit_order` / `add_market_order` clears `fill_buf`
2. Calls `match_against_asks` (for bids) or `match_against_bids` (for asks)
3. These loop over price levels calling `fill_at_level` which drains the front of each `VecDeque`
4. For limit orders, unfilled remainder is placed via `place_resting`
5. For market orders, unfilled remainder produces `OrderStatus::Cancelled`
6. Fills are moved out via `self.fill_buf.drain(..).collect()`

### Caller responsibilities (the simulator must enforce these)
- Unique order IDs (duplicate IDs overwrite the cancel-lookup; old order becomes orphaned but still matchable)
- `qty > 0` (debug_assert only)
- `price >= 0` for limit orders (debug_assert only)
- No self-trade prevention — engine matches all crossing orders
- No tick size validation

## Test Coverage

### correctness.rs (20 tests)
Covers: basic placement, basic match, partial fills (both directions), multi-level sweep, price-time priority, market order full/partial/empty, cancel existing/nonexistent/re-add, self-trade, BBO updates, spread, FIFO ordering, exact price cross, inversion sweep, large quantities, volume tracking.

### edge_cases.rs (7 tests)
Covers: zero qty panics (debug mode), duplicate order ID behavior, cancel after fill, single-tick book, rapid fill-cancel-add cycles (1000 iterations), full book drain.

## Performance Targets

- Individual operations: sub-microsecond
- Mixed workload: >5M ops/sec (currently ~21M ops/sec)
- Measured with criterion; benchmark results in `target/criterion/`
