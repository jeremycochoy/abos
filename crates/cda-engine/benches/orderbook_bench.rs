use criterion::{black_box, criterion_group, criterion_main, Criterion};
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};

use cda_engine::{LimitOrder, MarketOrder, OrderBook, Side};

/// Build a book with `n` resting orders on each side, centered around `mid`.
fn populated_book(n: u64, mid: i64) -> OrderBook {
    let mut book = OrderBook::new();
    for i in 0..n {
        book.add_limit_order(LimitOrder {
            id: i,
            side: Side::Bid,
            price: mid - 1 - (i as i64 % 50),
            qty: 10,
            timestamp: i,
        });
        book.add_limit_order(LimitOrder {
            id: n + i,
            side: Side::Ask,
            price: mid + 1 + (i as i64 % 50),
            qty: 10,
            timestamp: i,
        });
    }
    book
}

fn bench_add_limit_no_cross(c: &mut Criterion) {
    c.bench_function("add_limit_no_cross", |b| {
        let mut book = populated_book(50, 1000);
        let mut id = 10_000_u64;
        b.iter(|| {
            id += 1;
            black_box(book.add_limit_order(LimitOrder {
                id,
                side: Side::Bid,
                price: 900,
                qty: 1,
                timestamp: id,
            }));
        });
    });
}

fn bench_add_limit_single_fill(c: &mut Criterion) {
    c.bench_function("add_limit_single_fill", |b| {
        b.iter_batched(
            || populated_book(50, 1000),
            |mut book| {
                black_box(book.add_limit_order(LimitOrder {
                    id: 99999,
                    side: Side::Bid,
                    price: 1001,
                    qty: 5,
                    timestamp: 99999,
                }));
            },
            criterion::BatchSize::SmallInput,
        );
    });
}

fn bench_market_order_sweep(c: &mut Criterion) {
    c.bench_function("market_order_sweep_10_levels", |b| {
        b.iter_batched(
            || populated_book(100, 1000),
            |mut book| {
                black_box(book.add_market_order(MarketOrder {
                    id: 99999,
                    side: Side::Bid,
                    qty: 100,
                }));
            },
            criterion::BatchSize::SmallInput,
        );
    });
}

fn bench_cancel(c: &mut Criterion) {
    c.bench_function("cancel_order_1k_book", |b| {
        b.iter_batched(
            || {
                let book = populated_book(500, 1000);
                (book, 250_u64) // cancel an order in the middle
            },
            |(mut book, id)| {
                black_box(book.cancel_order(id));
            },
            criterion::BatchSize::SmallInput,
        );
    });
}

fn bench_mixed_workload(c: &mut Criterion) {
    c.bench_function("mixed_workload_10k_ops", |b| {
        b.iter_batched(
            || {
                let book = populated_book(100, 10_000);
                let rng = SmallRng::seed_from_u64(42);
                (book, rng, 200_000_u64)
            },
            |(mut book, mut rng, mut next_id): (OrderBook, SmallRng, u64)| {
                for _ in 0..10_000 {
                    let r: u32 = rng.gen_range(0..100);
                    next_id += 1;
                    if r < 30 {
                        // 30% limit add
                        let side = if rng.gen_bool(0.5) { Side::Bid } else { Side::Ask };
                        let price = rng.gen_range(9_900..10_100);
                        black_box(book.add_limit_order(LimitOrder {
                            id: next_id,
                            side,
                            price,
                            qty: rng.gen_range(1..100),
                            timestamp: next_id,
                        }));
                    } else if r < 90 {
                        // 60% cancel
                        let cancel_id = rng.gen_range(0..next_id);
                        black_box(book.cancel_order(cancel_id));
                    } else {
                        // 10% market order
                        let side = if rng.gen_bool(0.5) { Side::Bid } else { Side::Ask };
                        black_box(book.add_market_order(MarketOrder {
                            id: next_id,
                            side,
                            qty: rng.gen_range(1..50),
                        }));
                    }
                }
            },
            criterion::BatchSize::SmallInput,
        );
    });
}

fn bench_throughput_1m(c: &mut Criterion) {
    c.bench_function("throughput_1m_ops", |b| {
        b.iter_batched(
            || {
                let book = populated_book(100, 10_000);
                let rng = SmallRng::seed_from_u64(123);
                (book, rng, 200_000_u64)
            },
            |(mut book, mut rng, mut next_id): (OrderBook, SmallRng, u64)| {
                for _ in 0..1_000_000 {
                    let r: u32 = rng.gen_range(0..100);
                    next_id += 1;
                    if r < 30 {
                        let side = if rng.gen_bool(0.5) { Side::Bid } else { Side::Ask };
                        let price = rng.gen_range(9_900..10_100);
                        black_box(book.add_limit_order(LimitOrder {
                            id: next_id,
                            side,
                            price,
                            qty: rng.gen_range(1..100),
                            timestamp: next_id,
                        }));
                    } else if r < 90 {
                        let cancel_id = rng.gen_range(0..next_id);
                        black_box(book.cancel_order(cancel_id));
                    } else {
                        let side = if rng.gen_bool(0.5) { Side::Bid } else { Side::Ask };
                        black_box(book.add_market_order(MarketOrder {
                            id: next_id,
                            side,
                            qty: rng.gen_range(1..50),
                        }));
                    }
                }
            },
            criterion::BatchSize::LargeInput,
        );
    });
}

criterion_group!(
    benches,
    bench_add_limit_no_cross,
    bench_add_limit_single_fill,
    bench_market_order_sweep,
    bench_cancel,
    bench_mixed_workload,
    bench_throughput_1m,
);
criterion_main!(benches);
