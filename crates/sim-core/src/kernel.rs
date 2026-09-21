use std::collections::BinaryHeap;

use rand::rngs::SmallRng;
use rand::SeedableRng;

use crate::agent::{Agent, AgentAction};
use crate::config::SimulationConfig;
use crate::event::{Event, EventPayload, OrderAction};
use crate::exchange::{Exchange, L1Bucket, L1Snapshot, RoutedMessage, TradeRecord};
use crate::latency::LatencyModel;
use crate::types::{MarketSnapshot, Nanos, Symbol};

/// Result of a completed simulation run.
pub struct SimulationResult {
    pub trades: Vec<TradeRecord>,
    pub l1_snapshots: Vec<L1Snapshot>,
    /// L1 bucket aggregates. Empty unless [`RunOptions::l1_bucket_ns`] is set.
    pub l1_buckets: Vec<L1Bucket>,
    pub events_processed: u64,
    pub end_time: Nanos,
}

/// Opt-in output reduction of one run (issue #8).
///
/// The default keeps the behavior of [`Kernel::run`]: every trade and every
/// L1 snapshot stays in memory, ready for export. Each option trades data
/// for memory, so a run that needs the full log must not set it.
#[derive(Debug, Clone)]
pub struct RunOptions {
    /// Keep every trade in [`SimulationResult::trades`]. `false` drops the
    /// trade log and bounds that part of the memory.
    pub keep_trades: bool,
    /// Replace the per-event L1 log with one [`L1Bucket`] per this many
    /// nanoseconds. The aggregation discards the L1 detail below that
    /// scale. `None` keeps the full log.
    pub l1_bucket_ns: Option<Nanos>,
}

impl Default for RunOptions {
    fn default() -> Self {
        Self { keep_trades: true, l1_bucket_ns: None }
    }
}

/// Discrete-event simulation kernel.
pub struct Kernel {
    queue: BinaryHeap<Event>,
    seq: u64,
    rng: SmallRng,
    latency: LatencyModel,
    exchanges: Vec<Exchange>,
    end_time: Nanos,
    // Reusable buffers
    snap_buf: Vec<MarketSnapshot>,
    action_buf: Vec<AgentAction>,
    msg_buf: Vec<RoutedMessage>,
}

impl Kernel {
    /// Run the simulation to completion, with the default output: every
    /// trade and every L1 snapshot stays in memory.
    #[must_use]
    pub fn run(
        config: &SimulationConfig,
        agents: Vec<Box<dyn Agent>>,
    ) -> SimulationResult {
        Self::run_with(config, agents, &RunOptions::default())
    }

    /// Run the simulation to completion, with explicit output options.
    ///
    /// # Panics
    /// Panics when `options.l1_bucket_ns` is `Some(0)`.
    #[must_use]
    pub fn run_with(
        config: &SimulationConfig,
        mut agents: Vec<Box<dyn Agent>>,
        options: &RunOptions,
    ) -> SimulationResult {
        assert!(options.l1_bucket_ns != Some(0), "l1_bucket_ns must be positive");
        let mut k = Self::init(config, agents.len());
        for ex in &mut k.exchanges {
            ex.set_run_options(options.keep_trades, options.l1_bucket_ns);
        }
        k.schedule_lifecycle(config, agents.len());
        let (events_processed, current_time) = k.event_loop(config.end_time, &mut agents);
        k.collect_results(events_processed, current_time)
    }

    fn init(config: &SimulationConfig, num_agents: usize) -> Self {
        let exchanges: Vec<Exchange> = config
            .symbols
            .iter()
            .enumerate()
            .map(|(i, &sym)| {
                #[allow(clippy::cast_possible_truncation)]
                let base = (i as u64 + 1) * 1_000_000_000;
                Exchange::new(sym, base)
            })
            .collect();

        Self {
            queue: BinaryHeap::with_capacity(num_agents * 4),
            seq: 0,
            rng: SmallRng::seed_from_u64(config.seed),
            latency: LatencyModel::new(&config.latency, num_agents),
            snap_buf: Vec::with_capacity(exchanges.len()),
            exchanges,
            end_time: config.end_time,
            action_buf: Vec::with_capacity(4),
            msg_buf: Vec::with_capacity(8),
        }
    }

    fn schedule_lifecycle(&mut self, config: &SimulationConfig, num_agents: usize) {
        if config.no_market_hours {
            // Always-open mode: open immediately, never close.
            for ex in &mut self.exchanges {
                ex.open(config.start_time);
            }
        } else {
            for &sym in &config.symbols {
                self.push(config.start_time, EventPayload::MarketOpen { symbol: sym });
                self.push(config.end_time, EventPayload::MarketClose { symbol: sym });
            }
        }
        for agent_id in 0..num_agents {
            self.push(config.start_time, EventPayload::WakeUp { agent_id });
        }
    }

    fn event_loop(
        &mut self,
        end_time: Nanos,
        agents: &mut [Box<dyn Agent>],
    ) -> (u64, Nanos) {
        let mut count: u64 = 0;
        let mut now = 0;

        while let Some(event) = self.queue.pop() {
            now = event.delivery_time;
            if now > end_time {
                break;
            }
            count += 1;
            self.dispatch(event.payload, now, agents);
        }
        (count, now)
    }

    #[allow(clippy::needless_pass_by_value)]
    fn dispatch(
        &mut self,
        payload: EventPayload,
        now: Nanos,
        agents: &mut [Box<dyn Agent>],
    ) {
        match payload {
            EventPayload::WakeUp { agent_id } => {
                self.snap_buf.clear();
                for ex in &self.exchanges {
                    self.snap_buf.push(ex.snapshot());
                }

                let mut actions = std::mem::take(&mut self.action_buf);
                actions.clear();
                agents[agent_id].wakeup_into(now, agent_id, &self.snap_buf, &mut actions);

                for action in actions.drain(..) {
                    self.route_action(agent_id, action, now);
                }
                self.action_buf = actions;
            }
            EventPayload::OrderArrival { agent_id, symbol, order } => {
                if let Some(idx) = self.find_exchange_idx(symbol) {
                    self.msg_buf.clear();
                    self.exchanges[idx].process_into(agent_id, order, now, &mut self.msg_buf);
                    self.schedule_messages(now);
                }
            }
            EventPayload::ExchangeResponse { agent_id, response } => {
                agents[agent_id].on_exchange_message(now, agent_id, response);
            }
            EventPayload::MarketOpen { symbol } => {
                if let Some(idx) = self.find_exchange_idx(symbol) {
                    self.exchanges[idx].open(now);
                }
            }
            EventPayload::MarketClose { symbol } => {
                if let Some(idx) = self.find_exchange_idx(symbol) {
                    self.msg_buf.clear();
                    self.exchanges[idx].close_into(now, &mut self.msg_buf);
                    self.schedule_messages(now);
                }
            }
        }
    }

    fn schedule_messages(&mut self, now: Nanos) {
        for i in 0..self.msg_buf.len() {
            let m = self.msg_buf[i];
            let lat = self.latency.exchange_to_agent(m.agent_id, &mut self.rng);
            self.push(now + lat, EventPayload::ExchangeResponse {
                agent_id: m.agent_id,
                response: m.message,
            });
        }
    }

    #[allow(clippy::needless_pass_by_value)]
    fn route_action(&mut self, agent_id: usize, action: AgentAction, now: Nanos) {
        match action {
            AgentAction::SubmitOrder { symbol, order } => {
                let lat = self.latency.agent_to_exchange(agent_id, &mut self.rng);
                self.push(now + lat, EventPayload::OrderArrival {
                    agent_id, symbol, order,
                });
            }
            AgentAction::CancelOrder { symbol, order_id } => {
                let lat = self.latency.agent_to_exchange(agent_id, &mut self.rng);
                self.push(now + lat, EventPayload::OrderArrival {
                    agent_id, symbol,
                    order: OrderAction::CancelOrder { order_id },
                });
            }
            AgentAction::ScheduleWakeUp { delay_ns } => {
                let wake = now + delay_ns;
                if wake <= self.end_time {
                    self.push(wake, EventPayload::WakeUp { agent_id });
                }
            }
        }
    }

    fn push(&mut self, delivery_time: Nanos, payload: EventPayload) {
        let seq = self.seq;
        self.seq += 1;
        self.queue.push(Event { delivery_time, seq, payload });
    }

    fn find_exchange_idx(&self, symbol: Symbol) -> Option<usize> {
        self.exchanges.iter().position(|ex| ex.symbol() == symbol)
    }

    fn collect_results(
        mut self,
        events_processed: u64,
        end_time: Nanos,
    ) -> SimulationResult {
        let mut trades = Vec::new();
        let mut l1_snapshots = Vec::new();
        let mut l1_buckets = Vec::new();
        for ex in &mut self.exchanges {
            ex.flush_l1_bucket();
            trades.append(&mut ex.trades);
            l1_snapshots.append(&mut ex.l1_snapshots);
            l1_buckets.append(&mut ex.l1_buckets);
        }
        trades.sort_by_key(|t| t.timestamp);
        l1_snapshots.sort_by_key(|s| s.timestamp);
        l1_buckets.sort_by_key(|b| b.bucket_start);
        SimulationResult { trades, l1_snapshots, l1_buckets, events_processed, end_time }
    }
}
