use rand::rngs::SmallRng;
use rand::SeedableRng;

use crate::agent::{Agent, AgentAction};
use crate::config::SimulationConfig;
use crate::event::{EventPayload, ExchangeMessage, OrderAction};
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

/// One exchange message on its way to an agent.
///
/// Responses to an agent never act on shared state: `on_exchange_message`
/// returns no actions and sees nothing but the agent itself. The kernel
/// therefore keeps them out of the global queue and applies them from a
/// per-agent inbox right before the agent's next wakeup, each with its
/// original delivery time, in exact (`delivery_time`, `seq`) order. The
/// agent sees the history the global queue would have given it, and the
/// queue stays half as busy.
#[derive(Clone, Copy)]
struct InboxEntry {
    delivery_time: Nanos,
    seq: u64,
    message: ExchangeMessage,
}

/// The global event queue: a binary min-heap of ((time, seq), slot) keys
/// with the payloads parked in a slot arena. A sift moves 24 bytes instead
/// of a whole event, and a popped slot is reused by the next push. Keys are
/// unique, so the pop order is the exact (time, seq) order of the former
/// `BinaryHeap<Event>`.
struct EventQueue {
    heap: Vec<HeapEntry>,
    slots: Vec<Option<EventPayload>>,
    free: Vec<u32>,
}

#[derive(Clone, Copy)]
struct HeapEntry {
    /// `(delivery_time << 64) | seq`.
    key: u128,
    slot: u32,
}

impl EventQueue {
    fn with_capacity(capacity: usize) -> Self {
        Self {
            heap: Vec::with_capacity(capacity),
            slots: Vec::with_capacity(capacity),
            free: Vec::new(),
        }
    }

    fn push(&mut self, delivery_time: Nanos, seq: u64, payload: EventPayload) {
        let slot = if let Some(slot) = self.free.pop() {
            self.slots[slot as usize] = Some(payload);
            slot
        } else {
            self.slots.push(Some(payload));
            u32::try_from(self.slots.len() - 1).expect("event queue slots exceed u32")
        };
        let key = (u128::from(delivery_time) << 64) | u128::from(seq);
        self.heap.push(HeapEntry { key, slot });
        self.sift_up(self.heap.len() - 1);
    }

    fn pop(&mut self) -> Option<(Nanos, u64, EventPayload)> {
        let top = *self.heap.first()?;
        let last = self.heap.pop().expect("heap is non-empty");
        if !self.heap.is_empty() {
            self.heap[0] = last;
            self.sift_down(0);
        }
        let payload = self.slots[top.slot as usize].take().expect("slot is filled");
        self.free.push(top.slot);
        #[allow(clippy::cast_possible_truncation)]
        Some(((top.key >> 64) as Nanos, top.key as u64, payload))
    }

    fn sift_up(&mut self, mut child: usize) {
        while child > 0 {
            let parent = (child - 1) / 2;
            if self.heap[parent].key <= self.heap[child].key {
                break;
            }
            self.heap.swap(parent, child);
            child = parent;
        }
    }

    /// Bottom-up sift-down: walk the hole to a leaf on the smaller-child
    /// path (one comparison per level), then sift the displaced entry back
    /// up. The displaced entry is the freshly pushed last leaf, so it
    /// belongs near the bottom and the walk back up is short.
    fn sift_down(&mut self, start: usize) {
        let end = self.heap.len();
        let displaced = self.heap[start];
        let mut hole = start;
        let mut child = 2 * hole + 1;
        while child < end {
            let right = child + 1;
            if right < end && self.heap[right].key < self.heap[child].key {
                child = right;
            }
            self.heap[hole] = self.heap[child];
            hole = child;
            child = 2 * hole + 1;
        }
        self.heap[hole] = displaced;
        self.sift_up_from(start, hole);
    }

    /// Sift the entry at `pos` up, stopping at `limit`.
    fn sift_up_from(&mut self, limit: usize, mut pos: usize) {
        while pos > limit {
            let parent = (pos - 1) / 2;
            if self.heap[parent].key <= self.heap[pos].key {
                break;
            }
            self.heap.swap(parent, pos);
            pos = parent;
        }
    }
}

/// Discrete-event simulation kernel.
pub struct Kernel {
    queue: EventQueue,
    seq: u64,
    rng: SmallRng,
    latency: LatencyModel,
    exchanges: Vec<Exchange>,
    end_time: Nanos,
    /// One response inbox per agent.
    inboxes: Vec<Vec<InboxEntry>>,
    /// Messages applied from the inboxes, part of `events_processed`.
    applied_messages: u64,
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
            queue: EventQueue::with_capacity(num_agents * 4),
            seq: 0,
            rng: SmallRng::seed_from_u64(config.seed),
            latency: LatencyModel::new(&config.latency, num_agents),
            snap_buf: Vec::with_capacity(exchanges.len()),
            exchanges,
            end_time: config.end_time,
            inboxes: vec![Vec::new(); num_agents],
            applied_messages: 0,
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
        let mut last_time = 0;
        let mut beyond: Option<(Nanos, u64)> = None;

        while let Some((time, seq, payload)) = self.queue.pop() {
            if time > end_time {
                beyond = Some((time, seq));
                break;
            }
            last_time = time;
            count += 1;
            self.dispatch(payload, time, seq, agents);
        }
        let final_time = self.drain_inboxes(end_time, last_time, beyond, agents);
        (count + self.applied_messages, final_time)
    }

    /// Apply every response due by `end_time` that no wakeup consumed, and
    /// give the time the run ends on: the first pending moment past
    /// `end_time` when one exists, else the last moment that happened.
    /// Both match what the global queue reported before inbox delivery.
    fn drain_inboxes(
        &mut self,
        end_time: Nanos,
        last_time: Nanos,
        beyond: Option<(Nanos, u64)>,
        agents: &mut [Box<dyn Agent>],
    ) -> Nanos {
        let mut last_applied = last_time;
        let mut first_beyond = beyond;
        for (agent_id, inbox) in self.inboxes.iter_mut().enumerate() {
            inbox.sort_unstable_by_key(|e| (e.delivery_time, e.seq));
            let due = inbox.partition_point(|e| e.delivery_time <= end_time);
            for entry in inbox.drain(..due) {
                self.applied_messages += 1;
                last_applied = last_applied.max(entry.delivery_time);
                agents[agent_id].on_exchange_message(entry.delivery_time, agent_id, entry.message);
            }
            if let Some(entry) = inbox.first() {
                let key = (entry.delivery_time, entry.seq);
                if first_beyond.is_none_or(|b| key < b) {
                    first_beyond = Some(key);
                }
            }
        }
        first_beyond.map_or(last_applied, |(time, _)| time)
    }

    /// Apply the responses of `agent_id` that are due strictly before the
    /// wakeup key (`now`, `wake_seq`), in (`delivery_time`, `seq`) order.
    fn deliver_due(
        &mut self,
        agent_id: usize,
        now: Nanos,
        wake_seq: u64,
        agents: &mut [Box<dyn Agent>],
    ) {
        let inbox = &mut self.inboxes[agent_id];
        if inbox.is_empty() {
            return;
        }
        inbox.sort_unstable_by_key(|e| (e.delivery_time, e.seq));
        let due = inbox.partition_point(|e| (e.delivery_time, e.seq) < (now, wake_seq));
        for entry in inbox.drain(..due) {
            self.applied_messages += 1;
            agents[agent_id].on_exchange_message(entry.delivery_time, agent_id, entry.message);
        }
    }

    #[allow(clippy::needless_pass_by_value)]
    fn dispatch(
        &mut self,
        payload: EventPayload,
        now: Nanos,
        seq: u64,
        agents: &mut [Box<dyn Agent>],
    ) {
        match payload {
            EventPayload::WakeUp { agent_id } => {
                self.deliver_due(agent_id, now, seq, agents);
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
            let seq = self.seq;
            self.seq += 1;
            self.inboxes[m.agent_id].push(InboxEntry {
                delivery_time: now + lat,
                seq,
                message: m.message,
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
        self.queue.push(delivery_time, seq, payload);
    }

    fn find_exchange_idx(&self, symbol: Symbol) -> Option<usize> {
        self.exchanges.iter().position(|ex| ex.symbol() == symbol)
    }

    fn collect_results(
        mut self,
        events_processed: u64,
        end_time: Nanos,
    ) -> SimulationResult {
        for ex in &mut self.exchanges {
            ex.flush_l1_bucket();
        }
        let exchanges = &mut self.exchanges;
        let trades = merge_by_time(exchanges, |ex| std::mem::take(&mut ex.trades), |t| t.timestamp);
        let l1_snapshots =
            merge_by_time(exchanges, |ex| std::mem::take(&mut ex.l1_snapshots), |s| s.timestamp);
        let l1_buckets =
            merge_by_time(exchanges, |ex| std::mem::take(&mut ex.l1_buckets), |b| b.bucket_start);
        SimulationResult { trades, l1_snapshots, l1_buckets, events_processed, end_time }
    }
}

/// Merge the per-exchange logs into one timestamp-ordered log.
///
/// Each exchange appends its records in event order, so its log is already
/// sorted by time. A k-way merge that prefers the earliest exchange on equal
/// keys gives exactly what the former concatenate-and-stable-sort gave, in
/// linear time.
fn merge_by_time<T: Copy, K: Ord>(
    exchanges: &mut [Exchange],
    take: impl Fn(&mut Exchange) -> Vec<T>,
    key: impl Fn(&T) -> K,
) -> Vec<T> {
    let logs: Vec<Vec<T>> = exchanges.iter_mut().map(take).collect();
    let total = logs.iter().map(Vec::len).sum();
    let mut merged = Vec::with_capacity(total);
    let mut cursors = vec![0_usize; logs.len()];
    loop {
        let mut best: Option<(usize, K)> = None;
        for (i, log) in logs.iter().enumerate() {
            if let Some(record) = log.get(cursors[i]) {
                let k = key(record);
                if best.as_ref().is_none_or(|(_, bk)| k < *bk) {
                    best = Some((i, k));
                }
            }
        }
        let Some((i, _)) = best else { break };
        merged.push(logs[i][cursors[i]]);
        cursors[i] += 1;
    }
    merged
}

#[cfg(test)]
mod merge_tests {
    use super::merge_by_time;
    use crate::exchange::Exchange;

    /// The merge equals concatenate-and-stable-sort, ties included.
    #[test]
    fn merge_matches_stable_sort_with_ties() {
        let mut exchanges: Vec<Exchange> =
            (0..3).map(|i| Exchange::new(i, u64::from(i) * 10)).collect();
        let logs: [&[u64]; 3] = [&[1, 5, 5, 9], &[0, 5, 9, 9, 12], &[5, 5]];
        for (ex, log) in exchanges.iter_mut().zip(logs) {
            for &t in log {
                ex.trades.push(crate::exchange::TradeRecord {
                    timestamp: t,
                    symbol: ex.symbol(),
                    price: 1,
                    qty: 1,
                    aggressor_side: cda_engine::Side::Bid,
                    maker_order_id: 0,
                    taker_order_id: 0,
                });
            }
        }
        let mut expected: Vec<_> =
            exchanges.iter().flat_map(|ex| ex.trades.clone()).collect();
        expected.sort_by_key(|t| t.timestamp);

        let merged =
            merge_by_time(&mut exchanges, |ex| std::mem::take(&mut ex.trades), |t| t.timestamp);
        let pairs: Vec<_> = merged.iter().map(|t| (t.timestamp, t.symbol)).collect();
        let expected_pairs: Vec<_> = expected.iter().map(|t| (t.timestamp, t.symbol)).collect();
        assert_eq!(pairs, expected_pairs);
    }
}

#[cfg(test)]
mod queue_tests {
    use std::collections::BinaryHeap;

    use super::EventQueue;
    use crate::event::{Event, EventPayload};

    /// The compact queue pops in the exact order of `BinaryHeap<Event>`,
    /// duplicate times included.
    #[test]
    fn compact_queue_matches_the_reference_heap() {
        let mut fast = EventQueue::with_capacity(8);
        let mut reference = BinaryHeap::new();
        let mut state: u64 = 20_260_921;
        let mut roll = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        let mut seq = 0;
        for _ in 0..10_000 {
            if roll() % 3 != 0 {
                let time = roll() % 64;
                let payload = EventPayload::WakeUp { agent_id: usize::try_from(seq).unwrap() };
                fast.push(time, seq, payload);
                reference.push(Event { delivery_time: time, seq, payload: EventPayload::WakeUp { agent_id: usize::try_from(seq).unwrap() } });
                seq += 1;
            } else {
                let got = fast.pop().map(|(time, s, _)| (time, s));
                let want = reference.pop().map(|e| (e.delivery_time, e.seq));
                assert_eq!(got, want);
            }
        }
        while let Some(want) = reference.pop() {
            let got = fast.pop().map(|(time, s, _)| (time, s));
            assert_eq!(got, Some((want.delivery_time, want.seq)));
        }
        assert!(fast.pop().is_none());
    }
}
