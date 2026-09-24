use cda_engine::fasthash::{FxHashMap, FxHashSet};
use cda_engine::{Fill, LimitOrder, MarketOrder, OrderBook, OrderStatus, Side};

use crate::event::{ExchangeMessage, OrderAction};
use crate::types::{AgentId, MarketSnapshot, Nanos, Symbol};

/// Record of a single trade, collected for Parquet output.
#[derive(Debug, Clone, Copy)]
pub struct TradeRecord {
    pub timestamp: Nanos,
    pub symbol: Symbol,
    pub price: i64,
    pub qty: u64,
    pub aggressor_side: Side,
    pub maker_order_id: u64,
    pub taker_order_id: u64,
}

/// L1 best-bid/ask snapshot, recorded on BBO changes.
#[derive(Debug, Clone, Copy)]
pub struct L1Snapshot {
    pub timestamp: Nanos,
    pub symbol: Symbol,
    pub bid_price: i64,
    pub ask_price: i64,
    pub bid_volume: u64,
    pub ask_volume: u64,
    pub last_trade_price: i64,
}

/// One k-nanosecond aggregate of the L1 log (issue #8).
///
/// The price fields read the quoted snapshots only, the snapshots with both
/// sides present; `0` means the bucket saw no quote yet. `volume` and
/// `quote_volume` sum over every snapshot of the bucket, exactly as the
/// kline export sums the full log.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct L1Bucket {
    pub bucket_start: Nanos,
    pub symbol: Symbol,
    pub first_bid: i64,
    pub min_bid: i64,
    pub last_bid: i64,
    pub first_ask: i64,
    pub max_ask: i64,
    pub last_ask: i64,
    pub volume: u64,
    pub quote_volume: u128,
}

impl L1Bucket {
    /// An empty bucket.
    #[must_use]
    pub fn new(bucket_start: Nanos, symbol: Symbol) -> Self {
        Self {
            bucket_start,
            symbol,
            first_bid: 0,
            min_bid: 0,
            last_bid: 0,
            first_ask: 0,
            max_ask: 0,
            last_ask: 0,
            volume: 0,
            quote_volume: 0,
        }
    }

    /// Fold one L1 state into the bucket.
    pub fn absorb(&mut self, bid_price: i64, ask_price: i64, bid_volume: u64, ask_volume: u64) {
        if bid_price > 0 && ask_price > 0 {
            if self.last_bid == 0 {
                self.first_bid = bid_price;
                self.min_bid = bid_price;
                self.first_ask = ask_price;
                self.max_ask = ask_price;
            }
            self.min_bid = self.min_bid.min(bid_price);
            self.max_ask = self.max_ask.max(ask_price);
            self.last_bid = bid_price;
            self.last_ask = ask_price;
        }
        self.volume += bid_volume + ask_volume;
        self.quote_volume += u128::try_from(bid_price.max(0)).unwrap_or(0) * u128::from(bid_volume)
            + u128::try_from(ask_price.max(0)).unwrap_or(0) * u128::from(ask_volume);
    }
}

/// A message routed to a specific agent.
#[derive(Debug, Clone, Copy)]
pub struct RoutedMessage {
    pub agent_id: AgentId,
    pub message: ExchangeMessage,
}

/// Interval diagnostics for one agent on each exchange.
///
/// Quantities use lots. Notionals use tick-lots. Times use simulation nanoseconds.
/// Configure this before the first order. The interval must be positive.
#[derive(Debug, Clone)]
pub struct FlowOptions {
    /// Agent whose external fills and submitted orders the recorder tracks.
    pub agent_id: AgentId,
    /// Width of intervals aligned to simulation time zero, in nanoseconds.
    pub interval_ns: Nanos,
    /// Price-response delays, in nanoseconds. Their order defines all response vectors.
    pub horizons_ns: Vec<Nanos>,
    /// Increasing boundaries for dimensionless submitted-notional/depth ratios.
    /// Bins are `[0, e0)`, `[e0, e1)`, ..., `[e_last, infinity)`.
    pub ratio_edges: Vec<f64>,
}

/// Flow on one symbol in one interval, with separate external and self matches.
///
/// Each external market match counts once. Maker and taker subsets contain only
/// the watched agent's external fills. Self matches never enter cost or response support.
/// Quantities use lots. Notionals and weighted numerators use tick-lots.
/// Divide a numerator by its matching eligible base. An empty base gives no estimate.
///
/// Shortfall is `s * (price / pre_trade_mid - 1)`, with `s = +1` for buys and `-1` for sells.
/// Response is `s * (mid_at_horizon / pre_trade_mid - 1)`.
/// A valid mid is positive and two-sided. Response vectors follow [`FlowOptions::horizons_ns`].
/// Responses use the book after all quote changes at the horizon, in event order.
/// Missing mids and horizons past the run end enter excluded support.
///
/// An order can be eligible or filled in several intervals. Use [`FlowTotals`] for distinct run totals.
#[derive(Debug, Clone, PartialEq)]
pub struct FlowBucket {
    /// Inclusive interval start in simulation nanoseconds.
    pub start: Nanos,
    /// Exchange symbol.
    pub symbol: Symbol,
    /// External matches across all agents, once per match.
    pub market_trades: u64,
    /// External matched quantity across all agents, in lots.
    pub market_qty: u64,
    /// External matched notional across all agents, in tick-lots.
    pub market_notional: u128,
    /// Same-owner matches across all agents, once per match.
    pub market_self_trades: u64,
    /// Same-owner matched quantity across all agents, in lots.
    pub market_self_qty: u64,
    /// Same-owner matched notional across all agents, once per match.
    pub market_self_notional: u128,
    /// Same-owner matches of the watched agent, once per match.
    pub self_trades: u64,
    /// Same-owner matched quantity of the watched agent, in lots.
    pub self_qty: u64,
    /// Same-owner matched notional of the watched agent, once per match.
    pub self_notional: u128,
    /// Watched taker quantity against other owners, in lots.
    pub taker_qty: u64,
    /// Watched taker notional against other owners, in tick-lots.
    pub taker_notional: u128,
    /// Watched maker quantity against other owners, in lots.
    pub maker_qty: u64,
    /// Watched maker notional against other owners, in tick-lots.
    pub maker_notional: u128,
    /// Watched orders accepted at exchange arrival during this interval.
    pub orders: u64,
    /// Watched new orders rejected at exchange arrival during this interval.
    pub rejected_orders: u64,
    /// Distinct accepted orders carried into or accepted during this interval.
    /// A carried order canceled in the interval remains in this cohort.
    pub eligible_orders: u64,
    /// Distinct eligible orders with an external fill during this interval.
    pub filled_orders: u64,
    /// External matches that fill a watched order, including repeated partial fills.
    pub fill_events: u64,
    /// Accepted watched order quantity during this interval, in lots.
    pub submitted_qty: u64,
    /// Submitted notional per finite depth-ratio bin, in tick-lots.
    /// Limits use limit price times quantity. Market orders use the pre-submit mid.
    /// Depth includes external live opposing displayed orders through the limit.
    pub ratio_notional: Vec<f64>,
    /// Accepted order count per finite depth-ratio bin.
    pub ratio_count: Vec<u64>,
    /// Valued orders with zero external executable depth.
    pub zero_depth_count: u64,
    /// Quantity of those zero-depth orders, in lots.
    pub zero_depth_qty: u64,
    /// Submitted notional of those zero-depth orders, in tick-lots.
    pub zero_depth_notional: f64,
    /// Market orders without a valid pre-submit valuation mid, separate from zero depth.
    pub unvalued_count: u64,
    /// Quantity of those unvalued orders, in lots.
    pub unvalued_qty: u64,
    /// Sum of external taker notional times signed shortfall.
    pub shortfall_taker: f64,
    /// External taker notional with a valid pre-trade mid.
    pub shortfall_taker_base: u128,
    /// Sum of external maker notional times signed shortfall.
    pub shortfall_maker: f64,
    /// External maker notional with a valid pre-trade mid.
    pub shortfall_maker_base: u128,
    /// External watched fills without a valid pre-trade mid.
    pub shortfall_excluded_count: u64,
    /// External watched notional without a valid pre-trade mid.
    pub shortfall_excluded_notional: u128,
    /// Sum of eligible fill notional times signed response, by horizon and original fill interval.
    pub response_num: Vec<f64>,
    /// Eligible external fill notional at each horizon.
    pub response_den: Vec<f64>,
    /// Eligible external fill count at each horizon.
    pub response_count: Vec<u64>,
    /// External fill notional without both valid mids or a completed horizon.
    pub response_excluded_notional: Vec<f64>,
    /// External fill count without both valid mids or a completed horizon.
    pub response_excluded_count: Vec<u64>,
}

impl FlowBucket {
    fn new(start: Nanos, symbol: Symbol, bins: usize, horizons: usize) -> Self {
        Self {
            start,
            symbol,
            market_trades: 0,
            market_qty: 0,
            market_notional: 0,
            market_self_trades: 0,
            market_self_qty: 0,
            market_self_notional: 0,
            self_trades: 0,
            self_qty: 0,
            self_notional: 0,
            taker_qty: 0,
            taker_notional: 0,
            maker_qty: 0,
            maker_notional: 0,
            orders: 0,
            rejected_orders: 0,
            eligible_orders: 0,
            filled_orders: 0,
            fill_events: 0,
            submitted_qty: 0,
            ratio_notional: vec![0.0; bins],
            ratio_count: vec![0; bins],
            zero_depth_count: 0,
            zero_depth_qty: 0,
            zero_depth_notional: 0.0,
            unvalued_count: 0,
            unvalued_qty: 0,
            shortfall_taker: 0.0,
            shortfall_taker_base: 0,
            shortfall_maker: 0.0,
            shortfall_maker_base: 0,
            shortfall_excluded_count: 0,
            shortfall_excluded_notional: 0,
            response_num: vec![0.0; horizons],
            response_den: vec![0.0; horizons],
            response_count: vec![0; horizons],
            response_excluded_notional: vec![0.0; horizons],
            response_excluded_count: vec![0; horizons],
        }
    }
}

/// Distinct watched-order totals for one symbol over the whole rollout.
///
/// `filled_orders / accepted_orders` is the rollout fill rate when the denominator is positive.
/// Interval cohort counts can repeat orders, so their sums cannot replace these totals.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FlowTotals {
    /// Exchange symbol.
    pub symbol: Symbol,
    /// Distinct watched orders accepted during the run.
    pub accepted_orders: u64,
    /// Watched new orders rejected during the run.
    pub rejected_orders: u64,
    /// Distinct watched orders with at least one external fill.
    pub filled_orders: u64,
    /// External matches that fill a watched order.
    pub fill_events: u64,
}

#[derive(Debug, Clone, Copy)]
struct PendingResponse {
    due: Nanos,
    bucket: usize,
    signed_notional: f64,
    mid0: f64,
}

/// Aggregate matches online and retain pending responses until each horizon resolves.
/// Keep live and filled order identities to form interval cohorts and distinct run totals.
struct FlowRecorder {
    options: FlowOptions,
    symbol: Symbol,
    buckets: Vec<FlowBucket>,
    pending: Vec<std::collections::VecDeque<PendingResponse>>,
    live: FxHashSet<u64>,
    filled_in_interval: FxHashSet<u64>,
    filled_in_previous_interval: FxHashSet<u64>,
    ever_filled: FxHashSet<u64>,
    totals: FlowTotals,
    current_interval: usize,
    prevailing_mid: Option<f64>,
}

impl FlowRecorder {
    fn new(options: FlowOptions, symbol: Symbol) -> Self {
        let horizons = options.horizons_ns.len();
        Self {
            options,
            symbol,
            buckets: Vec::new(),
            pending: vec![std::collections::VecDeque::new(); horizons],
            live: FxHashSet::default(),
            filled_in_interval: FxHashSet::default(),
            filled_in_previous_interval: FxHashSet::default(),
            ever_filled: FxHashSet::default(),
            totals: FlowTotals {
                symbol,
                accepted_orders: 0,
                rejected_orders: 0,
                filled_orders: 0,
                fill_events: 0,
            },
            current_interval: 0,
            prevailing_mid: None,
        }
    }

    #[allow(clippy::cast_possible_truncation)]
    fn bucket_index(&self, time: Nanos) -> usize {
        (time / self.options.interval_ns) as usize
    }

    fn bucket_at(&mut self, index: usize) -> &mut FlowBucket {
        let (bins, horizons) = (
            self.options.ratio_edges.len() + 1,
            self.options.horizons_ns.len(),
        );
        while self.buckets.len() <= index {
            let start = self.buckets.len() as Nanos * self.options.interval_ns;
            self.buckets
                .push(FlowBucket::new(start, self.symbol, bins, horizons));
        }
        &mut self.buckets[index]
    }

    fn roll_to(&mut self, index: usize) {
        while self.current_interval < index {
            self.current_interval += 1;
            let carried = self.live.len() as u64;
            let interval = self.current_interval;
            self.bucket_at(interval).eligible_orders = carried;
            self.filled_in_previous_interval = std::mem::take(&mut self.filled_in_interval);
        }
    }

    fn on_accept(&mut self, time: Nanos, agent_id: AgentId, order_id: u64, qty: u64) {
        if agent_id != self.options.agent_id {
            return;
        }
        let index = self.bucket_index(time);
        self.roll_to(index);
        let bucket = self.bucket_at(index);
        bucket.orders += 1;
        bucket.eligible_orders += 1;
        bucket.submitted_qty += qty;
        self.live.insert(order_id);
        self.totals.accepted_orders += 1;
    }

    fn on_reject(&mut self, time: Nanos, agent_id: AgentId) {
        if agent_id != self.options.agent_id {
            return;
        }
        let index = self.bucket_index(time);
        self.roll_to(index);
        self.bucket_at(index).rejected_orders += 1;
        self.totals.rejected_orders += 1;
    }

    fn on_depth(
        &mut self,
        time: Nanos,
        qty: u64,
        order_notional: Option<f64>,
        depth_notional: u128,
    ) {
        let index = self.bucket_index(time);
        self.roll_to(index);
        #[allow(clippy::cast_precision_loss)]
        let edges_below = order_notional
            .filter(|_| depth_notional > 0)
            .map(|notional| {
                let ratio = notional / depth_notional as f64;
                self.options
                    .ratio_edges
                    .partition_point(|&edge| edge <= ratio)
            });
        let bucket = self.bucket_at(index);
        match (order_notional, edges_below) {
            (Some(notional), Some(bin)) => {
                bucket.ratio_notional[bin] += notional;
                bucket.ratio_count[bin] += 1;
            }
            (Some(notional), None) => {
                bucket.zero_depth_count += 1;
                bucket.zero_depth_qty += qty;
                bucket.zero_depth_notional += notional;
            }
            (None, _) => {
                bucket.unvalued_count += 1;
                bucket.unvalued_qty += qty;
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn on_fill(
        &mut self,
        time: Nanos,
        taker_agent: AgentId,
        taker_order_id: u64,
        maker_agent: Option<AgentId>,
        maker_order_id: u64,
        taker_side: Side,
        price: i64,
        qty: u64,
        mid0: Option<f64>,
    ) {
        self.resolve_due(time);
        let index = self.bucket_index(time);
        self.roll_to(index);
        let watched = self.options.agent_id;
        let notional = u128::from(price.unsigned_abs()) * u128::from(qty);
        if maker_agent == Some(taker_agent) {
            let watched_self = taker_agent == watched;
            let bucket = self.bucket_at(index);
            bucket.market_self_trades += 1;
            bucket.market_self_qty += qty;
            bucket.market_self_notional += notional;
            if watched_self {
                bucket.self_trades += 1;
                bucket.self_qty += qty;
                bucket.self_notional += notional;
            }
            return;
        }
        {
            let bucket = self.bucket_at(index);
            bucket.market_trades += 1;
            bucket.market_qty += qty;
            bucket.market_notional += notional;
        }
        let taker_is_watched = taker_agent == watched;
        let maker_is_watched = maker_agent == Some(watched);
        if !taker_is_watched && !maker_is_watched {
            return;
        }
        let order_id = if maker_is_watched {
            maker_order_id
        } else {
            taker_order_id
        };
        let newly_filled_here = self.filled_in_interval.insert(order_id);
        let newly_filled_ever = self.ever_filled.insert(order_id);
        self.totals.fill_events += 1;
        if newly_filled_ever {
            self.totals.filled_orders += 1;
        }
        let side = if taker_is_watched {
            taker_side
        } else {
            match taker_side {
                Side::Bid => Side::Ask,
                Side::Ask => Side::Bid,
            }
        };
        let sign = match side {
            Side::Bid => 1.0,
            Side::Ask => -1.0,
        };
        #[allow(clippy::cast_precision_loss)]
        let notional_f64 = notional as f64;
        let valid_mid0 = mid0.filter(|&m| m.is_finite() && m > 0.0);
        let horizons = self.options.horizons_ns.len();
        {
            let bucket = self.bucket_at(index);
            bucket.fill_events += 1;
            if newly_filled_here {
                bucket.filled_orders += 1;
            }
            if taker_is_watched {
                bucket.taker_qty += qty;
                bucket.taker_notional += notional;
            } else {
                bucket.maker_qty += qty;
                bucket.maker_notional += notional;
            }
            if let Some(mid0) = valid_mid0 {
                #[allow(clippy::cast_precision_loss)]
                let cost = sign * (price as f64 / mid0 - 1.0);
                if taker_is_watched {
                    bucket.shortfall_taker += cost * notional_f64;
                    bucket.shortfall_taker_base += notional;
                } else {
                    bucket.shortfall_maker += cost * notional_f64;
                    bucket.shortfall_maker_base += notional;
                }
            } else {
                bucket.shortfall_excluded_count += 1;
                bucket.shortfall_excluded_notional += notional;
                for horizon in 0..horizons {
                    bucket.response_excluded_notional[horizon] += notional_f64;
                    bucket.response_excluded_count[horizon] += 1;
                }
            }
        }
        if let Some(mid0) = valid_mid0 {
            for (horizon, &delay) in self.options.horizons_ns.iter().enumerate() {
                self.pending[horizon].push_back(PendingResponse {
                    due: time + delay,
                    bucket: index,
                    signed_notional: sign * notional_f64,
                    mid0,
                });
            }
        }
    }

    fn on_order_gone(&mut self, time: Nanos, order_id: u64) {
        let index = self.bucket_index(time);
        self.roll_to(index);
        self.live.remove(&order_id);
        self.ever_filled.remove(&order_id);
    }

    fn resolve_due(&mut self, before: Nanos) {
        let mid = self.prevailing_mid;
        for horizon in 0..self.pending.len() {
            while let Some(&entry) = self.pending[horizon].front() {
                if entry.due >= before {
                    break;
                }
                self.pending[horizon].pop_front();
                let bucket = &mut self.buckets[entry.bucket];
                if let Some(mid) = mid {
                    bucket.response_num[horizon] +=
                        entry.signed_notional * (mid / entry.mid0 - 1.0);
                    bucket.response_den[horizon] += entry.signed_notional.abs();
                    bucket.response_count[horizon] += 1;
                } else {
                    bucket.response_excluded_notional[horizon] += entry.signed_notional.abs();
                    bucket.response_excluded_count[horizon] += 1;
                }
            }
        }
    }

    #[allow(clippy::cast_precision_loss)]
    fn on_quote(&mut self, time: Nanos, bid_price: i64, ask_price: i64) {
        self.resolve_due(time);
        self.prevailing_mid = if bid_price > 0 && ask_price > 0 {
            Some(f64::midpoint(bid_price as f64, ask_price as f64))
        } else {
            None
        };
    }

    fn flush(&mut self, end_time: Nanos) -> (Vec<FlowBucket>, FlowTotals) {
        self.resolve_due(end_time.saturating_add(1));
        for horizon in 0..self.pending.len() {
            while let Some(entry) = self.pending[horizon].pop_front() {
                let bucket = &mut self.buckets[entry.bucket];
                bucket.response_excluded_notional[horizon] += entry.signed_notional.abs();
                bucket.response_excluded_count[horizon] += 1;
            }
        }
        let last_index = self.bucket_index(end_time.saturating_sub(1));
        self.bucket_at(last_index);
        self.roll_to(last_index);
        while self.buckets.len() > last_index + 1 {
            let terminal = self.buckets.pop().expect("terminal bucket");
            let repeat_filled = self
                .filled_in_interval
                .iter()
                .filter(|id| self.filled_in_previous_interval.contains(id))
                .count() as u64;
            self.filled_in_interval.clear();
            let last = &mut self.buckets[last_index];
            last.eligible_orders += terminal.orders;
            last.filled_orders += terminal.filled_orders - repeat_filled;
            add_terminal_flows(last, &terminal);
        }
        (std::mem::take(&mut self.buckets), self.totals)
    }
}

fn add_terminal_flows(last: &mut FlowBucket, terminal: &FlowBucket) {
    last.market_trades += terminal.market_trades;
    last.market_qty += terminal.market_qty;
    last.market_notional += terminal.market_notional;
    last.market_self_trades += terminal.market_self_trades;
    last.market_self_qty += terminal.market_self_qty;
    last.market_self_notional += terminal.market_self_notional;
    last.self_trades += terminal.self_trades;
    last.self_qty += terminal.self_qty;
    last.self_notional += terminal.self_notional;
    last.taker_qty += terminal.taker_qty;
    last.taker_notional += terminal.taker_notional;
    last.maker_qty += terminal.maker_qty;
    last.maker_notional += terminal.maker_notional;
    last.orders += terminal.orders;
    last.rejected_orders += terminal.rejected_orders;
    last.fill_events += terminal.fill_events;
    last.submitted_qty += terminal.submitted_qty;
    for (total, part) in last.ratio_notional.iter_mut().zip(&terminal.ratio_notional) {
        *total += part;
    }
    for (total, part) in last.ratio_count.iter_mut().zip(&terminal.ratio_count) {
        *total += part;
    }
    last.zero_depth_count += terminal.zero_depth_count;
    last.zero_depth_qty += terminal.zero_depth_qty;
    last.zero_depth_notional += terminal.zero_depth_notional;
    last.unvalued_count += terminal.unvalued_count;
    last.unvalued_qty += terminal.unvalued_qty;
    last.shortfall_taker += terminal.shortfall_taker;
    last.shortfall_taker_base += terminal.shortfall_taker_base;
    last.shortfall_maker += terminal.shortfall_maker;
    last.shortfall_maker_base += terminal.shortfall_maker_base;
    last.shortfall_excluded_count += terminal.shortfall_excluded_count;
    last.shortfall_excluded_notional += terminal.shortfall_excluded_notional;
    for horizon in 0..last.response_num.len() {
        last.response_num[horizon] += terminal.response_num[horizon];
        last.response_den[horizon] += terminal.response_den[horizon];
        last.response_count[horizon] += terminal.response_count[horizon];
        last.response_excluded_notional[horizon] += terminal.response_excluded_notional[horizon];
        last.response_excluded_count[horizon] += terminal.response_excluded_count[horizon];
    }
}

/// Identity of a just-created order, as echoed to its owner.
#[derive(Clone, Copy)]
struct NewOrder {
    id: u64,
    user_id: u64,
    side: Side,
    price: i64,
    qty: u64,
}

struct OrderInfo {
    agent_id: AgentId,
    side: Side,
    price: i64,
    remaining_qty: u64,
    /// Owner's tag, echoed on every message about the order (0 = unset).
    user_id: u64,
}

/// Per-symbol exchange wrapping a [`cda_engine::OrderBook`].
pub struct Exchange {
    sym: Symbol,
    book: OrderBook,
    is_open: bool,
    next_order_id: u64,
    resting: FxHashMap<u64, OrderInfo>,
    last_trade_price: Option<i64>,
    last_trade_time: Option<Nanos>,
    /// Accumulated trade records (drained at end of simulation).
    pub trades: Vec<TradeRecord>,
    /// Accumulated L1 snapshots (drained at end of simulation).
    pub l1_snapshots: Vec<L1Snapshot>,
    /// Accumulated L1 buckets (drained at end of simulation; bucket mode only).
    pub l1_buckets: Vec<L1Bucket>,
    keep_trades: bool,
    l1_bucket_ns: Option<Nanos>,
    open_bucket: Option<L1Bucket>,
    flow: Option<FlowRecorder>,
    /// Reusable fill buffer of [`Self::process_into`].
    fill_buf: Vec<Fill>,
}

impl Exchange {
    /// Create a new closed exchange for the given symbol.
    #[must_use]
    pub fn new(symbol: Symbol, starting_order_id: u64) -> Self {
        Self {
            sym: symbol,
            book: OrderBook::new(),
            is_open: false,
            next_order_id: starting_order_id,
            resting: FxHashMap::default(),
            last_trade_price: None,
            last_trade_time: None,
            trades: Vec::with_capacity(1 << 16),
            l1_snapshots: Vec::with_capacity(1 << 16),
            l1_buckets: Vec::new(),
            keep_trades: true,
            l1_bucket_ns: None,
            open_bucket: None,
            flow: None,
            fill_buf: Vec::new(),
        }
    }

    /// Select the output of the run. The defaults keep every trade and the
    /// per-event L1 log, as [`crate::kernel::RunOptions`] documents.
    pub fn set_run_options(&mut self, keep_trades: bool, l1_bucket_ns: Option<Nanos>) {
        self.keep_trades = keep_trades;
        self.l1_bucket_ns = l1_bucket_ns;
    }

    /// Set diagnostics before trading. This discards any previous recorder and its data.
    pub fn set_flow_options(&mut self, options: Option<FlowOptions>) {
        self.flow = options.map(|options| FlowRecorder::new(options, self.sym));
    }

    /// Resolve responses through `end_time`, exclude later horizons, and drain the interval rows.
    /// Events at `end_time` enter the final interval without repeating filled orders.
    /// Only newly accepted terminal orders increase that interval's eligible cohort.
    /// Call once after the last event. `None` means flow recording was disabled.
    #[must_use]
    pub fn flush_flow(&mut self, end_time: Nanos) -> Option<(Vec<FlowBucket>, FlowTotals)> {
        self.flow.as_mut().map(|flow| flow.flush(end_time))
    }

    /// Push the open bucket, if any, into `l1_buckets`.
    pub fn flush_l1_bucket(&mut self) {
        if let Some(done) = self.open_bucket.take() {
            self.l1_buckets.push(done);
        }
    }

    /// The symbol this exchange serves.
    #[must_use]
    pub fn symbol(&self) -> Symbol {
        self.sym
    }

    /// Current top-of-book snapshot.
    #[must_use]
    pub fn snapshot(&self) -> MarketSnapshot {
        let best_bid = self.book.best_bid_level();
        let best_ask = self.book.best_ask_level();
        MarketSnapshot {
            best_bid,
            best_ask,
            last_trade_price: self.last_trade_price,
            last_trade_time: self.last_trade_time,
        }
    }

    /// Open the market for trading.
    pub fn open(&mut self, time: Nanos) {
        self.is_open = true;
        self.record_l1(time);
    }

    /// Close the market, cancelling all resting orders.
    pub fn close_into(&mut self, time: Nanos, out: &mut Vec<RoutedMessage>) {
        self.is_open = false;
        for (&oid, info) in &self.resting {
            self.book.cancel_order(oid);
            if let Some(flow) = self.flow.as_mut() {
                flow.on_order_gone(time, oid);
            }
            out.push(RoutedMessage {
                agent_id: info.agent_id,
                message: ExchangeMessage::OrderCancelled {
                    order_id: oid,
                    user_id: info.user_id,
                    symbol: self.sym,
                },
            });
        }
        self.resting.clear();
        self.record_l1(time);
    }

    /// Process an order action from an agent. Writes messages into `out`.
    #[allow(clippy::too_many_lines)]
    pub fn process_into(
        &mut self,
        agent_id: AgentId,
        action: OrderAction,
        time: Nanos,
        out: &mut Vec<RoutedMessage>,
    ) {
        if !self.is_open {
            // A new order never receives an id (order_id 0); its echoed
            // user_id still identifies it. A rejected cancel names its target.
            let (order_id, user_id) = match action {
                OrderAction::CancelOrder { order_id } => (order_id, 0),
                OrderAction::NewLimitOrder { user_id, .. }
                | OrderAction::NewMarketOrder { user_id, .. } => (0, user_id),
            };
            if !matches!(action, OrderAction::CancelOrder { .. }) {
                if let Some(flow) = self.flow.as_mut() {
                    flow.on_reject(time, agent_id);
                }
            }
            out.push(RoutedMessage {
                agent_id,
                message: ExchangeMessage::OrderRejected {
                    order_id,
                    user_id,
                    symbol: self.sym,
                },
            });
            return;
        }

        let bbo_before = (self.book.best_bid(), self.book.best_ask());
        #[allow(clippy::cast_precision_loss)]
        let mid0 = match bbo_before {
            (Some(bid), Some(ask)) if bid > 0 && ask > 0 => {
                Some(f64::midpoint(bid as f64, ask as f64))
            }
            _ => None,
        };

        match action {
            OrderAction::NewLimitOrder {
                side,
                price,
                qty,
                user_id,
            } => {
                let order = NewOrder {
                    id: self.alloc_id(),
                    user_id,
                    side,
                    price,
                    qty,
                };
                self.record_submission(agent_id, time, side, Some(price), qty, order.id, mid0);
                self.push_accepted(agent_id, order, out);
                let mut fills = std::mem::take(&mut self.fill_buf);
                fills.clear();
                let status = self.book.add_limit_order_into(
                    LimitOrder {
                        id: order.id,
                        side,
                        price,
                        qty,
                        timestamp: time,
                    },
                    &mut fills,
                );
                self.record_fills(agent_id, side, &fills, time, mid0, out);
                self.notify_taker(agent_id, order, status, &fills, time, out);
                self.fill_buf = fills;
            }
            OrderAction::NewMarketOrder { side, qty, user_id } => {
                let order = NewOrder {
                    id: self.alloc_id(),
                    user_id,
                    side,
                    price: 0,
                    qty,
                };
                self.record_submission(agent_id, time, side, None, qty, order.id, mid0);
                self.push_accepted(agent_id, order, out);
                let mut fills = std::mem::take(&mut self.fill_buf);
                fills.clear();
                let status = self.book.add_market_order_into(
                    MarketOrder {
                        id: order.id,
                        side,
                        qty,
                    },
                    &mut fills,
                );
                self.record_fills(agent_id, side, &fills, time, mid0, out);
                self.notify_taker(agent_id, order, status, &fills, time, out);
                self.fill_buf = fills;
            }
            OrderAction::CancelOrder { order_id } => {
                if self.book.cancel_order(order_id) {
                    let user_id = self
                        .resting
                        .remove(&order_id)
                        .map_or(0, |info| info.user_id);
                    if let Some(flow) = self.flow.as_mut() {
                        flow.on_order_gone(time, order_id);
                    }
                    out.push(RoutedMessage {
                        agent_id,
                        message: ExchangeMessage::OrderCancelled {
                            order_id,
                            user_id,
                            symbol: self.sym,
                        },
                    });
                } else {
                    out.push(RoutedMessage {
                        agent_id,
                        message: ExchangeMessage::OrderRejected {
                            order_id,
                            user_id: 0,
                            symbol: self.sym,
                        },
                    });
                }
            }
        }

        let bbo_after = (self.book.best_bid(), self.book.best_ask());
        if bbo_after != bbo_before {
            self.record_l1(time);
        }
    }

    /// Total resting orders on the book.
    #[must_use]
    pub fn order_count(&self) -> usize {
        self.book.order_count()
    }

    fn alloc_id(&mut self) -> u64 {
        let id = self.next_order_id;
        self.next_order_id += 1;
        id
    }

    /// Creation acknowledgment: sent exactly once for every new order —
    /// including one that fully fills at submission — and always emitted
    /// before the order's fills. (Per-message latency jitter may still
    /// deliver a fill first; messages are self-contained so delivery order
    /// carries no information.)
    fn push_accepted(&self, agent_id: AgentId, order: NewOrder, out: &mut Vec<RoutedMessage>) {
        out.push(RoutedMessage {
            agent_id,
            message: ExchangeMessage::OrderAccepted {
                order_id: order.id,
                user_id: order.user_id,
                symbol: self.sym,
                side: order.side,
                qty: order.qty,
            },
        });
    }

    #[allow(clippy::too_many_arguments, clippy::cast_precision_loss)]
    fn record_submission(
        &mut self,
        agent_id: AgentId,
        time: Nanos,
        side: Side,
        limit_price: Option<i64>,
        qty: u64,
        order_id: u64,
        mid0: Option<f64>,
    ) {
        let Some(flow) = self.flow.as_mut() else {
            return;
        };
        if agent_id != flow.options.agent_id {
            return;
        }
        let (_, depth_notional) = self.book.executable_depth(side, limit_price);
        let executable = |price: i64| match side {
            Side::Bid => limit_price.is_none_or(|limit| price <= limit),
            Side::Ask => limit_price.is_none_or(|limit| price >= limit),
        };
        let own_notional: u128 = flow
            .live
            .iter()
            .filter_map(|id| self.resting.get(id))
            .filter(|info| info.side != side && executable(info.price))
            .map(|info| u128::from(info.price.unsigned_abs()) * u128::from(info.remaining_qty))
            .sum();
        let external_depth = depth_notional.saturating_sub(own_notional);
        flow.on_accept(time, agent_id, order_id, qty);
        let order_notional = match limit_price {
            Some(limit) => Some(limit.unsigned_abs() as f64 * qty as f64),
            None => mid0
                .filter(|&m| m.is_finite() && m > 0.0)
                .map(|m| m * qty as f64),
        };
        flow.on_depth(time, qty, order_notional, external_depth);
    }

    fn record_fills(
        &mut self,
        taker: AgentId,
        taker_side: Side,
        fills: &[cda_engine::Fill],
        time: Nanos,
        mid0: Option<f64>,
        out: &mut Vec<RoutedMessage>,
    ) {
        let maker_side = match taker_side {
            Side::Bid => Side::Ask,
            Side::Ask => Side::Bid,
        };
        for fill in fills {
            self.last_trade_price = Some(fill.price);
            self.last_trade_time = Some(time);

            if let Some(flow) = self.flow.as_mut() {
                let maker_agent = self
                    .resting
                    .get(&fill.maker_order_id)
                    .map(|info| info.agent_id);
                flow.on_fill(
                    time,
                    taker,
                    fill.taker_order_id,
                    maker_agent,
                    fill.maker_order_id,
                    taker_side,
                    fill.price,
                    fill.qty,
                    mid0,
                );
            }

            if self.keep_trades {
                self.trades.push(TradeRecord {
                    timestamp: time,
                    symbol: self.sym,
                    price: fill.price,
                    qty: fill.qty,
                    aggressor_side: taker_side,
                    maker_order_id: fill.maker_order_id,
                    taker_order_id: fill.taker_order_id,
                });
            }

            if let Some(info) = self.resting.get_mut(&fill.maker_order_id) {
                info.remaining_qty = info.remaining_qty.saturating_sub(fill.qty);
                out.push(RoutedMessage {
                    agent_id: info.agent_id,
                    message: ExchangeMessage::OrderFilled {
                        order_id: fill.maker_order_id,
                        user_id: info.user_id,
                        symbol: self.sym,
                        side: maker_side,
                        price: fill.price,
                        qty: fill.qty,
                        remaining: info.remaining_qty,
                        notional: fill_notional(fill),
                    },
                });
                if info.remaining_qty == 0 {
                    self.resting.remove(&fill.maker_order_id);
                    if let Some(flow) = self.flow.as_mut() {
                        flow.on_order_gone(time, fill.maker_order_id);
                    }
                }
            }
        }
    }

    fn notify_taker(
        &mut self,
        taker: AgentId,
        order: NewOrder,
        status: OrderStatus,
        fills: &[Fill],
        time: Nanos,
        out: &mut Vec<RoutedMessage>,
    ) {
        let last_price = fills.last().map_or(0, |f| f.price);
        let notional: u128 = fills.iter().map(fill_notional).sum();
        if matches!(status, OrderStatus::Filled | OrderStatus::Cancelled { .. }) {
            if let Some(flow) = self.flow.as_mut() {
                flow.on_order_gone(time, order.id);
            }
        }
        match status {
            OrderStatus::Filled => {
                let total: u64 = fills.iter().map(|f| f.qty).sum();
                out.push(RoutedMessage {
                    agent_id: taker,
                    message: ExchangeMessage::OrderFilled {
                        order_id: order.id,
                        user_id: order.user_id,
                        symbol: self.sym,
                        side: order.side,
                        price: last_price,
                        qty: total,
                        remaining: 0,
                        notional,
                    },
                });
            }
            OrderStatus::Placed => {
                self.resting.insert(
                    order.id,
                    OrderInfo {
                        agent_id: taker,
                        side: order.side,
                        price: order.price,
                        remaining_qty: order.qty,
                        user_id: order.user_id,
                    },
                );
            }
            OrderStatus::Resting { remaining_qty } => {
                self.resting.insert(
                    order.id,
                    OrderInfo {
                        agent_id: taker,
                        side: order.side,
                        price: order.price,
                        remaining_qty,
                        user_id: order.user_id,
                    },
                );
                let filled: u64 = fills.iter().map(|f| f.qty).sum();
                if filled > 0 {
                    out.push(RoutedMessage {
                        agent_id: taker,
                        message: ExchangeMessage::OrderFilled {
                            order_id: order.id,
                            user_id: order.user_id,
                            symbol: self.sym,
                            side: order.side,
                            price: last_price,
                            qty: filled,
                            remaining: remaining_qty,
                            notional,
                        },
                    });
                }
            }
            OrderStatus::Cancelled { filled_qty } => {
                if filled_qty > 0 {
                    out.push(RoutedMessage {
                        agent_id: taker,
                        message: ExchangeMessage::OrderFilled {
                            order_id: order.id,
                            user_id: order.user_id,
                            symbol: self.sym,
                            side: order.side,
                            price: last_price,
                            qty: filled_qty,
                            remaining: order.qty - filled_qty,
                            notional,
                        },
                    });
                }
                out.push(RoutedMessage {
                    agent_id: taker,
                    message: ExchangeMessage::OrderCancelled {
                        order_id: order.id,
                        user_id: order.user_id,
                        symbol: self.sym,
                    },
                });
            }
        }
    }

    fn record_l1(&mut self, time: Nanos) {
        let (bid_price, bid_volume) = self.book.best_bid_level().unwrap_or((0, 0));
        let (ask_price, ask_volume) = self.book.best_ask_level().unwrap_or((0, 0));
        if let Some(flow) = self.flow.as_mut() {
            flow.on_quote(time, bid_price, ask_price);
        }
        if let Some(bucket_ns) = self.l1_bucket_ns {
            let bucket_start = time - time % bucket_ns;
            if self.open_bucket.map(|b| b.bucket_start) != Some(bucket_start) {
                self.flush_l1_bucket();
                self.open_bucket = Some(L1Bucket::new(bucket_start, self.sym));
            }
            if let Some(bucket) = self.open_bucket.as_mut() {
                bucket.absorb(bid_price, ask_price, bid_volume, ask_volume);
            }
            return;
        }
        self.l1_snapshots.push(L1Snapshot {
            timestamp: time,
            symbol: self.sym,
            bid_price,
            ask_price,
            bid_volume,
            ask_volume,
            last_trade_price: self.last_trade_price.unwrap_or(0),
        });
    }
}

fn fill_notional(fill: &Fill) -> u128 {
    u128::from(fill.price.unsigned_abs()) * u128::from(fill.qty)
}

#[cfg(test)]
mod tests {
    use super::*;

    const SYM: Symbol = 7;

    fn open_exchange() -> Exchange {
        let mut ex = Exchange::new(SYM, 1_000_000_000);
        ex.open(0);
        ex
    }

    fn limit(side: Side, price: i64, qty: u64, user_id: u64) -> OrderAction {
        OrderAction::NewLimitOrder {
            side,
            price,
            qty,
            user_id,
        }
    }

    // Every new order gets exactly one creation ack — with symbol, side, qty
    // and the echoed user_id — before any of its fills, INCLUDING an order
    // that fully fills at submission (which previously got no ack at all).
    #[test]
    fn full_fill_still_gets_creation_ack_before_its_fill() {
        let mut ex = open_exchange();
        let mut msgs = Vec::new();
        ex.process_into(0, limit(Side::Ask, 100, 10, 5), 0, &mut msgs);
        msgs.clear();
        ex.process_into(1, limit(Side::Bid, 100, 10, 9), 1, &mut msgs);

        let to_taker: Vec<_> = msgs.iter().filter(|m| m.agent_id == 1).collect();
        assert_eq!(to_taker.len(), 2, "creation ack + fill");
        match to_taker[0].message {
            ExchangeMessage::OrderAccepted {
                user_id,
                symbol,
                side,
                qty,
                ..
            } => {
                assert_eq!((user_id, symbol, side, qty), (9, SYM, Side::Bid, 10));
            }
            other => panic!("first taker message must be the creation ack, got {other:?}"),
        }
        match to_taker[1].message {
            ExchangeMessage::OrderFilled {
                user_id,
                symbol,
                side,
                qty,
                remaining,
                ..
            } => {
                assert_eq!(
                    (user_id, symbol, side, qty, remaining),
                    (9, SYM, Side::Bid, 10, 0)
                );
            }
            other => panic!("second taker message must be the fill, got {other:?}"),
        }
    }

    // An order that partially fills at submission and rests: the creation
    // ack comes first, and the fill reports the outstanding remainder, so an
    // owner can maintain its resting set from fill content alone.
    #[test]
    fn partial_fill_at_submission_emits_ack_then_fill_with_remainder() {
        let mut ex = open_exchange();
        let mut msgs = Vec::new();
        ex.process_into(0, limit(Side::Ask, 100, 4, 5), 0, &mut msgs);
        msgs.clear();
        ex.process_into(1, limit(Side::Bid, 100, 10, 9), 1, &mut msgs);

        let to_taker: Vec<_> = msgs.iter().filter(|m| m.agent_id == 1).collect();
        assert_eq!(to_taker.len(), 2, "creation ack + submission fill");
        match to_taker[0].message {
            ExchangeMessage::OrderAccepted { qty, side, .. } => {
                assert_eq!((qty, side), (10, Side::Bid));
            }
            other => panic!("first message must be the creation ack, got {other:?}"),
        }
        match to_taker[1].message {
            ExchangeMessage::OrderFilled {
                qty,
                remaining,
                side,
                ..
            } => {
                assert_eq!((qty, remaining, side), (4, 6, Side::Bid));
            }
            other => panic!("second message must be the submission fill, got {other:?}"),
        }
    }

    // A fill of a resting order carries the RESTING order's side and its
    // owner's user_id, not the aggressor's.
    #[test]
    fn maker_fill_carries_maker_side_and_user_id() {
        let mut ex = open_exchange();
        let mut msgs = Vec::new();
        ex.process_into(0, limit(Side::Ask, 100, 10, 5), 0, &mut msgs);
        msgs.clear();
        ex.process_into(1, limit(Side::Bid, 100, 4, 9), 1, &mut msgs);

        let maker_fill = msgs
            .iter()
            .find(|m| m.agent_id == 0)
            .expect("maker must be notified of the fill");
        match maker_fill.message {
            ExchangeMessage::OrderFilled {
                user_id,
                symbol,
                side,
                qty,
                remaining,
                ..
            } => {
                assert_eq!(
                    (user_id, symbol, side, qty, remaining),
                    (5, SYM, Side::Ask, 4, 6)
                );
            }
            other => panic!("maker must receive a fill, got {other:?}"),
        }
    }

    // A new order rejected on a closed market has no order id, but its
    // echoed user_id and symbol still identify it.
    #[test]
    fn rejected_new_order_echoes_user_id_and_symbol() {
        let mut ex = Exchange::new(SYM, 1_000_000_000); // never opened
        let mut msgs = Vec::new();
        ex.process_into(3, limit(Side::Bid, 100, 10, 42), 0, &mut msgs);
        assert_eq!(msgs.len(), 1);
        match msgs[0].message {
            ExchangeMessage::OrderRejected {
                order_id,
                user_id,
                symbol,
            } => {
                assert_eq!((order_id, user_id, symbol), (0, 42, SYM));
            }
            other => panic!("expected a rejection, got {other:?}"),
        }
    }

    // Cancels (agent-requested and market-close) echo the order's user_id.
    #[test]
    fn cancels_echo_the_orders_user_id() {
        let mut ex = open_exchange();
        let mut msgs = Vec::new();
        ex.process_into(0, limit(Side::Bid, 90, 10, 11), 0, &mut msgs);
        let oid = match msgs[0].message {
            ExchangeMessage::OrderAccepted { order_id, .. } => order_id,
            other => panic!("expected the creation ack, got {other:?}"),
        };

        msgs.clear();
        ex.process_into(0, OrderAction::CancelOrder { order_id: oid }, 1, &mut msgs);
        match msgs[0].message {
            ExchangeMessage::OrderCancelled {
                order_id,
                user_id,
                symbol,
            } => {
                assert_eq!((order_id, user_id, symbol), (oid, 11, SYM));
            }
            other => panic!("expected the cancel echo, got {other:?}"),
        }

        msgs.clear();
        ex.process_into(0, limit(Side::Bid, 90, 10, 12), 2, &mut msgs);
        msgs.clear();
        ex.close_into(3, &mut msgs);
        match msgs[0].message {
            ExchangeMessage::OrderCancelled {
                user_id, symbol, ..
            } => {
                assert_eq!((user_id, symbol), (12, SYM));
            }
            other => panic!("expected the close-cancel echo, got {other:?}"),
        }
    }

    fn flow_exchange(ratio_edges: Vec<f64>) -> Exchange {
        let mut ex = Exchange::new(SYM, 1);
        ex.set_flow_options(Some(FlowOptions {
            agent_id: 0,
            interval_ns: 100,
            horizons_ns: vec![10],
            ratio_edges,
        }));
        ex.open(0);
        ex
    }

    fn place(ex: &mut Exchange, owner: AgentId, time: Nanos, action: OrderAction) -> u64 {
        let mut out = Vec::new();
        ex.process_into(owner, action, time, &mut out);
        out.iter()
            .find_map(|routed| match routed.message {
                ExchangeMessage::OrderAccepted { order_id, .. } if routed.agent_id == owner => {
                    Some(order_id)
                }
                _ => None,
            })
            .expect("creation ack")
    }

    #[test]
    fn the_depth_ratio_excludes_the_watched_agents_own_resting_orders() {
        let mut ex = flow_exchange(vec![0.5, 1.0]);
        place(&mut ex, 0, 0, limit(Side::Ask, 100, 10, 0));
        place(&mut ex, 1, 1, limit(Side::Ask, 100, 10, 0));
        place(&mut ex, 0, 2, limit(Side::Bid, 100, 10, 0));
        let (rows, _) = ex.flush_flow(99).unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].ratio_count, vec![0, 0, 1]);
        assert_eq!(rows[0].ratio_notional, vec![0.0, 0.0, 1_000.0]);
        assert_eq!(rows[0].zero_depth_count, 1);
        assert_eq!(rows[0].zero_depth_qty, 10);
    }

    #[test]
    fn the_own_depth_exclusion_uses_the_remaining_quantity() {
        let mut ex = flow_exchange(vec![0.5, 2.0]);
        place(&mut ex, 0, 0, limit(Side::Ask, 100, 10, 0));
        place(&mut ex, 1, 1, limit(Side::Bid, 100, 4, 0));
        place(&mut ex, 1, 2, limit(Side::Ask, 100, 10, 0));
        place(&mut ex, 0, 3, limit(Side::Bid, 100, 16, 0));
        let (rows, _) = ex.flush_flow(99).unwrap();
        assert_eq!(rows[0].ratio_count, vec![0, 1, 0]);
    }

    #[test]
    fn a_carried_order_canceled_in_a_new_interval_stays_in_that_cohort() {
        let mut ex = flow_exchange(vec![0.5, 1.0]);
        let order_id = place(&mut ex, 0, 0, limit(Side::Bid, 100, 10, 0));
        ex.process_into(
            0,
            OrderAction::CancelOrder { order_id },
            110,
            &mut Vec::new(),
        );
        let (rows, _) = ex.flush_flow(199).unwrap();
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[1].eligible_orders, 1);
        assert_eq!(rows[1].orders, 0);
        assert_eq!(rows[1].filled_orders, 0);
    }

    #[test]
    fn an_event_exactly_at_the_end_lands_in_the_last_interval() {
        let mut ex = flow_exchange(vec![0.5, 1.0]);
        place(&mut ex, 0, 200, limit(Side::Bid, 100, 10, 0));
        let (rows, totals) = ex.flush_flow(200).unwrap();
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[1].orders, 1);
        assert_eq!(rows[1].eligible_orders, 1);
        assert_eq!(totals.accepted_orders, 1);
    }

    #[test]
    fn a_fill_exactly_at_the_end_does_not_repeat_the_filled_order() {
        let mut ex = flow_exchange(vec![0.5, 1.0]);
        place(&mut ex, 0, 50, limit(Side::Bid, 100, 10, 0));
        place(&mut ex, 1, 150, limit(Side::Ask, 100, 3, 0));
        place(&mut ex, 1, 200, limit(Side::Ask, 100, 3, 0));
        let (rows, totals) = ex.flush_flow(200).unwrap();
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[1].filled_orders, 1);
        assert_eq!(rows[1].fill_events, 2);
        assert_eq!(rows[1].eligible_orders, 1);
        assert_eq!(rows[1].market_trades, 2);
        assert_eq!(totals.filled_orders, 1);
        assert_eq!(totals.fill_events, 2);
    }

    #[test]
    fn a_market_order_without_a_mid_is_unvalued_not_zero_depth() {
        let mut ex = flow_exchange(vec![0.5, 1.0]);
        place(&mut ex, 1, 0, limit(Side::Ask, 100, 5, 0));
        let mut out = Vec::new();
        ex.process_into(
            0,
            OrderAction::NewMarketOrder {
                side: Side::Bid,
                qty: 5,
                user_id: 0,
            },
            1,
            &mut out,
        );
        place(&mut ex, 0, 2, limit(Side::Ask, 90, 4, 0));
        let (rows, _) = ex.flush_flow(99).unwrap();
        assert_eq!((rows[0].unvalued_count, rows[0].unvalued_qty), (1, 5));
        assert_eq!(rows[0].zero_depth_count, 1);
        assert_eq!(rows[0].zero_depth_qty, 4);
        assert!((rows[0].zero_depth_notional - 360.0).abs() < 1e-12);
        assert_eq!(rows[0].ratio_count, vec![0, 0, 0]);
    }

    #[test]
    fn fixed_price_partial_fills_keep_the_pending_responses_bounded() {
        let mut ex = Exchange::new(SYM, 1);
        ex.set_flow_options(Some(FlowOptions {
            agent_id: 0,
            interval_ns: 1_000_000,
            horizons_ns: vec![10, 1_000],
            ratio_edges: vec![0.5, 1.0],
        }));
        ex.open(0);
        place(&mut ex, 1, 1, limit(Side::Bid, 90, 1, 0));
        place(&mut ex, 1, 2, limit(Side::Ask, 100, 50, 0));
        for fill in 1..=10 {
            place(&mut ex, 0, fill * 100, limit(Side::Bid, 100, 1, 0));
            let flow = ex.flow.as_ref().unwrap();
            let pending: Vec<usize> = flow
                .pending
                .iter()
                .map(std::collections::VecDeque::len)
                .collect();
            assert!(
                pending[0] <= 1,
                "fill {fill} keeps {} expired entries",
                pending[0]
            );
            assert_eq!(pending[1] as u64, fill);
        }
        let (rows, _) = ex.flush_flow(1_500).unwrap();
        let row = &rows[0];
        assert_eq!(row.response_count[0], 10);
        assert!((row.response_den[0] - 1_000.0).abs() < 1e-12);
        assert!(row.response_num[0].abs() < 1e-12);
        assert_eq!(row.response_count[1], 5);
        assert!((row.response_den[1] - 500.0).abs() < 1e-12);
        assert!(row.response_num[1].abs() < 1e-12);
        assert_eq!(row.response_excluded_count[1], 5);
        assert!((row.response_excluded_notional[1] - 500.0).abs() < 1e-12);
        assert_eq!(row.response_excluded_count[0], 0);
    }
}
