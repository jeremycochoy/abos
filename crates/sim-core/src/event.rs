use std::cmp::Ordering;

use crate::types::{AgentId, Nanos, Symbol};

/// An action an agent wants to perform on a symbol's order book.
///
/// `user_id` is an agent-chosen tag echoed verbatim on every exchange message
/// about the order (`0` = unset). It lets an agent attribute echoes to its
/// own bookkeeping directly (e.g. a per-market slot index) instead of
/// decoding exchange-assigned order ids.
#[derive(Debug, Clone, Copy)]
pub enum OrderAction {
    NewLimitOrder { side: cda_engine::Side, price: i64, qty: u64, user_id: u64 },
    NewMarketOrder { side: cda_engine::Side, qty: u64, user_id: u64 },
    CancelOrder { order_id: u64 },
}

/// A message sent from the exchange back to an agent.
///
/// Every message is self-contained: it names the market (`symbol`), echoes
/// the submitting agent's `user_id` tag (`0` = unset), and creation/fill
/// messages carry the ORDER OWNER's side and quantity. No message needs to
/// be correlated with another one to be interpreted.
#[derive(Debug, Clone, Copy)]
pub enum ExchangeMessage {
    /// Sent exactly once for every order the exchange creates — including an
    /// order that fully fills at submission — and always before any of its
    /// fills.
    OrderAccepted { order_id: u64, user_id: u64, symbol: Symbol, side: cda_engine::Side, qty: u64 },
    /// One fill of the order. `side` is the order owner's side (for a fill
    /// of a resting order, that is the resting order's side). `remaining` is
    /// the order's unfilled quantity AFTER this fill: 0 means the order is
    /// done, a positive value means it is still resting — so an owner can
    /// maintain its resting-order set from fills alone, without inferring
    /// lifecycle from message ordering.
    OrderFilled {
        order_id: u64,
        user_id: u64,
        symbol: Symbol,
        side: cda_engine::Side,
        price: i64,
        qty: u64,
        remaining: u64,
        notional: u128,
    },
    OrderCancelled { order_id: u64, user_id: u64, symbol: Symbol },
    /// A rejected NEW order is reported with `order_id` 0 (no id is ever
    /// allocated for it); its echoed `user_id` and `symbol` still identify
    /// it. A rejected cancel carries the cancel's target `order_id`.
    OrderRejected { order_id: u64, user_id: u64, symbol: Symbol },
}

/// Payload carried by each event in the simulation queue.
#[derive(Debug)]
pub enum EventPayload {
    WakeUp { agent_id: AgentId },
    OrderArrival { agent_id: AgentId, symbol: Symbol, order: OrderAction },
    ExchangeResponse { agent_id: AgentId, response: ExchangeMessage },
    MarketOpen { symbol: Symbol },
    MarketClose { symbol: Symbol },
}

/// A time-stamped event processed by the kernel.
#[derive(Debug)]
pub struct Event {
    pub delivery_time: Nanos,
    pub seq: u64,
    pub payload: EventPayload,
}

// Min-heap ordering: smallest (delivery_time, seq) has highest priority.
impl Eq for Event {}

impl PartialEq for Event {
    fn eq(&self, other: &Self) -> bool {
        self.delivery_time == other.delivery_time && self.seq == other.seq
    }
}

impl Ord for Event {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reverse so BinaryHeap (max-heap) yields the smallest first.
        other
            .delivery_time
            .cmp(&self.delivery_time)
            .then_with(|| other.seq.cmp(&self.seq))
    }
}

impl PartialOrd for Event {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
