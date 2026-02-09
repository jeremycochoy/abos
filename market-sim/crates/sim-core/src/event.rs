use std::cmp::Ordering;

use crate::types::{AgentId, Nanos, Symbol};

/// An action an agent wants to perform on a symbol's order book.
#[derive(Debug, Clone, Copy)]
pub enum OrderAction {
    NewLimitOrder { side: cda_engine::Side, price: i64, qty: u64 },
    NewMarketOrder { side: cda_engine::Side, qty: u64 },
    CancelOrder { order_id: u64 },
}

/// A message sent from the exchange back to an agent.
#[derive(Debug, Clone, Copy)]
pub enum ExchangeMessage {
    OrderAccepted { order_id: u64 },
    OrderFilled { order_id: u64, price: i64, qty: u64 },
    OrderCancelled { order_id: u64 },
    OrderRejected { order_id: u64 },
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
