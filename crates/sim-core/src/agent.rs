use crate::event::{ExchangeMessage, OrderAction};
use crate::types::{AgentId, MarketSnapshot, Nanos, Symbol};

/// An action returned by an agent during its wakeup callback.
#[derive(Debug, Clone)]
pub enum AgentAction {
    SubmitOrder { symbol: Symbol, order: OrderAction },
    CancelOrder { symbol: Symbol, order_id: u64 },
    ScheduleWakeUp { delay_ns: Nanos },
}

/// Trait implemented by all simulation agents. Must be object-safe.
pub trait Agent {
    /// Called when the agent's wakeup event fires.
    /// Writes actions into the provided buffer to avoid allocation.
    fn wakeup_into(
        &mut self,
        time: Nanos,
        agent_id: AgentId,
        snapshots: &[MarketSnapshot],
        actions: &mut Vec<AgentAction>,
    );

    /// Called when the exchange sends a message to this agent.
    fn on_exchange_message(
        &mut self,
        time: Nanos,
        agent_id: AgentId,
        message: ExchangeMessage,
    );
}
