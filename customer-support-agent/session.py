"""
session.py
----------
Manages a single customer support session — the short-term memory layer.

Key idea
--------
Short-term memory in LangGraph = the messages list.
We pass the accumulated message history back into every agent.invoke() call.
The agent can therefore "remember" everything said earlier in the session.
When the session ends (or the Python process exits), memory is gone.

This module provides:
  - run_turn()    run one user message through the agent, carry history forward
  - Session       context-manager that owns a session's state for multiple turns
"""

from dataclasses import dataclass, field

from langchain_core.messages import BaseMessage, HumanMessage

from agent import agent


# ---------------------------------------------------------------------------
# Low-level helper — single turn, stateless
# ---------------------------------------------------------------------------

def run_turn(
    customer_id: str,
    session_id: str,
    turn_count: int,
    message: str,
    prior_messages: list[BaseMessage] | None = None,
    escalation_flag: bool = False,
) -> list[BaseMessage]:
    """
    Send one user message to the agent and return the updated message list.

    Parameters
    ----------
    customer_id     : identifies the logged-in customer (injected as context)
    session_id      : unique ID for this conversation thread
    turn_count      : number of turns elapsed before this one
    message         : the customer's current message
    prior_messages  : message history from previous turns (short-term memory)
    escalation_flag : current value of the escalation flag in state

    Returns
    -------
    Full updated message list — pass this as prior_messages on the next turn.
    """
    prior_messages = prior_messages or []
    new_message    = HumanMessage(content=message)
    all_messages   = prior_messages + [new_message]

    print(f"\n{'─' * 62}")
    print(f"  Turn {turn_count + 1}  |  {message}")
    print(f"{'─' * 62}")

    result = agent.invoke(
        # Graph state — initial values for this invocation
        {
            "messages":           all_messages,
            "customer_id":        customer_id,
            "session_id":         session_id,
            "turn_count":         turn_count,
            "last_order_checked": "",
            "escalation_flag":    escalation_flag,
        },
        # Config — injected into tools via RunnableConfig
        config={
            "configurable": {
                "customer_id":     customer_id,
                "session_id":      session_id,
                "turn_count":      turn_count,
                "escalation_flag": escalation_flag,
                # Pass messages so tools like summarise_conversation can read them
                "messages":        all_messages,
            }
        },
    )

    response = result["messages"][-1].content
    print(f"\n  Aria: {response}")

    # Return full updated message list — this becomes prior_messages next turn
    return result["messages"]


# ---------------------------------------------------------------------------
# High-level helper — stateful session context manager
# Preferred for multi-turn demos: tracks history and turn count automatically
# ---------------------------------------------------------------------------

@dataclass
class Session:
    """
    Stateful wrapper for a multi-turn customer support session.

    Usage
    -----
    with Session(customer_id="CUST-101") as s:
        s.send("What is my account balance?")
        s.send("Check order ORD-001 please.")
        s.send("I want a refund for ORD-003.")

    All message history is kept in self.history and passed automatically
    to every turn — that is the short-term memory in action.
    """

    customer_id: str
    session_id: str = field(default_factory=lambda: "sess-" + __import__("uuid").uuid4().hex[:8])
    history: list[BaseMessage] = field(default_factory=list)
    turn_count: int = 0
    escalation_flag: bool = False

    def send(self, message: str) -> str:
        """Send a message and advance the session by one turn."""
        self.history = run_turn(
            customer_id     = self.customer_id,
            session_id      = self.session_id,
            turn_count      = self.turn_count,
            message         = message,
            prior_messages  = self.history,
            escalation_flag = self.escalation_flag,
        )
        self.turn_count += 1
        # Sync escalation_flag from updated state if tool wrote it via Command
        # (In a full graph setup this would be read directly from result["escalation_flag"])
        return self.history[-1].content

    def __enter__(self):
        print(f"\n{'═' * 62}")
        print(f"  Session started  |  customer={self.customer_id}  |  id={self.session_id}")
        print(f"{'═' * 62}")
        return self

    def __exit__(self, *_):
        print(f"\n{'═' * 62}")
        print(f"  Session ended  |  {self.turn_count} turn(s)  |  id={self.session_id}")
        print(f"{'═' * 62}\n")
