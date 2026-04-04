"""
state.py
--------
Two responsibilities:
  1. SupportAgentState  — the graph's short-term memory schema.
     All fields here exist only for the duration of one agent.invoke() call.
     Nothing is persisted anywhere after the session ends.

  2. Pydantic input schemas — define the structured input contracts for
     tools that take more than a plain string argument (from the docs:
     "Advanced schema definition").
"""

from typing import Literal

from langgraph.prebuilt.chat_agent_executor import AgentState
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Short-term state — extends LangGraph's built-in AgentState
# AgentState already gives us: messages (the conversation history)
# We add our own fields on top for session-level context.
# ---------------------------------------------------------------------------

class SupportAgentState(AgentState):
    """
    Short-term memory for one customer support session.

    All fields reset to their initial values at the start of every new
    agent.invoke() call — nothing carries over to the next session.

    Fields
    ------
    customer_id         : who is logged in (injected at invocation time)
    session_id          : unique ID for this conversation thread
    turn_count          : how many user turns have elapsed this session
    last_order_checked  : order ID most recently looked up (updated by tool)
    escalation_flag     : True when feedback rating < 3 (updated by tool)
    """

    customer_id: str
    session_id: str
    turn_count: int
    last_order_checked: str
    escalation_flag: bool


# ---------------------------------------------------------------------------
# Pydantic input schemas
# Concept from docs: "Advanced schema definition" using args_schema=
# These replace plain function signatures with rich, validated contracts
# that the LLM uses to understand exactly what arguments to supply.
# ---------------------------------------------------------------------------

class OrderLookupInput(BaseModel):
    """Input schema for check_order_status."""

    order_id: str = Field(
        description="Order ID in the format ORD-XXX (e.g. ORD-001)"
    )
    include_price: bool = Field(
        default=False,
        description="Set True to include the item price in the response"
    )


class RefundInput(BaseModel):
    """Input schema for process_refund_request."""

    order_id: str = Field(
        description="Order ID to refund, format ORD-XXX"
    )
    reason: str = Field(
        description="Customer's stated reason for the refund"
    )


class FeedbackInput(BaseModel):
    """Input schema for submit_feedback."""

    category: Literal["delivery", "product", "support", "billing"] = Field(
        description="Which area the feedback is about"
    )
    rating: int = Field(
        description="Star rating from 1 (worst) to 5 (best)",
        ge=1,
        le=5,
    )
    comment: str = Field(
        description="Customer's written feedback comment"
    )
