"""
tools.py
--------
All six LangChain tools for the customer support agent.
Each tool is annotated with the exact concept it demonstrates from the docs.

Concept map
-----------
Tool                        Concept demonstrated
--------------------------  -----------------------------------------------
check_order_status          Basic @tool + Pydantic args_schema
get_my_account              Context — reads customer_id from RunnableConfig
get_session_status          State reader — reads short-term state fields
summarise_conversation      State reader — reads state["messages"] history
process_refund_request      Stream writer — real-time progress updates
submit_feedback             State writer — mutates state via Command
"""

import time

from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.types import Command

from config import CUSTOMER_DB, ORDER_DB
from state import FeedbackInput, OrderLookupInput, RefundInput


# ---------------------------------------------------------------------------
# TOOL 1 — Basic tool with Pydantic schema
# Concept: @tool decorator, docstring becomes the tool description the LLM
#          reads, type hints define the schema, args_schema adds validation.
# ---------------------------------------------------------------------------

@tool(args_schema=OrderLookupInput)
def check_order_status(order_id: str, include_price: bool = False) -> str:
    """
    Check the current status of a customer order by its order ID.
    Optionally include the item price in the response.
    """
    order = ORDER_DB.get(order_id.upper())
    if not order:
        return f"Order '{order_id}' not found. Please double-check the order ID."

    lines = [
        f"Order {order_id.upper()}:",
        f"  Item   : {order['item']}",
        f"  Status : {order['status']}",
        f"  ETA    : {order['eta']}",
    ]
    if include_price:
        lines.append(f"  Price  : ₹{order['price']:,}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# TOOL 2 — Context-aware tool
# Concept: config["configurable"] carries immutable identity data injected
#          at invocation time (customer_id, session_id). It cannot change
#          mid-conversation — reads only, never writes.
# ---------------------------------------------------------------------------

@tool
def get_my_account(config: RunnableConfig) -> str:
    """
    Retrieve the current logged-in customer's profile and wallet balance.
    Uses the customer_id injected via config at session start.
    """
    customer_id = config.get("configurable", {}).get("customer_id", "UNKNOWN")
    customer = CUSTOMER_DB.get(customer_id)

    if not customer:
        return f"No account found for customer ID '{customer_id}'."

    return "\n".join([
        "Account Details:",
        f"  Name    : {customer['name']}",
        f"  Tier    : {customer['tier']}",
        f"  Email   : {customer['email']}",
        f"  Wallet  : ₹{customer['wallet']:,}",
    ])


# ---------------------------------------------------------------------------
# TOOL 3 — State reader (structured fields)
# Concept: Reads custom short-term state fields (turn_count, session_id,
#          escalation_flag) that accumulate during the conversation.
#          These are injected via config["configurable"] at each turn.
# ---------------------------------------------------------------------------

@tool
def get_session_status(config: RunnableConfig) -> str:
    """
    Return a summary of the current session's progress and escalation status.
    Helps the agent decide whether to keep resolving or offer a human handoff.
    """
    cfg          = config.get("configurable", {})
    turn_count   = cfg.get("turn_count", 0)
    session_id   = cfg.get("session_id", "unknown")
    escalated    = cfg.get("escalation_flag", False)

    if escalated:
        advice = "Escalation flag is set — offer human agent handoff immediately."
    elif turn_count < 3:
        advice = "Early in session — attempt self-service resolution first."
    else:
        advice = "Extended session — consider offering human agent option."

    return "\n".join([
        f"Session ID : {session_id}",
        f"Turns      : {turn_count}",
        f"Advice     : {advice}",
    ])


# ---------------------------------------------------------------------------
# TOOL 4 — State reader (message history)
# Concept: state["messages"] is the short-term memory — the full running log
#          of the conversation. Tools can inspect it to summarise, detect
#          topics, or build a handoff brief for a human agent.
# ---------------------------------------------------------------------------

@tool
def summarise_conversation(config: RunnableConfig) -> str:
    """
    List every customer message sent so far in the current session.
    Use this to create a handoff brief or give the customer a recap.
    """
    messages = config.get("configurable", {}).get("messages", [])
    human_turns = [m for m in messages if isinstance(m, HumanMessage)]

    if not human_turns:
        return "No customer messages found in the current session."

    lines = [f"{len(human_turns)} customer message(s) this session:"]
    for i, msg in enumerate(human_turns, 1):
        preview = msg.content[:80] + ("..." if len(msg.content) > 80 else "")
        lines.append(f"  {i}. {preview}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# TOOL 5 — Stream writer pattern
# Concept: emit real-time progress updates while the tool runs long tasks.
#          In production LangGraph, annotate the parameter with StreamWriter
#          from langgraph.types. Here we use print() to make the pattern
#          visible without adding graph complexity.
# ---------------------------------------------------------------------------

@tool(args_schema=RefundInput)
def process_refund_request(order_id: str, reason: str) -> str:
    """
    Process a refund for a delivered order. Streams step-by-step progress
    updates as each validation stage completes.
    """
    steps = [
        f"Validating order {order_id.upper()}...",
        "Checking 30-day refund eligibility window...",
        "Calculating refund amount...",
        "Initiating bank transfer...",
        "Queuing confirmation email...",
    ]

    order = ORDER_DB.get(order_id.upper())
    if not order:
        return f"Refund failed: order '{order_id}' not found."

    if order["status"] == "Processing":
        return (
            f"Refund failed: order {order_id.upper()} is still being processed.\n"
            "Please cancel the order first, then request a refund."
        )

    # Stream writer pattern — in production replace print() with:
    #   from langgraph.types import StreamWriter
    #   def process_refund_request(..., writer: StreamWriter) -> str:
    #       writer({"type": "progress", "step": step})
    for step in steps:
        print(f"  [STREAM] {step}")
        time.sleep(0.25)

    return "\n".join([
        f"Refund approved for order {order_id.upper()}:",
        f"  Item   : {order['item']}",
        f"  Amount : ₹{order['price']:,}",
        f"  Reason : {reason}",
        f"  ETA    : 3–5 business days.",
    ])


# ---------------------------------------------------------------------------
# TOOL 6 — State writer via Command
# Concept: tools can mutate the graph's short-term state, not just return
#          strings. Command(update={...}) writes fields back into the state
#          so subsequent tools and agent turns see the updated values.
#          Here we flip escalation_flag when feedback rating is below 3.
# ---------------------------------------------------------------------------

@tool(args_schema=FeedbackInput)
def submit_feedback(category: str, rating: int, comment: str, config: RunnableConfig):
    """
    Record customer feedback for this session. Automatically sets
    escalation_flag = True in the short-term state if rating is below 3,
    so the agent can proactively offer human support on the next turn.
    """
    customer_id = config.get("configurable", {}).get("customer_id", "UNKNOWN")
    customer    = CUSTOMER_DB.get(customer_id, {})
    name        = customer.get("name", "the customer")
    stars       = "★" * rating + "☆" * (5 - rating)
    escalate    = rating < 3

    note = (
        "  ⚠ Low rating — escalation_flag set to True in session state."
        if escalate else
        "  Positive feedback — no escalation needed."
    )

    summary = "\n".join([
        f"Feedback recorded for {name}:",
        f"  Category : {category}",
        f"  Rating   : {stars} ({rating}/5)",
        f"  Comment  : {comment}",
        note,
    ])

    # Command writes escalation_flag into the graph's short-term state.
    # The agent reads this automatically on its next reasoning step.
    return Command(update={"escalation_flag": escalate})


# ---------------------------------------------------------------------------
# Exported list — imported by agent.py
# ---------------------------------------------------------------------------

ALL_TOOLS = [
    check_order_status,       # Tool 1 — basic + Pydantic schema
    get_my_account,           # Tool 2 — context reader
    get_session_status,       # Tool 3 — state reader (structured fields)
    summarise_conversation,   # Tool 4 — state reader (message history)
    process_refund_request,   # Tool 5 — stream writer
    submit_feedback,          # Tool 6 — state writer via Command
]
