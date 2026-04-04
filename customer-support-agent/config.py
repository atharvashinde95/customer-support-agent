"""
config.py
---------
Capgemini Generative Engine client setup and all shared constants.
Every other module imports from here — change the model or key in one place.
"""

from langchain_openai import ChatOpenAI

# ---------------------------------------------------------------------------
# LLM — Capgemini Generative Engine (OpenAI-compatible endpoint)
# ---------------------------------------------------------------------------

llm = ChatOpenAI(
    base_url="https://openai.generative.engine.capgemini.com/v1",
    api_key="your_api_key",            # <-- replace with your actual key
    model="us.anthropic.claude-3-5-sonnet-20241022-v2:0",
    temperature=0,
    streaming=True,
)

# ---------------------------------------------------------------------------
# Fake databases — stand-ins for real backend APIs / DB calls
# ---------------------------------------------------------------------------

ORDER_DB: dict[str, dict] = {
    "ORD-001": {"item": "Laptop",  "status": "Shipped",    "eta": "2 days", "price": 85000},
    "ORD-002": {"item": "Mouse",   "status": "Processing", "eta": "5 days", "price": 1500},
    "ORD-003": {"item": "Desk",    "status": "Delivered",  "eta": "Done",   "price": 12000},
}

CUSTOMER_DB: dict[str, dict] = {
    "CUST-101": {"name": "Priya Sharma", "tier": "Gold",     "email": "priya@example.com", "wallet": 5000},
    "CUST-202": {"name": "Rahul Mehta",  "tier": "Standard", "email": "rahul@example.com", "wallet": 800},
    "CUST-303": {"name": "Aisha Khan",   "tier": "Platinum", "email": "aisha@example.com", "wallet": 15000},
}

# ---------------------------------------------------------------------------
# Agent system prompt — centralised so agent.py stays clean
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are Aria, a customer support agent for an e-commerce platform.

Your tools and when to use them:
- get_my_account              → ALWAYS call this first to greet the customer by name
- check_order_status          → look up any order; confirm details before a refund
- get_session_status          → check conversation depth and escalation flag
- summarise_conversation      → recap everything discussed this session
- process_refund_request      → process refunds with live progress updates
- submit_feedback             → record feedback; auto-sets escalation flag if rating < 3

Rules:
1. Greet by name on your very first turn (call get_my_account).
2. Before processing a refund, always confirm the order via check_order_status.
3. After recording feedback, call get_session_status — if escalation_flag is True,
   proactively offer to connect the customer with a human agent.
4. Keep responses concise, warm, and professional.
"""
