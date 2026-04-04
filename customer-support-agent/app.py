"""
app.py
------
Streamlit UI for the Customer Support Agent.
All business logic lives in session.py / tools.py / agent.py — untouched.
This file only handles display, state management in st.session_state,
and routing user input to the Session object.

Run:
    streamlit run app.py
"""

import uuid

import streamlit as st
from langchain_core.messages import HumanMessage

from config import CUSTOMER_DB
from session import Session

# ---------------------------------------------------------------------------
# Page config — must be first Streamlit call
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Aria — Support Agent",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS — clean dark-toned chat UI
# ---------------------------------------------------------------------------

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Mono:wght@400;500&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 1.5rem; padding-bottom: 2rem; }

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #0f1117;
    border-right: 1px solid #1e2130;
}
section[data-testid="stSidebar"] * { color: #c9d1e0 !important; }
section[data-testid="stSidebar"] label {
    color: #6b7a99 !important;
    font-size: 0.72rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}

/* Chat bubbles */
.msg-user { display:flex; justify-content:flex-end; margin:0.6rem 0; }
.msg-aria  { display:flex; justify-content:flex-start; margin:0.6rem 0; }

.bubble-user {
    background: #2563eb;
    color: #fff;
    padding: 0.65rem 1rem;
    border-radius: 18px 18px 4px 18px;
    max-width: 72%;
    font-size: 0.92rem;
    line-height: 1.55;
}
.bubble-aria {
    background: #1e2130;
    color: #dde3f0;
    padding: 0.65rem 1rem;
    border-radius: 18px 18px 18px 4px;
    max-width: 78%;
    font-size: 0.92rem;
    line-height: 1.6;
    border: 1px solid #2a3045;
}
.avatar-aria {
    width:30px; height:30px; border-radius:50%;
    background: linear-gradient(135deg, #2563eb, #7c3aed);
    display:flex; align-items:center; justify-content:center;
    font-size:0.75rem; color:white; font-weight:600;
    margin-right:8px; flex-shrink:0; margin-top:2px;
}

/* Tool badge */
.tool-badge {
    display:inline-block;
    background:#0f1117;
    border:1px solid #2a3045;
    color:#6b7a99;
    font-family:'DM Mono',monospace;
    font-size:0.72rem;
    padding:2px 8px;
    border-radius:4px;
    margin:4px 2px;
}

/* Stream log */
.stream-log {
    background:#0a0c10;
    border:1px solid #1e2130;
    border-radius:8px;
    padding:0.5rem 0.75rem;
    font-family:'DM Mono',monospace;
    font-size:0.75rem;
    color:#4ade80;
    margin:4px 0 8px 0;
}

/* Status pills */
.pill { display:inline-block; padding:2px 10px; border-radius:100px; font-size:0.72rem; font-weight:500; }
.pill-ok   { background:#052e16; color:#4ade80; border:1px solid #166534; }
.pill-warn { background:#431407; color:#fb923c; border:1px solid #9a3412; }
.pill-blue { background:#0c1a3a; color:#60a5fa; border:1px solid #1d4ed8; }

/* Input */
.stTextInput > div > div > input {
    background:#1e2130 !important;
    border:1px solid #2a3045 !important;
    color:#dde3f0 !important;
    border-radius:10px !important;
    font-family:'DM Sans',sans-serif !important;
}
.stTextInput > div > div > input:focus {
    border-color:#2563eb !important;
    box-shadow:0 0 0 2px rgba(37,99,235,0.25) !important;
}
.stButton > button {
    background:#2563eb !important;
    color:white !important;
    border:none !important;
    border-radius:10px !important;
    font-weight:500 !important;
    padding:0.5rem 1.4rem !important;
}
.stButton > button:hover { background:#1d4ed8 !important; }
hr { border-color:#1e2130 !important; }
.stSelectbox > div > div {
    background:#1e2130 !important;
    border-color:#2a3045 !important;
    color:#dde3f0 !important;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TOOL_LABELS = {
    "check_order_status":        "📦 check_order_status",
    "check_order_status_simple": "📦 check_order_status",
    "get_my_account":            "👤 get_my_account",
    "get_session_status":        "📊 get_session_status",
    "summarise_conversation":    "📝 summarise_conversation",
    "process_refund_request":    "💸 process_refund_request",
    "submit_feedback":           "⭐ submit_feedback",
}

REFUND_STREAM_STEPS = [
    "Validating order...",
    "Checking 30-day eligibility window...",
    "Calculating refund amount...",
    "Initiating bank transfer...",
    "Queuing confirmation email...",
]

QUICK_ACTIONS = {
    "My account":     "Can you pull up my account details?",
    "Order ORD-001":  "Check status of ORD-001, include the price.",
    "Order ORD-002":  "What is the status of ORD-002?",
    "Refund ORD-003": "I want a refund for ORD-003 — the item arrived damaged.",
    "Session status": "How is this session looking so far?",
    "Recap":          "Can you recap everything we discussed today?",
    "Bad feedback":   "Feedback: category delivery, rating 2, package was two weeks late.",
    "Good feedback":  "Feedback: category support, rating 5, Aria was very helpful!",
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_tool_badges(history) -> list[str]:
    """Pull unique tool names from the message history."""
    seen, badges = set(), []
    for m in history:
        if hasattr(m, "tool_calls") and m.tool_calls:
            for tc in m.tool_calls:
                name = tc.get("name", "")
                label = TOOL_LABELS.get(name, f"🔧 {name}")
                if label not in seen:
                    seen.add(label)
                    badges.append(label)
    return badges


def render_message(role: str, content: str, badges: list = None, stream: list = None):
    """Render one chat bubble."""
    if role == "user":
        st.markdown(
            f'<div class="msg-user"><div class="bubble-user">{content}</div></div>',
            unsafe_allow_html=True,
        )
        return

    badges_html = ""
    if badges:
        badges_html = "<div style='margin-bottom:6px'>" + "".join(
            f'<span class="tool-badge">{b}</span>' for b in badges
        ) + "</div>"

    stream_html = ""
    if stream:
        rows = "".join(f"<div>▸ {s}</div>" for s in stream)
        stream_html = f'<div class="stream-log">{rows}</div>'

    escaped = (
        content
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace("\n", "<br>")
    )

    st.markdown(f"""
    <div class="msg-aria">
        <div class="avatar-aria">AR</div>
        <div style="flex:1">
            {badges_html}
            <div class="bubble-aria">{escaped}</div>
            {stream_html}
        </div>
    </div>""", unsafe_allow_html=True)


def new_session(customer_id: str):
    """Reset all session state."""
    st.session_state.session     = Session(
        customer_id=customer_id,
        session_id="sess-" + uuid.uuid4().hex[:8],
    )
    st.session_state.chat        = []
    st.session_state.customer_id = customer_id
    st.session_state.escalated   = False
    st.session_state.turn        = 0

# ---------------------------------------------------------------------------
# Initialise on first load
# ---------------------------------------------------------------------------

if "session" not in st.session_state:
    new_session("CUST-101")

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("### 💬 Aria Support")
    st.markdown("<hr>", unsafe_allow_html=True)

    # Customer selector
    customer_options = {k: f"{v['name']} ({k})" for k, v in CUSTOMER_DB.items()}
    selected_cid = st.selectbox(
        "LOGGED IN AS",
        options=list(customer_options.keys()),
        format_func=lambda k: customer_options[k],
        index=list(customer_options.keys()).index(st.session_state.customer_id),
    )
    if selected_cid != st.session_state.customer_id:
        new_session(selected_cid)
        st.rerun()

    st.markdown("<hr>", unsafe_allow_html=True)

    # Customer profile
    cust = CUSTOMER_DB[st.session_state.customer_id]
    st.markdown(f"**Name** &nbsp;&nbsp; {cust['name']}", unsafe_allow_html=True)
    st.markdown(f"**Tier** &nbsp;&nbsp;&nbsp; {cust['tier']}", unsafe_allow_html=True)
    st.markdown(f"**Email** &nbsp; {cust['email']}", unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    # Live stats
    st.markdown("**SESSION STATUS**")
    st.markdown(
        f'<span class="pill pill-blue">Turns: {st.session_state.turn}</span>',
        unsafe_allow_html=True,
    )
    st.markdown(" ", unsafe_allow_html=True)
    if st.session_state.escalated:
        st.markdown('<span class="pill pill-warn">⚠ Escalation flagged</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="pill pill-ok">● Active</span>', unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    # Quick actions
    st.markdown("**QUICK ACTIONS**")
    for label, msg in QUICK_ACTIONS.items():
        if st.button(label, use_container_width=True, key=f"qa_{label}"):
            st.session_state.pending = msg
            st.rerun()

    st.markdown("<hr>", unsafe_allow_html=True)
    if st.button("🗑 New session", use_container_width=True):
        new_session(st.session_state.customer_id)
        st.rerun()

# ---------------------------------------------------------------------------
# Main panel
# ---------------------------------------------------------------------------

st.markdown("""
<div style="margin-bottom:1.2rem;">
    <span style="font-size:1.5rem;font-weight:600;color:#dde3f0;">Aria</span>
    <span style="font-size:0.85rem;color:#6b7a99;margin-left:10px;">
        Customer Support Agent &nbsp;·&nbsp; Short-term memory &nbsp;·&nbsp; LangChain + LangGraph
    </span>
</div>
""", unsafe_allow_html=True)

# Chat history
if not st.session_state.chat:
    st.markdown("""
    <div style="text-align:center;color:#3a4260;padding:3rem 0;font-size:0.9rem;">
        Start the conversation — pick a quick action or type below.
    </div>""", unsafe_allow_html=True)
else:
    for msg in st.session_state.chat:
        render_message(
            role    = msg["role"],
            content = msg["content"],
            badges  = msg.get("badges"),
            stream  = msg.get("stream"),
        )

# ---------------------------------------------------------------------------
# Input bar
# ---------------------------------------------------------------------------

st.markdown('<div style="height:0.8rem"></div>', unsafe_allow_html=True)
col_in, col_btn = st.columns([6, 1])

with col_in:
    user_input = st.text_input(
        label="msg",
        label_visibility="collapsed",
        placeholder="Type a message...",
        key="input_box",
        value=st.session_state.pop("pending", ""),
    )
with col_btn:
    send = st.button("Send", use_container_width=True)

# ---------------------------------------------------------------------------
# Handle send
# ---------------------------------------------------------------------------

if (send or user_input) and user_input.strip():
    message = user_input.strip()

    # Push user bubble
    st.session_state.chat.append({"role": "user", "content": message})

    # Detect refund to show stream steps in UI
    is_refund   = any(k in message.lower() for k in ["refund", "return", "money back"])
    stream_show = REFUND_STREAM_STEPS if is_refund else None

    with st.spinner("Aria is thinking..."):
        try:
            # ── Only line that touches business logic ──
            response = st.session_state.session.send(message)

            badges = get_tool_badges(st.session_state.session.history)

            if st.session_state.session.escalation_flag:
                st.session_state.escalated = True

            st.session_state.turn += 1

            st.session_state.chat.append({
                "role":    "aria",
                "content": response,
                "badges":  badges,
                "stream":  stream_show,
            })

        except Exception as exc:
            st.session_state.chat.append({
                "role":    "aria",
                "content": f"⚠ Error: {exc}",
                "badges":  [],
                "stream":  None,
            })

    st.rerun()
