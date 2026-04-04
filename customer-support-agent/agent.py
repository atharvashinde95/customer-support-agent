"""
agent.py
--------
Assembles the LangGraph react agent from the pieces defined in other modules.
This file has one job: wire llm + tools + state_schema + prompt together.

Nothing business-logic lives here — keep it thin.
"""

from langgraph.prebuilt import create_react_agent

from config import SYSTEM_PROMPT, llm
from state import SupportAgentState
from tools import ALL_TOOLS

# ---------------------------------------------------------------------------
# Agent — LangGraph ReAct agent
#
# create_react_agent wires:
#   model        → the LLM that decides which tool to call and when
#   tools        → the callable functions the model can invoke
#   prompt       → the system prompt that shapes agent behaviour
#   state_schema → our custom short-term state (replaces the default AgentState)
# ---------------------------------------------------------------------------

agent = create_react_agent(
    model=llm,
    tools=ALL_TOOLS,
    prompt=SYSTEM_PROMPT,
    state_schema=SupportAgentState,
)
