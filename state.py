"""
Shared state schema for the supervisor StateGraph.
"""
from typing import Annotated, List, Optional, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class NewAgentSpec(TypedDict, total=False):
    name: str
    description: str


class SupervisorState(TypedDict):
    # Full chat/tool-call history (LangGraph's reducer appends new messages).
    messages: Annotated[List[BaseMessage], add_messages]

    # The raw user task for this turn, pulled out of messages for convenience.
    task: str

    # "existing" | "create" -- set by router_node.
    route_decision: Optional[str]

    # Name of the sub-agent to execute (either matched, or the one just created).
    selected_agent: Optional[str]

    # Populated by router_node when route_decision == "create".
    new_agent_spec: Optional[NewAgentSpec]

    # Final text handed back to the user.
    final_output: Optional[str]
