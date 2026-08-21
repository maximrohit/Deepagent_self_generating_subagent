"""
Shared state schema for the supervisor StateGraph.
"""
from typing import Annotated, Any, Dict, List, Optional, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class NewAgentSpec(TypedDict, total=False):
    name: str
    description: str


class TodoItem(TypedDict, total=False):
    id: int
    description: str
    # Exact existing agent name when known; empty if a new component may be needed.
    assigned_agent: str
    # Optional flow_id hint from the flows manifest.
    flow_id: str
    # pending | running | done | blocked | unknown
    status: str
    result: str
    notes: str


class TopicAnalysis(TypedDict, total=False):
    """Initial research brief before planning TODOs."""
    entities: List[str]
    entity_types: List[str]
    domain: str
    problem_framing: str
    do_list: List[str]
    dont_list: List[str]
    matching_agents: List[str]
    matching_flows: List[str]
    reuse_decision: str
    create_needed: bool
    proposed_agent_name: str
    summary: str


class SupervisorState(TypedDict):
    # Full chat/tool-call history (LangGraph's reducer appends new messages).
    messages: Annotated[List[BaseMessage], add_messages]

    # The raw user task for this turn.
    task: str

    # Initial topic/domain/entity briefing (set before first plan).
    topic_analysis: Optional[TopicAnalysis]

    # Ordered TODO plan for the current validation iteration.
    todos: Optional[List[TodoItem]]

    # Accumulated per-todo results for the current iteration.
    todo_results: Optional[Dict[str, str]]

    # Draft answer built from the latest TODO execution (pre-validation).
    draft_output: Optional[str]

    # Running best draft across validation iterations.
    best_output: Optional[str]

    # 1-based validation iteration counter.
    validation_iteration: Optional[int]

    # Latest validator feedback (gaps to close).
    validation_gaps: Optional[List[str]]
    validation_verdict: Optional[str]  # PASS | FAIL

    # Legacy single-agent fields (kept for compatibility).
    route_decision: Optional[str]
    selected_agent: Optional[str]
    new_agent_spec: Optional[NewAgentSpec]

    # Final text handed back to the user.
    final_output: Optional[str]

    # Scratch for pipeline metadata.
    pipeline_meta: Optional[Dict[str, Any]]
