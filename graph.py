"""
Builds the supervisor StateGraph:

    START -> router_node -> [existing_agent_node | create_and_register_node]
                                                            |
                                                            v
                                                     execute_new_agent_node
                                    (both execution paths) -> END

router_node is the "Dynamic Router & Evaluator": it is given the FULL
description + capabilities of every currently-registered sub-agent (via
registry.describe_for_router()) and must justify its match against that
text, not against agent names.
"""
from __future__ import annotations

from typing import Literal, Optional

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from code_generator import make_create_sub_agent_tool
from registry import SubAgentRegistry
from state import SupervisorState
from utility.llm import Model

ROUTER_SYSTEM_PROMPT = """You are the routing/evaluation layer of a self-evolving \
multi-agent system.

You will be shown the FULL descriptions and capability lists of every \
currently registered sub-agent. You must decide, based on the SUBSTANCE of \
those descriptions and capabilities (never based on guessing from an agent's \
name alone), whether one of them can competently handle the user's task.

- If an existing sub-agent's description/capabilities genuinely cover the \
task, choose "existing" and return its exact name.
- If none of them do -- even partially or by loose analogy -- choose \
"create" and propose a name + a precise description of the capability gap \
that a brand-new sub-agent must fill.

Be conservative about reusing an existing agent: a superficial keyword \
overlap is not enough, the agent's stated capabilities must actually cover \
what the task requires.

Registered sub-agents:
{registry_block}
"""


class RouteDecision(BaseModel):
    action: Literal["existing", "create"] = Field(
        description="'existing' if a registered sub-agent's description/capabilities "
        "genuinely cover the task, else 'create'."
    )
    agent_name: str = Field(
        description="If action=='existing': the exact matched agent's name. "
        "If action=='create': a short proposed snake_case name for the new agent."
    )
    justification: str = Field(
        description="Why this decision follows from the agents' descriptions/capabilities."
    )
    new_agent_task_description: Optional[str] = Field(
        default=None,
        description="Only set when action=='create': a precise description of what "
        "the new sub-agent needs to be able to do.",
    )


def build_supervisor_graph(registry: SubAgentRegistry):
    router_llm = Model.with_structured_output(RouteDecision)
    create_sub_agent_tool = make_create_sub_agent_tool(registry)

    # ------------------------------------------------------------------ #
    # Nodes
    # ------------------------------------------------------------------ #
    def router_node(state: SupervisorState) -> dict:
        task = state["task"]
        system = ROUTER_SYSTEM_PROMPT.format(registry_block=registry.describe_for_router())
        decision: RouteDecision = router_llm.invoke(
            [SystemMessage(content=system), HumanMessage(content=task)]
        )

        update = {
            "route_decision": decision.action,
            "messages": [AIMessage(content=f"[router] {decision.justification}")],
        }
        if decision.action == "existing":
            update["selected_agent"] = decision.agent_name
        else:
            update["new_agent_spec"] = {
                "name": decision.agent_name,
                "description": decision.new_agent_task_description or task,
            }
        return update

    def route_after_router(state: SupervisorState) -> Literal["existing_agent_node", "create_and_register_node"]:
        return "existing_agent_node" if state["route_decision"] == "existing" else "create_and_register_node"

    def existing_agent_node(state: SupervisorState) -> dict:
        agent_name = state["selected_agent"]
        compiled = registry.get_compiled(agent_name)
        if compiled is None:
            raise RuntimeError(f"Router selected unknown/uncompiled agent '{agent_name}'")
        result = compiled.invoke({"messages": [HumanMessage(content=state["task"])]})
        output_text = result["messages"][-1].content
        return {
            "final_output": output_text,
            "messages": [AIMessage(content=output_text, name=agent_name)],
        }

    def create_and_register_node(state: SupervisorState) -> dict:
        spec = state["new_agent_spec"]
        # Directly invoke the tool function (supervisor "calling" create_sub_agent_tool).
        result_text = create_sub_agent_tool.invoke(
            {"task_description": spec["description"], "proposed_name": spec["name"]}
        )
        # The tool registered the agent under its *final* normalized name; recover it.
        # (Simplest robust approach: re-derive from registry via the slug we proposed.)
        matched = None
        for meta in registry.all_metadata():
            if meta.name.startswith(spec["name"].strip().lower().replace(" ", "_")):
                matched = meta.name
        return {
            "selected_agent": matched,
            "messages": [AIMessage(content=f"[codegen] {result_text}")],
        }

    def execute_new_agent_node(state: SupervisorState) -> dict:
        return existing_agent_node(state)  # identical execution path once registered

    # ------------------------------------------------------------------ #
    # Graph assembly
    # ------------------------------------------------------------------ #
    graph = StateGraph(SupervisorState)
    graph.add_node("router_node", router_node)
    graph.add_node("existing_agent_node", existing_agent_node)
    graph.add_node("create_and_register_node", create_and_register_node)
    graph.add_node("execute_new_agent_node", execute_new_agent_node)

    graph.add_edge(START, "router_node")
    graph.add_conditional_edges(
        "router_node",
        route_after_router,
        {"existing_agent_node": "existing_agent_node", "create_and_register_node": "create_and_register_node"},
    )
    graph.add_edge("existing_agent_node", END)
    graph.add_edge("create_and_register_node", "execute_new_agent_node")
    graph.add_edge("execute_new_agent_node", END)

    return graph.compile()
