"""
create_sub_agent_tool: the tool the supervisor calls when no existing
sub-agent's description/capabilities match the task.

Flow:
  1. Ask CODEGEN_MODEL to write a sub-agent module conforming to the required
     contract (AGENT_NAME / AGENT_DESCRIPTION / AGENT_CAPABILITIES / build_agent()).
  2. Sanity-check the generated code (syntax + contract) before persisting it.
  3. Write it to sub_agents/<slug>.py (this is the persistence step -- the file
     survives process restarts).
  4. Hand off to Registry.register_from_file() for dynamic import/compile and
     live in-memory registration.
"""
from __future__ import annotations

import ast
import re
from pathlib import Path

from langchain_core.tools import tool

from config import SUB_AGENTS_DIR
from registry import SubAgentRegistry
from utility.llm import Model

SUB_AGENT_TEMPLATE_INSTRUCTIONS = '''\
You are writing a new sub-agent module for a LangGraph multi-agent system.

The module MUST define exactly these top-level members and nothing else that
would conflict with them:

    AGENT_NAME: str
        A short, unique, snake_case identifier (e.g. "sql_analyst_agent").

    AGENT_DESCRIPTION: str
        A detailed, capability-level paragraph describing what this agent does,
        what kinds of tasks it should be routed, and its boundaries/limits.
        This is read by a router LLM to decide whether to route future tasks
        here, so it must be specific and unambiguous -- do not rely on the name.

    AGENT_CAPABILITIES: list[str]
        4-8 concise bullet-style strings enumerating concrete capabilities
        (e.g. "Writes and explains SQL SELECT/JOIN/window-function queries").

    def build_agent(model_name: str = "local", extra_tools=None):
        Constructs and returns a compiled, invokable LangGraph agent for this
        domain (use langgraph.prebuilt.create_react_agent with an appropriate
        system prompt and any domain tools you define in this same file).
        The returned object must support .invoke({{"messages": [...]}}).
        Import the shared local LLM with:
            from utility.llm import Model
        and pass Model to create_react_agent (ignore model_name).

        MUST accept optional extra_tools and merge them into the agent tool
        list. The runtime injects topic-relevant Tavily search tools here
        (finance_web_search / general_web_search / news_web_search). Example:

            from utility.llm import Model
            from utility.tavily_tools import finance_search_tool, merge_tools
            from deepagents import create_deep_agent

            tools = merge_tools([finance_search_tool], extra_tools)
            # If no domain tools: tools = merge_tools([], extra_tools)
            return create_deep_agent(
                model=Model,
                tools=tools,
                system_prompt="...",
                name=AGENT_NAME,
            )

        CRITICAL — Model vs tools (do not confuse these):
        - `Model` from utility.llm is the LANGUAGE MODEL. Pass it only as
          model=Model (or the first positional arg to create_react_agent).
        - NEVER put Model inside the tools list. tools may only contain
          BaseTool instances / @tool callables (e.g. finance_search_tool),
          never ChatOllama / Model.
        - Wrong: merge_tools([Model], extra_tools)
        - Right: merge_tools([finance_search_tool], extra_tools)
                 or merge_tools([], extra_tools)

        If this agent's domain needs live web data, also mention research /
        finance / news in AGENT_DESCRIPTION and AGENT_CAPABILITIES so the
        correct Tavily topic tools are selected automatically.

Rules:
- Pure Python, self-contained in one file. Only import from: langgraph,
  langchain_core, langchain, deepagents, utility.llm, utility.tavily_tools,
  and the Python standard library.
- Do not call init_chat_model or connect to cloud providers; always use
  Model from utility.llm as the model=, never as a tool.
- Prefer create_deep_agent(model=Model, tools=..., system_prompt=..., name=...)
  over create_react_agent when possible.
- Do not hard-code Tavily API keys; use the shared tools from
  utility.tavily_tools when web search is needed.
- Do not perform any file writes or shell execution inside tools unless the
  task explicitly requires that kind of tool -- prefer reasoning / text /
  computation tools, plus injected Tavily tools when relevant.
- No explanation text outside the code. Return ONLY a single python code
  block.
'''


def _extract_code_block(text: str) -> str:
    match = re.search(r"```(?:python)?\s*\n(.*?)```", text, re.DOTALL)
    if not match:
        raise ValueError("Codegen model did not return a fenced python code block.")
    return match.group(1).strip()


def _validate_contract(code: str) -> None:
    """Syntax + presence checks before we ever write/execute the file for real."""
    tree = ast.parse(code)  # raises SyntaxError if invalid
    top_level_names = set()
    has_build_agent = False
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    top_level_names.add(target.id)
        elif isinstance(node, ast.FunctionDef) and node.name == "build_agent":
            has_build_agent = True

    required = {"AGENT_NAME", "AGENT_DESCRIPTION", "AGENT_CAPABILITIES"}
    missing = required - top_level_names
    if missing:
        raise ValueError(f"Generated sub-agent is missing required assignments: {missing}")
    if not has_build_agent:
        raise ValueError("Generated sub-agent is missing a build_agent() function.")

    # Catch the common codegen mistake: putting Model (the LLM) into tools=.
    if re.search(
        r"merge_tools\(\s*\[\s*Model\s*\]|tools\s*=\s*\[\s*Model\s*\]|tools\s*=\s*Model\b",
        code,
    ):
        raise ValueError(
            "Generated sub-agent incorrectly puts Model in the tools list. "
            "Model is the LLM (pass as model=Model); tools must be BaseTool "
            "instances such as finance_search_tool, or merge_tools([], extra_tools)."
        )

def _slugify(name: str) -> str:
    slug = re.sub(r"[^a-z0-9_]+", "_", name.strip().lower())
    return re.sub(r"_+", "_", slug).strip("_") or "sub_agent"


def make_create_sub_agent_tool(registry: SubAgentRegistry):
    """
    Factory so the tool closes over the *live* registry instance used by the
    running supervisor graph -- registration is therefore immediately visible
    to subsequent router turns in the same process.
    """
    codegen_llm = Model

    @tool
    def create_sub_agent_tool(task_description: str, proposed_name: str) -> str:
        """
        Generate, persist, and hot-register a brand-new specialized LangGraph
        sub-agent capable of handling `task_description`.

        Call this ONLY after confirming no existing sub-agent's description/
        capabilities cover the task.

        Args:
            task_description: A detailed description of the capability gap --
                what the new agent needs to be able to do. This is used both
                to write the agent's code and its own routable description.
            proposed_name: A short human-readable working name/slug hint,
                e.g. "sql analyst" -> may be normalized internally.

        Returns:
            A confirmation string including the final registered agent name
            and description, so the caller can immediately route to it.
        """
        slug = _slugify(proposed_name)
        target_path = SUB_AGENTS_DIR / f"{slug}.py"
        if target_path.exists():
            # Avoid clobbering; make it unique instead of failing the turn.
            i = 2
            while (SUB_AGENTS_DIR / f"{slug}_{i}.py").exists():
                i += 1
            target_path = SUB_AGENTS_DIR / f"{slug}_{i}.py"

        prompt = (
            SUB_AGENT_TEMPLATE_INSTRUCTIONS
            + f"\n\nTask this agent must handle:\n{task_description}\n"
            + f"\nSuggested AGENT_NAME (adjust for valid snake_case if needed): {slug}\n"
        )

        response = codegen_llm.invoke(prompt)
        code = _extract_code_block(response.content)
        _validate_contract(code)  # raises -> tool error surfaces to supervisor, no file written

        target_path.write_text(code)

        meta = registry.register_from_file(target_path)

        return (
            f"Created and registered new sub-agent '{meta.name}' at {target_path.name}.\n"
            f"Description: {meta.description}\n"
            f"Capabilities: {meta.capabilities}\n"
            f"You can now route this task (and future similar ones) to '{meta.name}'."
        )

    return create_sub_agent_tool
