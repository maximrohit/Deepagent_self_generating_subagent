"""
Sub-agent creation pipeline (functional stages):

  1. design_capability_spec (GapAnalysisSpec)
       Identify existing agents that already cover part of the task, and the
       SINGLE missing functional component to create (if any).
  1b. research_prompt_brief (PromptResearchBrief)
       Web-research the MACRO domain + entity *attributes* from topic analysis
       (not the one-shot user question framing) to draft a holistic system prompt.
  2. generate_sub_agent_code
       Write (or revise) ONLY the missing-component .py module, embedding the
       researched prompt brief.
  3. validate_contract
       Static syntax + contract checks (cheap, no LLM).
  4. critique_sub_agent
       LLM critic reviews missing-component code vs reusable scope.
  5. self-correct loop (max SUBAGENT_CRITIC_MAX_ITERS)
       On FAIL: feed critic flaws back into generate_sub_agent_code and retry.
  6. persist_and_register
       Write to disk only after PASS (or max-iters best effort), then hot-register.
  7. register_composed_flow
       Upsert flows/manifest.json with the reuse + missing composition and the
       problem types it newly unlocks.

create_sub_agent_tool is the public entrypoint used by the supervisor graph.
"""
from __future__ import annotations

import ast
import json
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from config import SUB_AGENTS_DIR, SUBAGENT_CRITIC_MAX_ITERS
from prompt_library.create_research import SYNTHESIZE_AGENT_PROMPT_RESEARCH
from registry import SubAgentRegistry
from utility.domains import normalize_domain, proposed_agent_for_domain
from utility.guardrails import MACRO_DOMAIN_GUIDANCE
from utility.llm import Model
from utility.tavily_tools import _run_topic_search

if TYPE_CHECKING:
    from flow_manifest import FlowManifest

# --------------------------------------------------------------------------- #
# Shared codegen / critic prompts
# --------------------------------------------------------------------------- #

SUB_AGENT_TEMPLATE_INSTRUCTIONS = '''\
You are writing a new sub-agent module for a LangGraph multi-agent system.

The module MUST define exactly these top-level members and nothing else that
would conflict with them:

    AGENT_NAME: str
        A short, unique, snake_case MACRO domain identifier
        (e.g. "comics_lore_agent", "home_strength_training_agent").
        NEVER name after a single instance from the user question
        (wrong: "batman_agent"; right: "comics_lore_agent").

    AGENT_DESCRIPTION: str
        A detailed, capability-level paragraph describing the durable DOMAIN
        this agent owns, what kinds of peer tasks it should be routed, and
        boundaries/limits. Must cover peer instances in the same domain
        (e.g. Batman AND other comics heroes), not only the current ask.
        Explicitly state what this agent does NOT do (to avoid overlap).

    AGENT_CAPABILITIES: list[str]
        4-8 concise bullet-style strings at the DOMAIN level
        (e.g. "Explains comics continuity, characters, and story arcs across
        major publishers"), not micro one-offs.

    def build_agent(model_name: str = "local", extra_tools=None):
        Constructs and returns a compiled, invokable LangGraph agent for this
        domain. Prefer deepagents.create_deep_agent. The returned object must
        support .invoke({{"messages": [...]}}).

        Example:

            from utility.llm import Model
            from utility.tavily_tools import (
                TIME_RANGE_GUIDANCE,
                finance_search_tool,
                merge_tools,
            )
            from deepagents import create_deep_agent

            tools = merge_tools([finance_search_tool], extra_tools)
            # If no domain tools: tools = merge_tools([], extra_tools)
            return create_deep_agent(
                model=Model,
                tools=tools,
                system_prompt="..." + "\\n\\n" + TIME_RANGE_GUIDANCE,
                name=AGENT_NAME,
            )

        CRITICAL — Model vs tools (do not confuse these):
        - `Model` from utility.llm is the LANGUAGE MODEL. Pass it only as
          model=Model.
        - NEVER put Model inside the tools list.
        - Wrong: merge_tools([Model], extra_tools)
        - Right: merge_tools([finance_search_tool], extra_tools)
                 or merge_tools([], extra_tools)

        SEARCH time_range (when using Tavily tools):
        - finance_web_search / general_web_search / news_web_search / web_search
          all take time_range as a TOOL ARGUMENT: "day", "week", "month", or
          "year" based on the user-requested horizon.
        - The SAME sub-agent must answer a 1-week question and a 1-month
          question by changing time_range (not by refusing or hard-coding one
          lookback). If both are asked, call search twice and compare.
        - Include TIME_RANGE_GUIDANCE from utility.tavily_tools in the
          system_prompt whenever search tools are attached.
        - Also include HONESTY_AND_GEOGRAPHY_GUIDANCE from
          utility.guardrails: if evidence is missing or from the wrong market,
          the agent must say it does not know (never invent tickers).
        - Do NOT hard-code a single time_range for all queries.
        - Map user language: "1 week"→week, "1 month"→month, "today"→day,
          "this year"/structural→year.

Rules:
- AGENT_NAME / DESCRIPTION must be MACRO-domain (comics_lore_agent), never a
  micro proper-noun from the current question (batman_agent). Peer instances
  in the same domain must be in-scope for the same agent.
- ALWAYS import the shared LLM as: from utility.llm import Model
  Never: from langchain_core import Model / from langchain import Model.
- Pure Python, self-contained in one file. Only import from: langgraph,
  langchain_core, langchain, deepagents, utility.llm, utility.tavily_tools,
  and the Python standard library.
- ALWAYS import create_deep_agent from deepagents, NEVER from langgraph:
    Correct:   from deepagents import create_deep_agent
    Incorrect: from langgraph import create_deep_agent
- Prefer create_deep_agent(model=Model, tools=..., system_prompt=..., name=...).
- When a HOLISTIC PROMPT RESEARCH BRIEF is provided, the system_prompt MUST
  be built primarily from its SYSTEM_PROMPT_CORE (domain + entity attributes
  + peer coverage). Do NOT overfit the prompt to one user question.
- Do not hard-code API keys.
- No explanation text outside the code. Return ONLY a single python code block.
'''

SPEC_DESIGN_PROMPT = """You decompose a user capability gap into:
(A) parts ALREADY solvable by existing sub-agents, and
(B) the SINGLE missing functional component that must be newly created.

You will receive:
- the user's capability gap / task
- a proposed name hint
- the FULL list of currently registered sub-agents
- the ACTIVE multi-agent flows catalog (if any)

""" + MACRO_DOMAIN_GUIDANCE + """

Return GapAnalysisSpec:

1) reusable_components:
   Existing agents that already cover a DISTINCT part of the full task.
   Each entry needs: agent (exact name), role (functional role id),
   function (what that agent does in THIS workflow), covers (which slice of
   the user task it already handles).
   Prefer reuse aggressively. Do NOT reinvent market_analyst / report_reviewer /
   investment_strategist / primary_deep_agent / etc. when they already fit.

2) missing_component (the ONLY thing to create as a new sub-agent):
   - agent_name: unique snake_case MACRO domain name (comics_lore_agent, NOT batman_agent)
   - role: functional role id for the new piece
   - purpose: one paragraph of the durable DOMAIN this agent owns (peers included)
   - capabilities: 3-6 bullets at domain level (not one character/product)
   - function: one-line stage function for the flows manifest
   - out_of_scope: work that stays with reusable/existing agents
   - why_not_reuse: why no existing agent covers THIS missing domain

3) composed flow (full problem solved by reuse + new component):
   - composed_flow_id, composed_flow_name, composed_flow_description
   - problem_types: problem classes newly unlocked by this MACRO domain
   - example_queries: 2-4 example queries across peer instances in the domain
     (e.g. Batman AND Superman AND Wonder Woman — not only the current ask)
   - flow_stages: ordered stages spanning reusable agents + the new agent.
     For the new stage set agent to the missing_component.agent_name and
     is_new=true. For reused stages is_new=false and agent must be an
     existing agent name.

Rules:
- Create ONLY the missing functional MACRO niche — never a micro one-off.
- If existing agents already cover the FULL task, set missing_component to a
  clearly marked no-op by using agent_name="__none__" and empty capabilities,
  and still return the composed flow that uses only existing agents.
- Never claim capabilities already listed under another agent in the new
  component.
- Prefer extending an existing multi-agent flow over inventing an unrelated
  generalist.
- primary_deep_agent is GENERIC research only. It does NOT count as covering
  a missing MACRO domain (comics, fitness, nutrition, etc.). Still create the
  domain specialist when that domain is absent from the registry.
- Geography/domain reuse (critical):
  * Indian equity / NSE / BSE / intraday India trade → MUST reuse
    indian_stock_trader_3m and/or market_analyst. Do NOT invent
    market_context_agent or similar overlaps.
  * US equity trade → reuse us_stock_trader_3m.
  * Report critique → reuse report_reviewer.
  * CIO multi-horizon verdict → reuse investment_strategist.
- Overlap with market_analyst / investment_strategist (same finance niche) is
  grounds to set missing_component.agent_name="__none__". Overlap with
  primary_deep_agent alone is NOT grounds for __none__.
- Reject micro names: if the proposed name is a single franchise/character/
  company from the user question, rename to the parent domain before returning.
- composed_flow_id must be a durable domain flow id (comics_versus_analysis),
  never a micro fight title (supercommandodhruv_vs_batman_agent).
- flow_stages must ONLY reference real existing agent names plus the new
  missing_component.agent_name. Never invent finance stages for a comics task.
"""

CRITIC_PROMPT = """You are a STRICT sub-agent creation critic. Your default is FAIL.

You protect the registry from MICRO one-offs and SAME-DOMAIN specialist overlap.
You do NOT block legitimate new MACRO domain specialists.

""" + MACRO_DOMAIN_GUIDANCE + """

Review the proposed sub-agent Python module against:
1. Contract correctness (AGENT_NAME / DESCRIPTION / CAPABILITIES / build_agent).
2. Model-vs-tools mistakes (Model must never appear in tools=).
3. SAME-DOMAIN specialist overlap ONLY:
   - FAIL if the new agent duplicates an EXISTING specialist in the SAME domain
     (e.g. another Indian equity trader when indian_stock_trader_3m already exists).
   - Do NOT FAIL merely because primary_deep_agent can web-search generally.
     primary_deep_agent is a GENERIC fallback, not a domain specialist.
     A comics / fitness / nutrition MACRO agent is allowed and preferred even
     though primary can search those topics.
   - Do NOT invent overlaps with agents that are not in the registry list.
   - Do NOT claim comics overlaps with market_analyst / stock traders.
4. Faithfulness to GapAnalysisSpec.missing_component ONLY.
5. Code quality: create_deep_agent MUST be imported from deepagents.
6. Geography: an "India" niche agent must not claim US/LSE/global generalist scope.
7. MACRO vs MICRO:
   - FAIL only for instance locks (batman_agent, dhruv_only_agent).
   - comics_lore_agent / home_strength_training_agent ARE MACRO — PASS them
     when the domain is missing from the registry.

Return CriticVerdict:
- verdict: PASS when the niche is a missing MACRO domain (even if primary exists)
- flaws: concrete, actionable list (required on FAIL)
- overlap_warnings: ONLY real same-domain specialist collisions
- summary: one-sentence status

If the right fix is "do not create; reuse existing SAME-DOMAIN specialist X", say so.
If the only alternative is primary_deep_agent, that is NOT a reason to FAIL — PASS.
"""


# --------------------------------------------------------------------------- #
# Structured LLM outputs
# --------------------------------------------------------------------------- #

class ReusableComponent(BaseModel):
    agent: str = Field(description="Exact existing sub-agent name to reuse.")
    role: str = Field(description="Functional role id in the composed flow.")
    function: str = Field(description="What this agent does in this workflow.")
    covers: str = Field(description="Which slice of the user task this already handles.")


class FlowStageSpec(BaseModel):
    order: int = Field(description="1-based stage order in the composed pipeline.")
    role: str = Field(description="Functional role id.")
    agent: str = Field(description="Existing agent name, or the new component's agent_name.")
    function: str = Field(description="Stage responsibility.")
    is_new: bool = Field(
        default=False,
        description="True only for the newly created missing component stage.",
    )


class MissingComponentSpec(BaseModel):
    agent_name: str = Field(
        description=(
            "Unique snake_case MACRO domain name for the NEW component "
            "(e.g. comics_lore_agent, NOT batman_agent), or '__none__' if nothing to create."
        )
    )
    role: str = Field(description="Functional role id for the missing component.")
    purpose: str = Field(
        description="Durable DOMAIN this agent owns (must cover peer instances, not one entity)."
    )
    capabilities: List[str] = Field(
        default_factory=list,
        description="3-6 domain-level capability bullets (peers included), not one character/product.",
    )
    function: str = Field(
        default="",
        description="One-line stage function for the flows manifest.",
    )
    out_of_scope: List[str] = Field(
        default_factory=list,
        description="Work that must remain with reusable/existing agents.",
    )
    why_not_reuse: str = Field(
        default="",
        description="Why no existing agent covers this missing slice.",
    )


class GapAnalysisSpec(BaseModel):
    reusable_components: List[ReusableComponent] = Field(
        default_factory=list,
        description="Existing agents that already cover part of the task.",
    )
    missing_component: MissingComponentSpec = Field(
        description="The only functional niche to create as a new sub-agent."
    )
    composed_flow_id: str = Field(description="snake_case id for the composed flow.")
    composed_flow_name: str = Field(description="Human-readable flow name.")
    composed_flow_description: str = Field(
        description="What the full reuse+new composition achieves."
    )
    problem_types: List[str] = Field(
        description="Problem classes newly unlocked by this composition."
    )
    example_queries: List[str] = Field(
        description="2-4 example queries the composed flow can handle."
    )
    flow_stages: List[FlowStageSpec] = Field(
        description="Ordered stages spanning reusable agents + the new component."
    )

    # Back-compat views used by codegen/critic helpers.
    @property
    def agent_name(self) -> str:
        return self.missing_component.agent_name

    @property
    def purpose(self) -> str:
        return self.missing_component.purpose

    @property
    def capabilities(self) -> List[str]:
        return list(self.missing_component.capabilities)

    @property
    def out_of_scope(self) -> List[str]:
        return list(self.missing_component.out_of_scope)

    @property
    def why_not_reuse(self) -> str:
        return self.missing_component.why_not_reuse


# Keep alias so older references/docs still resolve conceptually.
AgentCapabilitySpec = GapAnalysisSpec


class CriticVerdict(BaseModel):
    verdict: Literal["PASS", "FAIL"] = Field(
        description="PASS only if the sub-agent is ready to finalize."
    )
    flaws: List[str] = Field(
        default_factory=list,
        description="Actionable issues to fix when verdict=FAIL.",
    )
    overlap_warnings: List[str] = Field(
        default_factory=list,
        description="Capabilities that collide with existing agents.",
    )
    summary: str = Field(description="One-sentence status for progress logs.")


class EntityAttributeNote(BaseModel):
    entity: str = Field(description="Entity name from topic analysis.")
    attributes: List[str] = Field(
        default_factory=list,
        description="Durable attributes / abilities / roles (not matchup outcomes).",
    )


class PromptResearchBrief(BaseModel):
    domain_overview: str = Field(
        description="2-4 sentences on what this MACRO domain covers."
    )
    entity_attribute_notes: List[EntityAttributeNote] = Field(
        default_factory=list,
        description="Per-entity durable attributes used as domain exemplars.",
    )
    peer_coverage: str = Field(
        description="How peer instances in the same domain stay in-scope."
    )
    evidence_habits: str = Field(
        description="How the agent should search, cite, and admit uncertainty."
    )
    system_prompt_core: str = Field(
        description="Main system-prompt body for codegen to embed (MACRO specialist)."
    )
    open_questions: List[str] = Field(
        default_factory=list,
        description="Gaps to treat carefully at runtime.",
    )


def _format_prompt_research_brief(brief: PromptResearchBrief) -> str:
    entity_blocks = []
    for note in brief.entity_attribute_notes:
        attrs = "\n".join(f"    - {a}" for a in note.attributes) or "    - (none)"
        entity_blocks.append(f"- {note.entity}:\n{attrs}")
    entities = "\n".join(entity_blocks) or "- (none)"
    gaps = "\n".join(f"- {g}" for g in brief.open_questions) or "- (none)"
    return (
        f"Domain overview:\n{brief.domain_overview}\n\n"
        f"Entity attribute notes (exemplars only):\n{entities}\n\n"
        f"Peer coverage:\n{brief.peer_coverage}\n\n"
        f"Evidence habits:\n{brief.evidence_habits}\n\n"
        f"Open questions:\n{gaps}\n\n"
        f"SYSTEM_PROMPT_CORE (embed as the agent system_prompt body):\n"
        f"{brief.system_prompt_core}"
    )


def _topic_context_from_analysis(topic_context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    ctx = dict(topic_context or {})
    return {
        "domain": normalize_domain(str(ctx.get("domain") or "general")),
        "entities": list(ctx.get("entities") or [])[:6],
        "entity_types": list(ctx.get("entity_types") or [])[:6],
        "problem_framing": str(ctx.get("problem_framing") or ""),
        "do_list": list(ctx.get("do_list") or [])[:8],
        "dont_list": list(ctx.get("dont_list") or [])[:8],
        "proposed_agent_name": str(ctx.get("proposed_agent_name") or ""),
        "summary": str(ctx.get("summary") or ""),
    }


def _build_prompt_research_queries(ctx: Dict[str, Any], agent_name: str) -> List[str]:
    """Queries about domain + entity attributes — not the user matchup question."""
    domain = ctx["domain"]
    entity_types = ", ".join(ctx["entity_types"]) or "domain entities"
    queries = [
        f"{domain} domain overview key concepts roles and typical attributes",
        f"{domain} {entity_types} common attributes abilities constraints and evidence sources",
    ]
    for entity in ctx["entities"][:3]:
        queries.append(
            f"{entity} {domain} character OR topic attributes abilities background role"
        )
    queries.append(
        f"{domain} peer examples comparable to {', '.join(ctx['entities'][:2]) or entity_types}"
    )
    # Keep agent_name only as domain hint, not as a micro lock.
    if agent_name and agent_name not in queries[0]:
        queries.append(f"{agent_name.replace('_', ' ')} responsibilities and scope")
    return queries[:6]


def research_prompt_brief(
    *,
    proposed_name: str,
    topic_context: Optional[Dict[str, Any]] = None,
) -> PromptResearchBrief:
    """
    Stage 1b — research MACRO domain + entity attributes for a holistic prompt.

    Uses topic-analysis entities/types/domain already available in the flow.
    Deliberately avoids optimizing for the one-shot user question framing.
    """
    ctx = _topic_context_from_analysis(topic_context)
    agent_name = (
        ctx["proposed_agent_name"]
        or proposed_name
        or proposed_agent_for_domain(ctx["domain"])
    )
    _log(
        "1b-research",
        f"researching holistic prompt for domain={ctx['domain']} "
        f"agent='{agent_name}' entities={ctx['entities']}",
    )

    snippets: List[str] = []
    topic = "finance" if ctx["domain"] in {"finance", "crypto"} else "general"
    for i, query in enumerate(_build_prompt_research_queries(ctx, agent_name), 1):
        try:
            raw = _run_topic_search(query, topic=topic, time_range="year")  # type: ignore[arg-type]
            payload = json.loads(raw)
            answer = (payload.get("answer") or "").strip()
            hits = payload.get("results") or []
            hit_lines = []
            for h in hits[:4]:
                title = (h.get("title") or "").strip()
                content = (h.get("content") or "").strip()[:420]
                url = (h.get("url") or "").strip()
                if content:
                    hit_lines.append(f"  • {title}: {content} ({url})")
            block = (
                f"Query[{i}]: {query}\n"
                f"Answer: {answer or '(none)'}\n"
                f"Hits:\n" + ("\n".join(hit_lines) if hit_lines else "  • (none)")
            )
            snippets.append(block)
            _log("1b-research", f"query[{i}] ok | hits={len(hits)}")
        except Exception as exc:  # noqa: BLE001 — research is best-effort
            snippets.append(f"Query[{i}]: {query}\nERROR: {exc}")
            _log("1b-research", f"query[{i}] failed: {exc}")

    synthesizer = Model.with_structured_output(PromptResearchBrief)
    brief: PromptResearchBrief = synthesizer.invoke(
        [
            SystemMessage(content=SYNTHESIZE_AGENT_PROMPT_RESEARCH),
            HumanMessage(
                content=(
                    f"Proposed agent name: {agent_name}\n"
                    f"MACRO domain: {ctx['domain']}\n"
                    f"Entities (exemplars only): {ctx['entities']}\n"
                    f"Entity types: {ctx['entity_types']}\n"
                    f"Topic framing (context only — do NOT overfit): "
                    f"{ctx['problem_framing'] or ctx['summary'] or '(none)'}\n"
                    f"DO list from analysis:\n"
                    + "\n".join(f"- {d}" for d in ctx["do_list"]) 
                    + "\nDON'T list from analysis:\n"
                    + "\n".join(f"- {d}" for d in ctx["dont_list"])
                    + "\n\nWeb research snippets:\n"
                    + ("\n\n".join(snippets) if snippets else "(no snippets)")
                )
            ),
        ]
    )
    _log(
        "1b-research",
        f"brief ready | entities_noted={len(brief.entity_attribute_notes)} | "
        f"prompt_chars={len(brief.system_prompt_core)}",
    )
    return brief


# --------------------------------------------------------------------------- #
# Stage helpers + progress logging
# --------------------------------------------------------------------------- #

def _log(stage: str, message: str) -> None:
    print(f"[create:{stage}] {message}", flush=True)


def _extract_code_block(text: str) -> str:
    match = re.search(r"```(?:python)?\s*\n(.*?)```", text, re.DOTALL)
    if not match:
        # Some local models return bare python; accept if it parses.
        stripped = text.strip()
        try:
            ast.parse(stripped)
            return stripped
        except SyntaxError as exc:
            raise ValueError(
                "Codegen model did not return a fenced python code block."
            ) from exc
    return match.group(1).strip()


def _repair_common_codegen_mistakes(code: str) -> str:
    """
    Deterministic fixes for recurring local-LLM mistakes so the critic loop
    is not burned on issues we can safely rewrite.
    """
    original = code

    # Wrong package for create_deep_agent (most common failure mode).
    code = re.sub(
        r"from\s+langgraph(?:\.[\w.]+)?\s+import\s+\(\s*([^)]*?)\s*\)",
        lambda m: (
            "from deepagents import create_deep_agent"
            if "create_deep_agent" in m.group(1)
            else m.group(0)
        ),
        code,
        flags=re.S,
    )
    code = re.sub(
        r"from\s+langgraph(?:\.[\w.]+)?\s+import\s+([^\n]+)",
        lambda m: (
            "from deepagents import create_deep_agent"
            if "create_deep_agent" in m.group(1)
            else m.group(0)
        ),
        code,
    )
    # Bare usage without a deepagents import.
    if "create_deep_agent" in code and not re.search(
        r"from\s+deepagents\s+import\s+.*\bcreate_deep_agent\b", code
    ):
        code = "from deepagents import create_deep_agent\n" + code

    # Model must come from utility.llm — never langchain_core / langchain / langgraph.
    code = re.sub(
        r"from\s+langchain(?:_core)?(?:\.[\w.]+)?\s+import\s+\([^)]*\bModel\b[^)]*\)\s*\n?",
        "from utility.llm import Model\n",
        code,
        flags=re.S,
    )
    code = re.sub(
        r"from\s+langchain(?:_core)?(?:\.[\w.]+)?\s+import\s+([^\n]*\bModel\b[^\n]*)\n?",
        "from utility.llm import Model\n",
        code,
    )
    code = re.sub(
        r"from\s+langgraph(?:\.[\w.]+)?\s+import\s+([^\n]*\bModel\b[^\n]*)\n?",
        "from utility.llm import Model\n",
        code,
    )
    code = re.sub(
        r"import\s+langchain(?:_core)?(?:\.[\w.]+)?\s+as\s+Model\b\s*\n?",
        "from utility.llm import Model\n",
        code,
    )
    # Fake Tavily helper packages the local LLM invents.
    code = re.sub(
        r"from\s+langchain(?:_core)?(?:\.[\w.]+)?\s+import\s+[^\n]*\bTavilyTools\b[^\n]*\n?",
        "",
        code,
    )
    code = re.sub(
        r"from\s+langchain\s+import\s+TavilyTools\s*\n?",
        "",
        code,
    )
    if re.search(r"\bModel\b", code) and not re.search(
        r"from\s+utility\.llm\s+import\s+.*\bModel\b", code
    ):
        code = "from utility.llm import Model\n" + code

    # Model accidentally placed in a tools list literal.
    code = re.sub(r"merge_tools\(\s*\[\s*Model\s*\]\s*,", "merge_tools([],", code)
    code = re.sub(r"tools\s*=\s*\[\s*Model\s*\]", "tools = []", code)

    if code != original:
        _log("2-repair", "applied deterministic fixes (imports/tools) to generated code")
    return code


def validate_contract(code: str) -> None:
    """Stage 3 — syntax + presence checks (no LLM)."""
    tree = ast.parse(code)
    top_level_names = set()
    has_build_agent = False
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    top_level_names.add(target.id)
        elif isinstance(node, ast.FunctionDef) and node.name == "build_agent":
            has_build_agent = True
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            top_level_names.add(node.target.id)

    required = {"AGENT_NAME", "AGENT_DESCRIPTION", "AGENT_CAPABILITIES"}
    missing = required - top_level_names
    if missing:
        raise ValueError(f"Generated sub-agent is missing required assignments: {missing}")
    if not has_build_agent:
        raise ValueError("Generated sub-agent is missing a build_agent() function.")

    if re.search(
        r"merge_tools\(\s*\[\s*Model\s*\]|tools\s*=\s*\[\s*Model\s*\]|tools\s*=\s*Model\b",
        code,
    ):
        raise ValueError(
            "Generated sub-agent incorrectly puts Model in the tools list. "
            "Model is the LLM (pass as model=Model); tools must be BaseTool "
            "instances such as finance_search_tool, or merge_tools([], extra_tools)."
        )

    # Model must be imported from utility.llm when referenced.
    if re.search(r"\bModel\b", code):
        if re.search(
            r"from\s+langchain(?:_core)?(?:\.[\w.]+)?\s+import\s+[^\n]*\bModel\b"
            r"|from\s+langgraph(?:\.[\w.]+)?\s+import\s+[^\n]*\bModel\b"
            r"|import\s+langchain(?:_core)?(?:\.[\w.]+)?\s+as\s+Model\b",
            code,
        ):
            raise ValueError(
                "Generated sub-agent imports Model from langchain/langchain_core/"
                "langgraph. Use: from utility.llm import Model"
            )
        if not re.search(r"from\s+utility\.llm\s+import\s+.*\bModel\b", code):
            raise ValueError(
                "Generated sub-agent uses Model but does not import it from "
                "utility.llm. Use: from utility.llm import Model"
            )

    if re.search(r"\bTavilyTools\b", code):
        raise ValueError(
            "Generated sub-agent references non-existent TavilyTools. "
            "Import tools from utility.tavily_tools "
            "(e.g. general_search_tool, finance_search_tool, merge_tools)."
        )

    # create_deep_agent lives in the deepagents package, not langgraph.
    if re.search(
        r"from\s+langgraph(?:\.[\w.]+)?\s+import\s+.*\bcreate_deep_agent\b",
        code,
    ):
        raise ValueError(
            "Generated sub-agent imports create_deep_agent from langgraph. "
            "Use: from deepagents import create_deep_agent"
        )
    if "create_deep_agent" in code and not re.search(
        r"from\s+deepagents\s+import\s+.*\bcreate_deep_agent\b", code
    ):
        raise ValueError(
            "Generated sub-agent uses create_deep_agent but does not import it from "
            "deepagents. Use: from deepagents import create_deep_agent"
        )


def _slugify(name: str) -> str:
    slug = re.sub(r"[^a-z0-9_]+", "_", name.strip().lower())
    return re.sub(r"_+", "_", slug).strip("_") or "sub_agent"


# Tokens that indicate a durable domain (not a one-off instance).
_MACRO_DOMAIN_TOKENS = frozenset(
    {
        "comics",
        "comic",
        "superhero",
        "superheroes",
        "fiction",
        "lore",
        "cinema",
        "film",
        "movie",
        "history",
        "fitness",
        "strength",
        "training",
        "workout",
        "nutrition",
        "meal",
        "recipe",
        "cooking",
        "equity",
        "stock",
        "market",
        "finance",
        "trading",
        "trader",
        "analyst",
        "news",
        "research",
        "legal",
        "travel",
        "education",
        "science",
        "tech",
        "technology",
        "sports",
        "music",
        "literature",
        "mythology",
        "general",
        "primary",
        "investment",
        "strategist",
        "reviewer",
        "report",
        "home",
        "health",
        "wellness",
        "indian",
        "us",
        "global",
    }
)

_AGENT_SUFFIX_RE = re.compile(
    r"_(agent|analyst|specialist|assistant|expert|bot|helper)$"
)


def _agent_stem_parts(agent_name: str) -> List[str]:
    slug = _slugify(agent_name)
    stem = _AGENT_SUFFIX_RE.sub("", slug)
    return [p for p in stem.split("_") if p]


def _looks_micro_agent_name(agent_name: str, task_description: str) -> bool:
    """True when the proposed name is locked to an instance from the question."""
    name = (agent_name or "").strip()
    if name in {"", "__none__", "none", "null"}:
        return False
    parts = _agent_stem_parts(name)
    if not parts:
        return False
    instance_parts = [p for p in parts if p not in _MACRO_DOMAIN_TOKENS]
    # Pure domain vocabulary (comics_lore, home_strength_training) → macro.
    if not instance_parts:
        return False
    task_tokens = set(_slugify(task_description).split("_"))
    # Micro if any non-domain token in the name is drawn from the current ask
    # (batman_agent, spiderman_quiz_agent, apple_news_agent, reliance_stock_only…).
    return any(p in task_tokens for p in instance_parts)


class MacroDomainRewrite(BaseModel):
    agent_name: str = Field(
        description="MACRO snake_case domain name (comics_lore_agent, not batman_agent)."
    )
    purpose: str = Field(description="Durable domain purpose covering peer instances.")
    capabilities: List[str] = Field(
        description="3-6 domain-level capabilities spanning peer instances."
    )
    function: str = Field(description="One-line stage function for the flows manifest.")
    problem_types: List[str] = Field(
        description="Problem classes for the MACRO domain (not one character)."
    )
    example_queries: List[str] = Field(
        description="2-4 example queries across peer instances in the domain."
    )
    composed_flow_id: str = Field(description="snake_case MACRO flow id.")
    composed_flow_name: str = Field(description="Human-readable MACRO flow name.")
    composed_flow_description: str = Field(
        description="What the composed MACRO flow achieves."
    )


MACRO_REWRITE_PROMPT = """You rewrite a proposed NEW sub-agent so it is MACRO-domain,
not micro / one-off.

""" + MACRO_DOMAIN_GUIDANCE + """

Input includes the user task and a draft missing-component that may be too narrow
(e.g. batman_agent). Return a MACRO rewrite:
- agent_name like comics_lore_agent / superhero_fiction_agent
- purpose + capabilities that cover peer instances (Batman AND Superman AND …)
- example_queries that span peers, not only the current ask
- problem_types / flow naming at domain level

Do NOT return __none__ unless the domain is already fully covered (caller handles that).
Return ONLY MacroDomainRewrite.
"""


def _enforce_macro_domain_on_spec(
    spec: GapAnalysisSpec, task_description: str
) -> GapAnalysisSpec:
    """If gap analysis proposed a micro agent name, lift it to a durable domain."""
    missing = spec.missing_component
    name = (missing.agent_name or "").strip()
    if name in {"", "__none__", "none", "null"} or not missing.capabilities:
        return spec
    if not _looks_micro_agent_name(name, task_description):
        return spec

    _log(
        "1-spec",
        f"micro name '{name}' detected — rewriting to MACRO domain before codegen",
    )
    rewriter = Model.with_structured_output(MacroDomainRewrite)
    rewrite: MacroDomainRewrite = rewriter.invoke(
        [
            SystemMessage(content=MACRO_REWRITE_PROMPT),
            HumanMessage(
                content=(
                    f"User task:\n{task_description}\n\n"
                    f"Draft agent_name: {missing.agent_name}\n"
                    f"Draft purpose: {missing.purpose}\n"
                    f"Draft capabilities:\n"
                    + "\n".join(f"- {c}" for c in missing.capabilities)
                    + f"\nDraft function: {missing.function}\n"
                    f"Draft flow_id: {spec.composed_flow_id}\n"
                    f"Draft flow_name: {spec.composed_flow_name}\n"
                )
            ),
        ]
    )
    new_name = _slugify(rewrite.agent_name)
    if _looks_micro_agent_name(new_name, task_description):
        # Last-resort: prefix with a generic domain tag so critic can still refine.
        new_name = f"domain_{new_name}" if not new_name.startswith("domain_") else new_name
        _log("1-spec", f"rewrite still micro-ish; using '{new_name}'")

    old_name = missing.agent_name
    missing.agent_name = new_name
    missing.purpose = rewrite.purpose
    missing.capabilities = list(rewrite.capabilities) or missing.capabilities
    missing.function = rewrite.function or missing.function
    spec.composed_flow_id = _slugify(rewrite.composed_flow_id) or spec.composed_flow_id
    spec.composed_flow_name = rewrite.composed_flow_name or spec.composed_flow_name
    spec.composed_flow_description = (
        rewrite.composed_flow_description or spec.composed_flow_description
    )
    if rewrite.problem_types:
        spec.problem_types = list(rewrite.problem_types)
    if rewrite.example_queries:
        spec.example_queries = list(rewrite.example_queries)
    for stage in spec.flow_stages:
        if stage.is_new or stage.agent == old_name:
            stage.agent = new_name
            stage.is_new = True
    _log("1-spec", f"macro rewrite: '{old_name}' → '{new_name}'")
    return spec


def _unique_target_path(slug: str) -> Path:
    target_path = SUB_AGENTS_DIR / f"{slug}.py"
    if not target_path.exists():
        return target_path
    i = 2
    while (SUB_AGENTS_DIR / f"{slug}_{i}.py").exists():
        i += 1
    return SUB_AGENTS_DIR / f"{slug}_{i}.py"


# --------------------------------------------------------------------------- #
# Pipeline stages
# --------------------------------------------------------------------------- #

def design_capability_spec(
    task_description: str,
    proposed_name: str,
    registry: SubAgentRegistry,
    flows_manifest: Optional["FlowManifest"] = None,
) -> GapAnalysisSpec:
    """Stage 1 — reuse existing roles; isolate only the missing functional niche."""
    _log("1-spec", f"analyzing reusable vs missing components for '{proposed_name}'...")
    designer = Model.with_structured_output(GapAnalysisSpec)
    flows_block = (
        flows_manifest.describe_for_router()
        if flows_manifest is not None
        else "(no flows manifest loaded)"
    )
    spec: GapAnalysisSpec = designer.invoke(
        [
            SystemMessage(content=SPEC_DESIGN_PROMPT),
            HumanMessage(
                content=(
                    f"Proposed name hint: {proposed_name}\n\n"
                    f"Capability gap / full user task:\n{task_description}\n\n"
                    f"Existing registered sub-agents:\n{registry.describe_for_router()}\n\n"
                    f"Active flows catalog:\n{flows_block}"
                )
            ),
        ]
    )
    _log(
        "1-spec",
        f"reuse={len(spec.reusable_components)} | "
        f"new='{spec.missing_component.agent_name}' | "
        f"caps={len(spec.missing_component.capabilities)} | "
        f"flow='{spec.composed_flow_id}' | stages={len(spec.flow_stages)}",
    )
    for i, comp in enumerate(spec.reusable_components, 1):
        _log(
            "1-spec",
            f"  reuse[{i}]: {comp.agent} as {comp.role} — covers: {comp.covers}",
        )
    _log("1-spec", f"missing purpose: {spec.missing_component.purpose}")
    _log("1-spec", f"why_not_reuse: {spec.missing_component.why_not_reuse}")
    return _enforce_macro_domain_on_spec(spec, task_description)


def generate_sub_agent_code(
    spec: GapAnalysisSpec,
    *,
    critique_feedback: Optional[CriticVerdict] = None,
    iteration: int = 1,
    prompt_research: Optional[PromptResearchBrief] = None,
) -> str:
    """Stage 2 — generate or revise ONLY the missing-component module."""
    if critique_feedback is None:
        _log("2-codegen", f"iteration {iteration}: generating missing-component module...")
        revision_block = ""
    else:
        _log("2-codegen", f"iteration {iteration}: revising from critic FAIL...")
        flaws = "\n".join(f"- {f}" for f in critique_feedback.flaws) or "- (none listed)"
        overlaps = (
            "\n".join(f"- {o}" for o in critique_feedback.overlap_warnings) or "- (none)"
        )
        revision_block = (
            "\n\nPREVIOUS VERSION WAS REJECTED BY THE CRITIC.\n"
            f"Critic summary: {critique_feedback.summary}\n"
            f"Flaws to fix:\n{flaws}\n"
            f"Overlap warnings:\n{overlaps}\n"
            "Rewrite the FULL module correcting every flaw. Return ONLY the "
            "python code block.\n"
        )

    reused = "\n".join(
        f"- {c.agent} ({c.role}): {c.covers}" for c in spec.reusable_components
    ) or "- (none)"

    research_block = ""
    if prompt_research is not None:
        research_block = (
            "\n\nHOLISTIC PROMPT RESEARCH BRIEF (mandatory — embed system_prompt_core "
            "into the agent system_prompt; do NOT shrink scope to a single user "
            "question or one entity matchup):\n"
            + _format_prompt_research_brief(prompt_research)
            + "\n"
        )

    prompt = (
        SUB_AGENT_TEMPLATE_INSTRUCTIONS
        + "\n\nIMPORTANT: Implement ONLY the missing functional component below. "
        "Do NOT re-implement work already covered by reusable existing agents.\n"
        + f"\nReusable existing agents (OUT OF SCOPE for this module):\n{reused}\n"
        + "\nApproved missing-component spec (must follow exactly):\n"
        + f"AGENT_NAME: {spec.missing_component.agent_name}\n"
        + f"ROLE: {spec.missing_component.role}\n"
        + f"PURPOSE: {spec.missing_component.purpose}\n"
        + f"STAGE FUNCTION: {spec.missing_component.function}\n"
        + "CAPABILITIES:\n"
        + "\n".join(f"- {c}" for c in spec.missing_component.capabilities)
        + "\nOUT OF SCOPE (do not claim these):\n"
        + "\n".join(f"- {o}" for o in spec.missing_component.out_of_scope)
        + research_block
        + revision_block
    )
    response = Model.invoke(prompt)
    content = response.content if isinstance(response.content, str) else str(response.content)
    code = _repair_common_codegen_mistakes(_extract_code_block(content))
    _log("2-codegen", f"iteration {iteration}: generated {len(code)} chars of source")
    return code


def critique_sub_agent(
    code: str,
    spec: GapAnalysisSpec,
    registry: SubAgentRegistry,
    *,
    iteration: int,
) -> CriticVerdict:
    """Stage 4 — LLM critic reviews missing-component code + scope."""
    _log("4-critic", f"iteration {iteration}: reviewing generated module...")
    critic = Model.with_structured_output(CriticVerdict)
    verdict: CriticVerdict = critic.invoke(
        [
            SystemMessage(content=CRITIC_PROMPT),
            HumanMessage(
                content=(
                    f"Approved GapAnalysisSpec:\n{spec.model_dump_json(indent=2)}\n\n"
                    f"Existing registered sub-agents:\n{registry.describe_for_router()}\n\n"
                    f"Proposed module source:\n```python\n{code}\n```"
                )
            ),
        ]
    )
    verdict = _normalize_critic_verdict(verdict, code, spec, registry)
    _log(
        "4-critic",
        f"iteration {iteration}: verdict={verdict.verdict} | "
        f"flaws={len(verdict.flaws)} | overlaps={len(verdict.overlap_warnings)} | "
        f"{verdict.summary}",
    )
    for i, flaw in enumerate(verdict.flaws, 1):
        _log("4-critic", f"  flaw[{i}]: {flaw}")
    for i, warn in enumerate(verdict.overlap_warnings, 1):
        _log("4-critic", f"  overlap[{i}]: {warn}")
    return verdict


_GENERIC_OVERLAP_MARKERS = (
    "primary_deep_agent",
    "primary deep",
    "general-purpose",
    "general purpose",
    "generic",
)
_FINANCE_OVERLAP_MARKERS = (
    "market_analyst",
    "indian_stock",
    "us_stock",
    "investment_strategist",
    "report_reviewer",
    "nse",
    "bse",
    "equity",
    "stock trader",
)
_KNOWN_MACRO_NAMES = frozenset(
    {
        "comics_lore_agent",
        "superhero_fiction_agent",
        "home_strength_training_agent",
        "nutrition_meal_planning_agent",
        "general_research_agent",
    }
)


def _extract_agent_name_from_code(code: str) -> str:
    m = re.search(r'AGENT_NAME\s*=\s*["\']([a-zA-Z0-9_]+)["\']', code or "")
    return (m.group(1) if m else "").strip()


def _is_spurious_overlap_text(text: str, *, proposed_domain_tokens: set[str]) -> bool:
    blob = (text or "").lower()
    if any(m in blob for m in _GENERIC_OVERLAP_MARKERS):
        return True
    # Finance overlaps are spurious when creating a non-finance domain agent.
    if proposed_domain_tokens & {"comics", "superhero", "fitness", "nutrition", "lore"}:
        if any(m in blob for m in _FINANCE_OVERLAP_MARKERS):
            return True
    # Invented agents not in registry phrases often look like this.
    if "comics_domain_expertise" in blob or "already exists" in blob and "comics_lore" in blob:
        return True
    return False


def _normalize_critic_verdict(
    verdict: CriticVerdict,
    code: str,
    spec: GapAnalysisSpec,
    registry: SubAgentRegistry,
) -> CriticVerdict:
    """Drop bogus primary/finance overlaps; PASS missing MACRO domain agents."""
    agent_name = _extract_agent_name_from_code(code) or (
        spec.missing_component.agent_name or ""
    )
    agent_name = _slugify(agent_name)
    domain_tokens = set(_agent_stem_parts(agent_name)) | {
        t
        for t in _slugify(spec.missing_component.purpose or "").split("_")
        if t
    }

    real_overlaps = [
        w
        for w in (verdict.overlap_warnings or [])
        if not _is_spurious_overlap_text(w, proposed_domain_tokens=domain_tokens)
    ]
    real_flaws = []
    for f in verdict.flaws or []:
        fl = f.lower()
        if _is_spurious_overlap_text(f, proposed_domain_tokens=domain_tokens):
            continue
        # Circular "rename comics_lore_agent to comics_lore_agent" noise.
        if "rename" in fl and agent_name and agent_name in fl and "micro" in fl:
            continue
        if "too micro" in fl and agent_name in _KNOWN_MACRO_NAMES:
            continue
        if "locked to a single instance" in fl and agent_name in _KNOWN_MACRO_NAMES:
            continue
        real_flaws.append(f)

    # True same-domain specialist already registered?
    same_domain_hit = False
    for meta in registry.all_metadata():
        if meta.name == agent_name:
            same_domain_hit = True
            break
        if meta.name in _GENERIC_OVERLAP_MARKERS:
            continue
        # Only treat existing comics_* as blocking another comics_* create.
        if any(t in meta.name for t in ("comics", "superhero")) and any(
            t in agent_name for t in ("comics", "superhero")
        ):
            if meta.name != agent_name:
                same_domain_hit = True
                break

    is_macro = agent_name in _KNOWN_MACRO_NAMES or (
        not _looks_micro_agent_name(agent_name, spec.missing_component.purpose or "")
        and bool(agent_name)
        and agent_name not in {"__none__", "none"}
    )

    if (
        verdict.verdict == "FAIL"
        and is_macro
        and not same_domain_hit
        and not real_overlaps
        and (
            not real_flaws
            or all(
                "primary" in f.lower() or "reuse" in f.lower() and "primary" in f.lower()
                for f in real_flaws
            )
        )
    ):
        _log(
            "4-critic",
            f"overriding FAIL->PASS for missing MACRO domain '{agent_name}' "
            f"(ignored spurious primary/finance overlap claims)",
        )
        return CriticVerdict(
            verdict="PASS",
            flaws=[],
            overlap_warnings=[],
            summary=(
                f"PASS (normalized): '{agent_name}' is a missing MACRO domain "
                f"specialist; primary_deep_agent is not a blocking overlap."
            ),
        )

    return CriticVerdict(
        verdict=verdict.verdict,
        flaws=real_flaws,
        overlap_warnings=real_overlaps,
        summary=verdict.summary,
    )


def persist_and_register(
    code: str,
    registry: SubAgentRegistry,
    preferred_slug: str,
):
    """Stage 6 — write to disk and hot-register into the live registry."""
    code = _repair_common_codegen_mistakes(code)
    validate_contract(code)
    slug = _slugify(preferred_slug)
    target_path = _unique_target_path(slug)
    _log("6-persist", f"writing {target_path.name}...")
    target_path.write_text(code, encoding="utf-8")
    try:
        meta = registry.register_from_file(target_path)
    except Exception:
        # Don't leave a broken module on disk that will poison warm_boot.
        try:
            target_path.unlink(missing_ok=True)
        except OSError:
            pass
        _log("6-persist", f"register failed — removed broken file {target_path.name}")
        raise
    _log(
        "6-persist",
        f"registered '{meta.name}' with {len(meta.capabilities)} capabilities "
        f"at {target_path.name}",
    )
    return meta


def register_composed_flow(
    spec: GapAnalysisSpec,
    *,
    new_agent_name: Optional[str],
    flows_manifest: Optional["FlowManifest"],
    registry: SubAgentRegistry,
    task_description: str,
) -> Optional[str]:
    """
    Stage 7 — upsert the composed multi-agent flow into flows/manifest.json so
    coverage tracking reflects the new reusable+missing composition.
    """
    if flows_manifest is None:
        _log("7-flows", "skipped — no flows manifest provided")
        return None

    from flow_manifest import FlowDefinition, FlowStage

    stages: List[FlowStage] = []
    existing_names = {m.name for m in registry.all_metadata()}
    if new_agent_name:
        existing_names.add(new_agent_name)

    task_blob = (task_description or "").lower()
    comics_task = any(
        m in task_blob
        for m in ("batman", "comics", "superhero", "dhruv", "marvel", "who will win")
    )
    finance_agents = {
        "indian_stock_trader_3m",
        "us_stock_trader_3m",
        "us_stock_trader_3m_2",
        "market_analyst",
        "investment_strategist",
        "report_reviewer",
    }

    for stage in sorted(spec.flow_stages, key=lambda s: s.order):
        agent = new_agent_name if stage.is_new and new_agent_name else stage.agent
        if stage.is_new and not new_agent_name:
            # No new agent was created; drop the new stage.
            continue
        if agent not in existing_names:
            _log("7-flows", f"dropping unknown stage agent '{agent}'")
            continue
        if comics_task and agent in finance_agents:
            _log("7-flows", f"dropping finance agent '{agent}' from comics flow")
            continue
        stages.append(
            FlowStage(
                order=stage.order,
                role=stage.role,
                agent=agent,
                function=stage.function,
            )
        )

    # If the model forgot stages, synthesize from reusable + missing.
    if not stages:
        order = 1
        for comp in spec.reusable_components:
            stages.append(
                FlowStage(
                    order=order,
                    role=comp.role,
                    agent=comp.agent,
                    function=comp.function,
                )
            )
            order += 1
        if new_agent_name:
            stages.append(
                FlowStage(
                    order=order,
                    role=spec.missing_component.role or "missing_component",
                    agent=new_agent_name,
                    function=spec.missing_component.function
                    or spec.missing_component.purpose[:120],
                )
            )

    if not stages:
        _log("7-flows", "skipped — no stages to register")
        return None

    # Normalize order to 1..N
    stages = [
        FlowStage(order=i, role=s.role, agent=s.agent, function=s.function)
        for i, s in enumerate(sorted(stages, key=lambda x: x.order), start=1)
    ]

    flow_id = _slugify(spec.composed_flow_id or f"flow_with_{new_agent_name or 'reuse'}")
    problem_types = list(spec.problem_types) or [
        f"Composed workflow covering: {task_description[:160]}"
    ]
    example_queries = list(spec.example_queries) or [task_description[:200]]

    flow = FlowDefinition(
        flow_id=flow_id,
        name=spec.composed_flow_name or flow_id.replace("_", " ").title(),
        description=spec.composed_flow_description
        or "Auto-registered composition of reused agents plus a missing component.",
        problem_types=problem_types,
        example_queries=example_queries,
        stages=stages,
    )
    flows_manifest.upsert_flow(flow)
    flows_manifest.sync_with_agents(m.name for m in registry.all_metadata())
    flows_manifest.save()
    _log(
        "7-flows",
        f"upserted flow '{flow.flow_id}' status={flow.status} | "
        f"pipeline={' -> '.join(f'{s.role}[{s.agent}]' for s in stages)}",
    )
    _log("7-flows", f"problem coverage now includes: {problem_types}")
    return flow.flow_id


def create_sub_agent_with_critic_loop(
    task_description: str,
    proposed_name: str,
    registry: SubAgentRegistry,
    *,
    flows_manifest: Optional["FlowManifest"] = None,
    topic_context: Optional[Dict[str, Any]] = None,
    max_iters: int = SUBAGENT_CRITIC_MAX_ITERS,
):
    """
    Full creation pipeline with reuse analysis + prompt research + self-correction.

    topic_context should carry topic-analysis fields already in the flow
    (entities, entity_types, domain, do/don't, proposed_agent_name) so prompt
    research stays MACRO and entity-attribute focused.

    Returns (meta_or_none, status_text). meta is None when the gap is fully
    covered by existing agents (no new module created) but a composed flow
    may still be registered.
    """
    _log("pipeline", f"start | proposed='{proposed_name}' | max_iters={max_iters}")

    # Stage 1 — decompose into reusable + missing
    spec = design_capability_spec(
        task_description, proposed_name, registry, flows_manifest=flows_manifest
    )

    missing_name = (spec.missing_component.agent_name or "").strip()
    # Spec stage already ran macro rewrite; if still micro, abort creation.
    if missing_name and missing_name not in {"__none__", "none", "null"}:
        if _looks_micro_agent_name(missing_name, task_description):
            status = (
                f"Aborted create: proposed agent '{missing_name}' is still MICRO "
                f"(instance-level). Refusing to persist; ask planner for a MACRO "
                f"domain agent (e.g. comics_lore_agent for Batman questions)."
            )
            _log("pipeline", status)
            return None, status

    no_new_needed = (
        missing_name in {"", "__none__", "none", "null"}
        or not spec.missing_component.capabilities
    )

    if no_new_needed:
        _log(
            "pipeline",
            "existing agents already cover the task — no new sub-agent will be created",
        )
        flow_id = register_composed_flow(
            spec,
            new_agent_name=None,
            flows_manifest=flows_manifest,
            registry=registry,
            task_description=task_description,
        )
        reused = ", ".join(c.agent for c in spec.reusable_components) or "(none)"
        status = (
            f"No new sub-agent created. Reuse existing: {reused}. "
            f"Flows manifest updated: {flow_id or 'n/a'}."
        )
        _log("pipeline", f"done | {status}")
        return None, status

    # Stage 1b — holistic prompt research from topic entities/domain (not the ask).
    research_ctx = dict(topic_context or {})
    research_ctx.setdefault("proposed_agent_name", missing_name or proposed_name)
    if not research_ctx.get("domain"):
        research_ctx["domain"] = normalize_domain(proposed_name)
    try:
        prompt_research = research_prompt_brief(
            proposed_name=missing_name or proposed_name,
            topic_context=research_ctx,
        )
    except Exception as exc:  # noqa: BLE001 — fall back to spec-only codegen
        _log("1b-research", f"research failed ({exc}); continuing with spec-only codegen")
        prompt_research = None

    critique: Optional[CriticVerdict] = None
    code: Optional[str] = None
    last_contract_error: Optional[str] = None
    meta = None

    for iteration in range(1, max_iters + 1):
        _log("pipeline", f"--- self-correct iteration {iteration}/{max_iters} ---")

        # Stage 2
        code = generate_sub_agent_code(
            spec,
            critique_feedback=critique,
            iteration=iteration,
            prompt_research=prompt_research,
        )
        # Safety net: repair again in case revision path bypasses helpers.
        code = _repair_common_codegen_mistakes(code)

        # Stage 3
        try:
            _log("3-validate", f"iteration {iteration}: running contract checks...")
            validate_contract(code)
            _log("3-validate", f"iteration {iteration}: contract OK")
            last_contract_error = None
        except (SyntaxError, ValueError) as exc:
            # One more repair pass for import mistakes before counting a fail.
            repaired = _repair_common_codegen_mistakes(code)
            if repaired != code:
                code = repaired
                try:
                    validate_contract(code)
                    _log("3-validate", f"iteration {iteration}: contract OK after repair")
                    last_contract_error = None
                except (SyntaxError, ValueError) as exc2:
                    last_contract_error = str(exc2)
                    _log("3-validate", f"iteration {iteration}: FAIL — {exc2}")
                    critique = CriticVerdict(
                        verdict="FAIL",
                        flaws=[f"Contract/static validation error: {exc2}"],
                        overlap_warnings=[],
                        summary="Static contract validation failed.",
                    )
                    continue
            else:
                last_contract_error = str(exc)
                _log("3-validate", f"iteration {iteration}: FAIL — {exc}")
                critique = CriticVerdict(
                    verdict="FAIL",
                    flaws=[f"Contract/static validation error: {exc}"],
                    overlap_warnings=[],
                    summary="Static contract validation failed.",
                )
                continue

        # Stage 4
        critique = critique_sub_agent(code, spec, registry, iteration=iteration)

        # Stage 5 decision
        if critique.verdict == "PASS":
            _log("5-loop", f"PASS on iteration {iteration} — finalizing")
            meta = persist_and_register(code, registry, spec.agent_name)
            flow_id = register_composed_flow(
                spec,
                new_agent_name=meta.name,
                flows_manifest=flows_manifest,
                registry=registry,
                task_description=task_description,
            )
            reused = ", ".join(c.agent for c in spec.reusable_components) or "(none)"
            status = (
                f"Finalized missing component '{meta.name}' after {iteration} "
                f"iteration(s). Reused: {reused}. Flow: {flow_id or 'n/a'}. "
                f"Critic: {critique.summary}"
            )
            _log("pipeline", f"done | {status}")
            return meta, status

        _log("5-loop", f"FAIL on iteration {iteration} — will revise if budget remains")

    # Max iterations exhausted.
    _log(
        "5-loop",
        f"max iterations ({max_iters}) reached without PASS",
    )
    # Abort only on REAL same-domain specialist collisions — not primary/generic.
    blocking_overlaps = []
    if critique:
        blocking_overlaps = [
            w
            for w in (critique.overlap_warnings or [])
            if not _is_spurious_overlap_text(
                w,
                proposed_domain_tokens=set(
                    _agent_stem_parts(spec.missing_component.agent_name or "")
                ),
            )
        ]
        blocking_flaws = [
            f
            for f in (critique.flaws or [])
            if (
                ("overlap" in f.lower() or "reuse" in f.lower())
                and not _is_spurious_overlap_text(
                    f,
                    proposed_domain_tokens=set(
                        _agent_stem_parts(spec.missing_component.agent_name or "")
                    ),
                )
            )
        ]
    else:
        blocking_flaws = []

    if critique and (blocking_overlaps or blocking_flaws):
        reused = ", ".join(c.agent for c in spec.reusable_components) or "(see critic flaws)"
        status = (
            f"Creation aborted after {max_iters} critic FAILs due to overlap/"
            f"reuse requirement. Reuse existing agents instead: {reused}. "
            f"Flaws: {'; '.join(critique.flaws[:3])}"
        )
        _log("pipeline", f"done (aborted) | {status}")
        # Do not register junk flows on abort when nothing reusable is real.
        if spec.reusable_components:
            register_composed_flow(
                spec,
                new_agent_name=None,
                flows_manifest=flows_manifest,
                registry=registry,
                task_description=task_description,
            )
        return None, status

    if code is None or last_contract_error:
        raise ValueError(
            f"Sub-agent creation failed after {max_iters} iterations. "
            f"Last contract error: {last_contract_error}"
        )

    meta = persist_and_register(code, registry, spec.agent_name)
    flow_id = register_composed_flow(
        spec,
        new_agent_name=meta.name,
        flows_manifest=flows_manifest,
        registry=registry,
        task_description=task_description,
    )
    flaw_summary = "; ".join((critique.flaws if critique else [])[:3]) or "n/a"
    reused = ", ".join(c.agent for c in spec.reusable_components) or "(none)"
    status = (
        f"Best-effort finalize of '{meta.name}' after {max_iters} iterations "
        f"without critic PASS. Reused: {reused}. Flow: {flow_id or 'n/a'}. "
        f"Remaining flaws: {flaw_summary}"
    )
    _log("pipeline", f"done (best-effort) | {status}")
    return meta, status


# --------------------------------------------------------------------------- #
# Public tool for the supervisor graph
# --------------------------------------------------------------------------- #

def make_create_sub_agent_tool(
    registry: SubAgentRegistry,
    flows_manifest: Optional["FlowManifest"] = None,
):
    """
    Factory so the tool closes over the *live* registry (and optional flows
    manifest) used by the running supervisor graph.
    """

    @tool
    def create_sub_agent_tool(task_description: str, proposed_name: str) -> str:
        """
        Analyze reusable existing agents, create ONLY the missing functional
        sub-agent component (critic loop max 3), persist/register it, and update
        the flows manifest with the composed multi-agent functionality.

        Call this ONLY after confirming no existing sub-agent alone covers the
        full task.

        Args:
            task_description: Full capability gap / user problem.
            proposed_name: Working name/slug hint for the missing component.

        Returns:
            Confirmation including reused agents, new component (if any), and
            the updated flow id.
        """
        meta, status = create_sub_agent_with_critic_loop(
            task_description,
            proposed_name,
            registry,
            flows_manifest=flows_manifest,
        )
        if meta is None:
            return (
                f"No new sub-agent required.\nPipeline: {status}\n"
                "Route using the reused agents / updated flow in the flows manifest."
            )
        return (
            f"Created missing-component sub-agent '{meta.name}' at "
            f"{Path(meta.file_path).name}.\n"
            f"Pipeline: {status}\n"
            f"Description: {meta.description}\n"
            f"Capabilities: {meta.capabilities}\n"
            f"You can now route via '{meta.name}' and/or the updated composed flow."
        )

    return create_sub_agent_tool


# Back-compat aliases
_validate_contract = validate_contract
