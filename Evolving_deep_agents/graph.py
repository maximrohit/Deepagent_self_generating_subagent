"""
Builds the supervisor StateGraph:

    START
      -> analyze_topic_node       # entities, domain, do/don't, manifest check
      -> plan_todos_node          # TODOs grounded in the topic briefing
      -> execute_todos_node       # manifest match / create-missing / invoke
      -> draft_synthesize_node    # build draft from TODO results
      -> validate_query_node      # check draft vs ORIGINAL user query
           |-- PASS or iter==max -> finalize_node -> END
           |-- FAIL & iter<max  -> plan_todos_node (refinement TODOs;
                                   prior draft + gaps + topic analysis as input;
                                   may CREATE missing agents on iters 2/3)

Quality validation sits ABOVE sub-agent creation: creation still only fills
true missing niches; existing functional agents are preferred first.
"""
from __future__ import annotations

from typing import List, Literal, Optional

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from code_generator import create_sub_agent_with_critic_loop
from config import QUERY_VALIDATE_MAX_ITERS
from flow_manifest import FlowManifest
from prompt_library import (
    ANALYZE_TOPIC_PROMPT,
    DRAFT_SYNTH_PROMPT,
    FINALIZE_PROMPT,
    PLAN_TODOS_PROMPT,
    VALIDATE_PROMPT,
)
from registry import SubAgentRegistry
from state import SupervisorState, TodoItem, TopicAnalysis
from utility.domains import (
    DOMAIN_IDS,
    FINANCE_ONLY_AGENT_NAMES,
    GENERIC_AGENT_NAMES,
    classify_task_domain,
    domain_catalog_for_prompt,
    domains_compatible,
    infer_agent_domain,
    is_finance_like_domain,
    needs_specialist_create,
    normalize_domain,
    proposed_agent_for_domain,
)
from utility.llm import Model


class PlannedTodo(BaseModel):
    id: int = Field(description="1-based order.")
    description: str = Field(description="Concrete executable step.")
    assigned_agent: str = Field(
        default="",
        description="Exact existing agent name, or empty if CREATE_NEEDED.",
    )
    flow_id: str = Field(default="", description="Matching flow_id if any.")
    notes: str = Field(
        default="",
        description="Short rationale or CREATE_NEEDED: <gap>.",
    )


class TodoPlan(BaseModel):
    todos: List[PlannedTodo] = Field(description="Ordered TODO list.")
    plan_summary: str = Field(description="One-sentence plan overview.")


class QueryValidation(BaseModel):
    verdict: Literal["PASS", "FAIL"] = Field(
        description="PASS only if the draft satisfies the original user query."
    )
    gaps: List[str] = Field(
        default_factory=list,
        description="Actionable missing pieces vs the original query.",
    )
    summary: str = Field(description="One-sentence validation status.")


class TopicBriefing(BaseModel):
    entities: List[str] = Field(
        default_factory=list,
        description="Named entities in the user question.",
    )
    entity_types: List[str] = Field(
        default_factory=list,
        description="What those entities are (character, ticker, …).",
    )
    domain: str = Field(
        description=(
            "MACRO domain id for routing/creation. Must be one of: "
            + ", ".join(DOMAIN_IDS)
        )
    )
    problem_framing: str = Field(
        description="What a correct answer must cover."
    )
    do_list: List[str] = Field(
        default_factory=list,
        description="Actions we should take.",
    )
    dont_list: List[str] = Field(
        default_factory=list,
        description="Actions / agents / framings to avoid.",
    )
    matching_agents: List[str] = Field(
        default_factory=list,
        description="Exact existing agent names that fit the domain.",
    )
    matching_flows: List[str] = Field(
        default_factory=list,
        description="Exact existing flow_ids that fit.",
    )
    reuse_decision: str = Field(
        description="Reuse vs create decision in one sentence."
    )
    create_needed: bool = Field(
        description="True if a MACRO domain specialist must be created."
    )
    proposed_agent_name: str = Field(
        default="",
        description="MACRO agent name to create, or empty.",
    )
    summary: str = Field(description="One-sentence briefing.")


_SCORE_STOPWORDS = frozenset(
    {
        "with", "without", "will", "would", "this", "that", "from", "have",
        "been", "were", "their", "about", "into", "over", "under", "than",
        "then", "when", "what", "which", "your", "ours", "they", "them",
        "time", "week", "month", "year", "days", "using", "provide", "clear",
        "between", "super", "data", "more", "most", "also", "only", "both",
        "each", "same", "such", "make", "made", "does", "done", "task",
        "todo", "step", "user", "query", "answer", "draft", "prior",
    }
)


def _flow(stage: str, message: str) -> None:
    print(f"[flow:{stage}] {message}", flush=True)


def _classify_task_domain(text: str) -> str:
    """Return MACRO domain id from user/TODO text (extended catalog)."""
    return classify_task_domain(text)


def _is_finance_task(text: str) -> bool:
    return is_finance_like_domain(_classify_task_domain(text))


def _is_non_finance_task(text: str) -> bool:
    return not _is_finance_task(text)


def _agent_looks_finance(
    name: str, description: str = "", capabilities: Optional[List[str]] = None
) -> bool:
    """True for finance/crypto specialists — not generic/primary research agents."""
    n = (name or "").strip().lower()
    if not n or n in GENERIC_AGENT_NAMES:
        return False
    if n in FINANCE_ONLY_AGENT_NAMES:
        return True
    return infer_agent_domain(name, description, capabilities) in {"finance", "crypto"}


def _agent_domain(
    name: str, description: str = "", capabilities: Optional[List[str]] = None
) -> str:
    return infer_agent_domain(name, description, capabilities)


def _domains_compatible(task_domain: str, agent_domain: str) -> bool:
    return domains_compatible(task_domain, agent_domain)


def _proposed_name_for_domain(domain: str, notes: str = "") -> str:
    """Canonical MACRO name for a domain; ignore micro hints from the planner."""
    domain = normalize_domain(domain)
    canonical = proposed_agent_for_domain(domain)
    if needs_specialist_create(domain):
        return canonical
    if "CREATE_NEEDED" in (notes or "").upper() and ":" in notes:
        hint = notes.split(":", 1)[-1].strip()
        token = hint.split()[0].strip(",.;").lower() if hint else ""
        if token and not any(
            bad in token for bad in ("_vs_", "versus", "batman", "dhruv", "only")
        ):
            if token not in {"", "none", "gap"}:
                return token
    return canonical


def _find_domain_agent(registry: SubAgentRegistry, domain: str) -> Optional[str]:
    domain = normalize_domain(domain)
    specialists: List[str] = []
    generics: List[str] = []
    for meta in registry.all_metadata():
        ad = _agent_domain(meta.name, meta.description, meta.capabilities)
        if ad == domain:
            specialists.append(meta.name)
        elif (not is_finance_like_domain(domain)) and meta.name in GENERIC_AGENT_NAMES:
            generics.append(meta.name)
    if specialists:
        specialists.sort()
        return specialists[0]
    if generics:
        generics.sort()
        if "primary_deep_agent" in generics:
            return "primary_deep_agent"
        return generics[0]
    return None


def _todo_looks_like_prompt_echo(description: str) -> bool:
    blob = (description or "").lower()
    markers = (
        "assign the assigned_agent",
        "exact existing agent name",
        "use known flows when the domain matches",
        "create a new flow for",
        "domain matches the user question domain",
        "finance/india/us equity",
    )
    return any(m in blob for m in markers)


def _sanitize_todo_assignments(
    todos: List[TodoItem], task: str, registry: SubAgentRegistry
) -> List[TodoItem]:
    """Strip cross-domain planner mistakes (e.g. stock trader on Batman)."""
    task_domain = _classify_task_domain(task)
    cleaned: List[TodoItem] = []
    for todo in todos:
        if _todo_looks_like_prompt_echo(todo.get("description") or ""):
            _flow(
                "plan",
                f"dropping prompt-echo TODO[{todo.get('id')}]: {todo.get('description')}",
            )
            continue
        blob = f"{task} {todo.get('description', '')}"
        todo_domain = _classify_task_domain(blob)
        domain = task_domain if task_domain != "general" else todo_domain
        name = (todo.get("assigned_agent") or "").strip()
        if name and registry.exists(name):
            meta = next((m for m in registry.all_metadata() if m.name == name), None)
            if meta is not None:
                ad = _agent_domain(meta.name, meta.description, meta.capabilities)
                if not _domains_compatible(domain, ad):
                    _flow(
                        "plan",
                        f"sanitizing TODO[{todo.get('id')}]: '{name}' "
                        f"({ad}) incompatible with domain={domain}",
                    )
                    todo["assigned_agent"] = ""
                    name = ""
        if not name:
            specialist = _find_domain_agent(registry, domain)
            if specialist:
                meta = next(
                    (m for m in registry.all_metadata() if m.name == specialist), None
                )
                ad = (
                    _agent_domain(meta.name, meta.description, meta.capabilities)
                    if meta
                    else "general"
                )
                if ad == domain:
                    todo["assigned_agent"] = specialist
                    todo["notes"] = (todo.get("notes") or "") or f"domain={domain}"
                    cleaned.append(todo)
                    continue
            if domain != "finance":
                # Always use canonical MACRO domain name (ignore micro planner hints).
                proposed = _proposed_name_for_domain(domain, "")
                if registry.exists(proposed):
                    todo["assigned_agent"] = proposed
                    todo["notes"] = f"domain={domain}"
                else:
                    todo["assigned_agent"] = ""
                    todo["notes"] = f"CREATE_NEEDED: {proposed}"
            elif registry.exists("primary_deep_agent"):
                todo["assigned_agent"] = "primary_deep_agent"
        cleaned.append(todo)

    # If the planner only echoed rules, synthesize one real domain TODO.
    if not cleaned:
        proposed = _proposed_name_for_domain(task_domain, "")
        notes = (
            f"domain={task_domain}"
            if registry.exists(proposed)
            else f"CREATE_NEEDED: {proposed}"
        )
        cleaned.append(
            {
                "id": 1,
                "description": (
                    f"Answer the user question with a {task_domain} domain specialist: {task}"
                ),
                "assigned_agent": proposed if registry.exists(proposed) else "",
                "flow_id": "",
                "status": "pending",
                "result": "",
                "notes": notes,
            }
        )
        _flow("plan", f"synthesized domain TODO for {task_domain} → {proposed}")
    return cleaned


def _resolve_agent_for_todo(
    todo: TodoItem,
    registry: SubAgentRegistry,
    flows_manifest: FlowManifest | None,
    *,
    user_task: str = "",
) -> Optional[str]:
    """Pick a live agent only when its domain matches the task domain."""
    blob = f"{user_task} {todo.get('description', '')} {todo.get('notes', '')}".lower()
    task_domain = _classify_task_domain(user_task)
    todo_domain = _classify_task_domain(blob)
    domain = task_domain if task_domain != "general" else todo_domain
    non_finance = domain != "finance"
    notes = todo.get("notes") or ""

    # Honor CREATE_NEEDED: do not silently substitute primary/finance.
    if notes.upper().startswith("CREATE_NEEDED"):
        proposed = _proposed_name_for_domain(domain, notes)
        if registry.exists(proposed):
            return proposed
        return None

    name = (todo.get("assigned_agent") or "").strip()
    if name and registry.exists(name):
        meta = next((m for m in registry.all_metadata() if m.name == name), None)
        if meta is not None:
            ad = _agent_domain(meta.name, meta.description, meta.capabilities)
            if _domains_compatible(domain, ad):
                return name
            _flow(
                "todo",
                f"rejecting '{name}' (agent_domain={ad}) for task_domain={domain}",
            )

    flow_id = (todo.get("flow_id") or "").strip()
    if flows_manifest and flow_id and domain == "finance":
        flow = flows_manifest.get(flow_id)
        if flow and flow.status == "active" and flow.required_agents:
            for agent in flow.required_agents:
                if registry.exists(agent):
                    return agent

    # Prefer an existing same-domain specialist (comics_lore_agent, etc.).
    specialist = _find_domain_agent(registry, domain)
    if specialist:
        meta = next((m for m in registry.all_metadata() if m.name == specialist), None)
        if meta and _agent_domain(meta.name, meta.description, meta.capabilities) == domain:
            return specialist

    # Keyword score ONLY within compatible domains; ignore stopwords like "time".
    scored = []
    for meta in registry.all_metadata():
        ad = _agent_domain(meta.name, meta.description, meta.capabilities)
        if not _domains_compatible(domain, ad):
            continue
        if non_finance and _agent_looks_finance(
            meta.name, meta.description, meta.capabilities
        ):
            continue
        # Specialist domains must not soft-match onto primary via weak tokens.
        if needs_specialist_create(domain) and meta.name in GENERIC_AGENT_NAMES:
            continue
        hay = f"{meta.name} {meta.description} {' '.join(meta.capabilities)}".lower()
        tokens = [
            tok
            for tok in blob.replace("'", " ").split()
            if len(tok) > 3 and tok not in _SCORE_STOPWORDS
        ]
        score = sum(1 for tok in tokens if tok in hay)
        if is_finance_like_domain(domain):
            if "india" in blob or "nse" in blob or "bse" in blob or "nifty" in blob:
                if "india" in hay or "nse" in hay or "bse" in hay:
                    score += 5
                if "nyse" in hay or "nasdaq" in hay:
                    score -= 3
            if "intraday" in blob or "trade" in blob:
                if "trader" in meta.name or "trade" in hay:
                    score += 3
        # Boost true same-domain specialists.
        if ad == domain:
            score += 6
        if meta.name in GENERIC_AGENT_NAMES and domain == "general":
            score += 2
        if score > 0:
            scored.append((score, 0 if ad == domain else 1, meta.name))
    if scored:
        scored.sort(key=lambda x: (x[1], -x[0], x[2]))
        return scored[0][2]

    # Missing specialist domain → None so execute path CREATE_NEEDED.
    if needs_specialist_create(domain):
        return None
    if registry.exists("primary_deep_agent"):
        return "primary_deep_agent"
    return None


def _todo_results_block(todos: List[TodoItem]) -> str:
    blocks = []
    for t in todos:
        blocks.append(
            f"TODO[{t.get('id')}] agent={t.get('assigned_agent')} "
            f"status={t.get('status')}\n"
            f"Step: {t.get('description')}\n"
            f"Result:\n{t.get('result') or '(empty)'}"
        )
    return "\n\n---\n\n".join(blocks) if blocks else "(no todo results)"


def _format_topic_analysis(analysis: TopicAnalysis) -> str:
    def _bullets(items: List[str]) -> str:
        return "\n".join(f"- {x}" for x in items) if items else "- (none)"

    return (
        f"Summary: {analysis.get('summary', '')}\n"
        f"Domain: {analysis.get('domain', 'general')}\n"
        f"Entities: {', '.join(analysis.get('entities') or []) or '(none)'}\n"
        f"Entity types: {', '.join(analysis.get('entity_types') or []) or '(none)'}\n"
        f"Problem framing:\n{analysis.get('problem_framing', '')}\n"
        f"DO:\n{_bullets(list(analysis.get('do_list') or []))}\n"
        f"DON'T:\n{_bullets(list(analysis.get('dont_list') or []))}\n"
        f"Matching agents: {', '.join(analysis.get('matching_agents') or []) or '(none)'}\n"
        f"Matching flows: {', '.join(analysis.get('matching_flows') or []) or '(none)'}\n"
        f"Reuse decision: {analysis.get('reuse_decision', '')}\n"
        f"Create needed: {analysis.get('create_needed', False)}\n"
        f"Proposed agent: {analysis.get('proposed_agent_name') or '(n/a)'}"
    )


def _harden_topic_briefing(
    briefing: TopicBriefing,
    task: str,
    registry: SubAgentRegistry,
    flows_manifest: FlowManifest | None,
) -> TopicAnalysis:
    """Merge LLM briefing with deterministic domain/manifest checks."""
    domain = _classify_task_domain(task)
    # Prefer deterministic domain when it is more specific than LLM "general".
    llm_domain = normalize_domain(briefing.domain or "general")
    if domain == "general" and llm_domain != "general":
        domain = llm_domain
    elif domain != "general":
        pass  # keep deterministic
    else:
        domain = llm_domain

    matching_agents: List[str] = []
    for meta in registry.all_metadata():
        ad = _agent_domain(meta.name, meta.description, meta.capabilities)
        if _domains_compatible(domain, ad) and ad == domain:
            matching_agents.append(meta.name)
        elif domain == "general" and meta.name in GENERIC_AGENT_NAMES:
            matching_agents.append(meta.name)
    # Deduplicate while preserving order.
    seen = set()
    matching_agents = [a for a in matching_agents if not (a in seen or seen.add(a))]

    # Also keep LLM suggestions that actually exist and are domain-compatible.
    for name in briefing.matching_agents:
        if name in seen or not registry.exists(name):
            continue
        meta = next((m for m in registry.all_metadata() if m.name == name), None)
        if meta is None:
            continue
        ad = _agent_domain(meta.name, meta.description, meta.capabilities)
        if _domains_compatible(domain, ad):
            matching_agents.append(name)
            seen.add(name)

    matching_flows: List[str] = []
    if flows_manifest is not None:
        for flow in flows_manifest.all_flows():
            if getattr(flow, "status", "") != "active":
                continue
            agents = list(flow.required_agents or [])
            if (not is_finance_like_domain(domain)) and any(
                _agent_looks_finance(a) for a in agents
            ):
                continue
            if is_finance_like_domain(domain) and not any(
                _agent_looks_finance(a) for a in agents
            ):
                # Allow general_web_research as weak finance match only.
                if flow.flow_id != "general_web_research":
                    continue
            # Specialist domains: only keep flows whose agents are compatible.
            if needs_specialist_create(domain):
                ok = True
                for a in agents:
                    meta = next((m for m in registry.all_metadata() if m.name == a), None)
                    if meta is None:
                        ok = False
                        break
                    ad = _agent_domain(meta.name, meta.description, meta.capabilities)
                    if not _domains_compatible(domain, ad):
                        ok = False
                        break
                if not ok:
                    continue
            matching_flows.append(flow.flow_id)
        for fid in briefing.matching_flows:
            if fid not in matching_flows and flows_manifest.get(fid):
                flow = flows_manifest.get(fid)
                agents = list(flow.required_agents or []) if flow else []
                if (not is_finance_like_domain(domain)) and any(
                    _agent_looks_finance(a) for a in agents
                ):
                    continue
                matching_flows.append(fid)

    proposed = _proposed_name_for_domain(domain, "")
    specialist_exists = any(
        _agent_domain(m.name, m.description, m.capabilities) == domain
        for m in registry.all_metadata()
        if domain != "general"
    )
    if domain == "general":
        create_needed = False
        proposed_name = ""
        reuse = (
            f"Reuse primary_deep_agent / matching agents: "
            f"{', '.join(matching_agents) or 'primary_deep_agent'}"
        )
    elif specialist_exists:
        create_needed = False
        proposed_name = ""
        reuse = f"Reuse existing {domain} specialist(s): {', '.join(matching_agents)}"
    else:
        create_needed = needs_specialist_create(domain)
        proposed_name = proposed if create_needed else ""
        reuse = (
            f"No {domain} specialist in manifest — CREATE_NEEDED: {proposed_name}; "
            f"do not use finance agents"
            if create_needed
            else f"Use general research for {domain}: "
            f"{', '.join(matching_agents) or 'primary_deep_agent'}"
        )

    # Seed default do/don't if the model left them thin.
    do_list = list(briefing.do_list or [])
    dont_list = list(briefing.dont_list or [])
    if not is_finance_like_domain(domain):
        if not any("finance" in d.lower() or "stock" in d.lower() for d in dont_list):
            dont_list.append(
                f"Do not use stock/finance agents for a {domain} question"
            )
    if domain == "comics":
        if not any("compar" in d.lower() or "abilit" in d.lower() for d in do_list):
            do_list.append(
                "Research each entity's abilities, feats, and prep-time tactics in comics lore"
            )
        if not any("micro" in d.lower() or "only" in d.lower() for d in dont_list):
            dont_list.append("Do not create a micro batman_only / dhruv_only agent")
    if needs_specialist_create(domain) and not any(
        "macro" in d.lower() or "domain" in d.lower() for d in do_list
    ):
        do_list.append(
            f"Use or create the MACRO {domain} specialist "
            f"({proposed_agent_for_domain(domain)})"
        )

    entities = list(briefing.entities or [])
    entity_types = list(briefing.entity_types or [])
    if domain == "comics" and not entity_types:
        entity_types = ["comic-book characters / superheroes"]

    analysis: TopicAnalysis = {
        "entities": entities,
        "entity_types": entity_types,
        "domain": domain,
        "problem_framing": briefing.problem_framing
        or f"Solve the user question within the {domain} domain.",
        "do_list": do_list,
        "dont_list": dont_list,
        "matching_agents": matching_agents,
        "matching_flows": matching_flows,
        "reuse_decision": reuse,
        "create_needed": create_needed,
        "proposed_agent_name": proposed_name,
        "summary": briefing.summary
        or f"{domain} briefing for: {', '.join(entities) or task[:80]}",
    }
    return analysis


def build_supervisor_graph(
    registry: SubAgentRegistry,
    flows_manifest: FlowManifest | None = None,
):
    analyzer = Model.with_structured_output(TopicBriefing)
    planner = Model.with_structured_output(TodoPlan)
    validator = Model.with_structured_output(QueryValidation)
    synthesizer = Model
    max_iters = max(1, QUERY_VALIDATE_MAX_ITERS)

    def analyze_topic_node(state: SupervisorState) -> dict:
        """Initial research: entities, domain, do/don't, manifest reuse check."""
        existing = state.get("topic_analysis")
        # Reuse briefing on refinement loops; only run once per user task.
        if existing and (state.get("validation_iteration") or 0) >= 1:
            _flow("analyze", "reusing prior topic briefing for refinement")
            return {}

        task = state["task"]
        _flow("analyze", "researching entities, domain, and manifest coverage...")
        flows_block = (
            flows_manifest.describe_for_router()
            if flows_manifest is not None
            else "(no flows manifest)"
        )
        briefing: TopicBriefing = analyzer.invoke(
            [
                SystemMessage(content=ANALYZE_TOPIC_PROMPT),
                HumanMessage(
                    content=(
                        f"User task:\n{task}\n\n"
                        f"Registered sub-agents:\n{registry.describe_for_router()}\n\n"
                        f"Active flows:\n{flows_block}"
                    )
                ),
            ]
        )
        analysis = _harden_topic_briefing(
            briefing, task, registry, flows_manifest
        )
        _flow("analyze", f"domain={analysis.get('domain')} | {analysis.get('summary')}")
        _flow(
            "analyze",
            f"entities={analysis.get('entities')} | types={analysis.get('entity_types')}",
        )
        for i, item in enumerate(analysis.get("do_list") or [], 1):
            _flow("analyze", f"  DO[{i}]: {item}")
        for i, item in enumerate(analysis.get("dont_list") or [], 1):
            _flow("analyze", f"  DONT[{i}]: {item}")
        _flow(
            "analyze",
            f"manifest agents={analysis.get('matching_agents') or []} | "
            f"flows={analysis.get('matching_flows') or []}",
        )
        _flow(
            "analyze",
            f"decision: {analysis.get('reuse_decision')} | "
            f"create_needed={analysis.get('create_needed')} "
            f"proposed={analysis.get('proposed_agent_name') or '-'}",
        )
        return {
            "topic_analysis": analysis,
            "messages": [
                AIMessage(
                    content=f"[analyze] {_format_topic_analysis(analysis)}",
                    name="topic_analysis",
                )
            ],
        }

    def plan_todos_node(state: SupervisorState) -> dict:
        task = state["task"]
        iteration = int(state.get("validation_iteration") or 0) + 1
        prior_draft = state.get("best_output") or state.get("draft_output") or ""
        gaps = state.get("validation_gaps") or []
        analysis = state.get("topic_analysis") or {}

        mode = "initial" if iteration == 1 else f"refinement#{iteration}"
        _flow("plan", f"writing TODO list ({mode}) against manifests + topic briefing...")

        flows_block = (
            flows_manifest.describe_for_router()
            if flows_manifest is not None
            else "(no flows manifest)"
        )
        human = (
            f"User task (ORIGINAL — must be fully satisfied):\n{task}\n\n"
            f"TOPIC BRIEFING (obey this — entities/domain/do/don't/manifest):\n"
            f"{_format_topic_analysis(analysis) if analysis else '(missing — infer carefully)'}\n\n"
            f"Active flows:\n{flows_block}\n\n"
            f"Registered sub-agents:\n{registry.describe_for_router()}\n\n"
        )
        if iteration > 1:
            human += (
                f"PRIOR DRAFT (use as input; improve it):\n{prior_draft}\n\n"
                f"VALIDATION GAPS to close this iteration:\n"
                + "\n".join(f"- {g}" for g in gaps)
                + "\n\n"
                "Write refinement TODOs that close ONLY these gaps while keeping "
                "good parts of the prior draft. Still obey the TOPIC BRIEFING domain.\n"
            )

        plan: TodoPlan = planner.invoke(
            [SystemMessage(content=PLAN_TODOS_PROMPT), HumanMessage(content=human)]
        )
        todos: List[TodoItem] = []
        for t in plan.todos:
            todos.append(
                {
                    "id": t.id,
                    "description": t.description,
                    "assigned_agent": t.assigned_agent or "",
                    "flow_id": t.flow_id or "",
                    "status": "pending",
                    "result": "",
                    "notes": t.notes or "",
                }
            )
        # If briefing says create_needed and no specialist exists, stamp notes.
        if analysis.get("create_needed") and analysis.get("proposed_agent_name"):
            proposed = analysis["proposed_agent_name"]
            for todo in todos:
                name = (todo.get("assigned_agent") or "").strip()
                if name and registry.exists(name):
                    meta = next(
                        (m for m in registry.all_metadata() if m.name == name), None
                    )
                    if meta and _agent_domain(
                        meta.name, meta.description, meta.capabilities
                    ) == analysis.get("domain"):
                        continue
                if not name or _agent_looks_finance(name):
                    todo["assigned_agent"] = ""
                    todo["notes"] = f"CREATE_NEEDED: {proposed}"

        todos = _sanitize_todo_assignments(todos, task, registry)
        _flow("plan", f"iter={iteration} | {plan.plan_summary}")
        _flow(
            "plan",
            f"task_domain={analysis.get('domain') or _classify_task_domain(task)} after sanitize",
        )
        for t in todos:
            _flow(
                "plan",
                f"  TODO[{t['id']}] agent={t.get('assigned_agent') or '(unassigned)'} "
                f"| flow={t.get('flow_id') or '-'} | {t['description']}",
            )
        return {
            "todos": todos,
            "todo_results": {},
            "validation_iteration": iteration,
            "messages": [
                AIMessage(content=f"[plan:iter{iteration}] {plan.plan_summary}")
            ],
        }

    def execute_todos_node(state: SupervisorState) -> dict:
        task = state["task"]
        iteration = int(state.get("validation_iteration") or 1)
        todos = list(state.get("todos") or [])
        results = dict(state.get("todo_results") or {})
        messages: List[AIMessage] = []
        prior_context_parts: List[str] = []
        prior_draft = state.get("best_output") or state.get("draft_output") or ""
        analysis = state.get("topic_analysis") or {}
        if analysis:
            prior_context_parts.append(
                "TOPIC BRIEFING (stay in this domain; obey do/don't):\n"
                + _format_topic_analysis(analysis)
            )
        if prior_draft:
            prior_context_parts.append(f"PRIOR DRAFT FROM EARLIER ITERATION:\n{prior_draft}")

        for todo in todos:
            todo_id = str(todo["id"])
            _flow("todo", f"[iter{iteration}] --- TODO[{todo_id}]: {todo['description']}")
            agent_name = _resolve_agent_for_todo(
                todo, registry, flows_manifest, user_task=task
            )
            notes = todo.get("notes") or ""

            needs_create = (not agent_name) or notes.upper().startswith("CREATE_NEEDED")
            if needs_create and not agent_name:
                domain = _classify_task_domain(task)
                proposed = _proposed_name_for_domain(domain, notes)
                _flow(
                    "todo",
                    f"TODO[{todo_id}] no suitable existing agent "
                    f"(domain={domain}) — creation critic pipeline as '{proposed}'...",
                )
                gap = notes.split(":", 1)[-1].strip() if ":" in notes else todo["description"]
                entities = ", ".join(analysis.get("entities") or []) or "(from task)"
                entity_types = ", ".join(analysis.get("entity_types") or []) or "(unspecified)"
                create_task = (
                    f"MACRO domain specialist for: {domain}\n"
                    f"Entities involved (exemplars only): {entities}\n"
                    f"Entity types / attributes focus: {entity_types}\n"
                    f"Domain briefing:\n{_format_topic_analysis(analysis) if analysis else '(none)'}\n"
                    f"TODO step:\n{todo['description']}\n"
                    f"Gap hint:\n{gap or proposed}\n"
                    "NOTE: User question framing is secondary — build a durable "
                    "domain agent covering peers and entity attributes, not one ask."
                )
                meta, status = create_sub_agent_with_critic_loop(
                    create_task,
                    proposed,
                    registry,
                    flows_manifest=flows_manifest,
                    topic_context=analysis,
                )
                if meta is not None:
                    agent_name = meta.name
                    messages.append(
                        AIMessage(content=f"[create:iter{iteration}:todo-{todo_id}] {status}")
                    )
                else:
                    # Prefer generic research over any finance specialist.
                    if domain != "finance" and registry.exists("primary_deep_agent"):
                        agent_name = "primary_deep_agent"
                        _flow(
                            "todo",
                            f"TODO[{todo_id}] create skipped — falling back to primary_deep_agent",
                        )
                    else:
                        agent_name = _resolve_agent_for_todo(
                            todo, registry, flows_manifest, user_task=task
                        )
                    messages.append(
                        AIMessage(
                            content=f"[create:iter{iteration}:todo-{todo_id}-reuse] {status}"
                        )
                    )

            if not agent_name or not registry.exists(agent_name):
                msg = (
                    f"I don't know how to complete this step with the current "
                    f"sub-agent mix: {todo['description']}"
                )
                todo["status"] = "unknown"
                todo["result"] = msg
                todo["assigned_agent"] = agent_name or ""
                results[todo_id] = msg
                prior_context_parts.append(f"TODO[{todo_id}] UNKNOWN: {msg}")
                _flow("todo", f"TODO[{todo_id}] blocked — no agent")
                messages.append(AIMessage(content=f"[todo-{todo_id}] {msg}"))
                continue

            # Final domain guard: never run finance agents on non-finance tasks.
            meta = next((m for m in registry.all_metadata() if m.name == agent_name), None)
            task_domain = _classify_task_domain(task)
            if meta and _agent_looks_finance(
                meta.name, meta.description, meta.capabilities
            ) and task_domain != "finance":
                proposed = _proposed_name_for_domain(task_domain, notes)
                _flow(
                    "todo",
                    f"TODO[{todo_id}] blocking finance agent '{agent_name}' on "
                    f"{task_domain} task — CREATE_NEEDED '{proposed}'",
                )
                meta_new, status = create_sub_agent_with_critic_loop(
                    f"MACRO {task_domain} domain agent.\n"
                    f"Entities (exemplars): {', '.join(analysis.get('entities') or [])}\n"
                    f"Entity types: {', '.join(analysis.get('entity_types') or [])}\n"
                    f"Domain briefing:\n{_format_topic_analysis(analysis) if analysis else '(none)'}\n"
                    f"Step:\n{todo['description']}\n"
                    "NOTE: Prefer durable domain coverage over one-shot user framing.",
                    proposed,
                    registry,
                    flows_manifest=flows_manifest,
                    topic_context=analysis,
                )
                if meta_new is None:
                    if registry.exists("primary_deep_agent"):
                        agent_name = "primary_deep_agent"
                        _flow(
                            "todo",
                            f"TODO[{todo_id}] domain create failed — using primary_deep_agent",
                        )
                    else:
                        msg = (
                            "I don't know — no suitable non-finance specialist exists yet "
                            f"for: {todo['description']}"
                        )
                        todo["status"] = "unknown"
                        todo["result"] = msg
                        results[todo_id] = msg
                        prior_context_parts.append(f"TODO[{todo_id}] UNKNOWN: {msg}")
                        messages.append(AIMessage(content=f"[todo-{todo_id}] {msg}"))
                        continue
                else:
                    agent_name = meta_new.name
                    messages.append(
                        AIMessage(content=f"[create:iter{iteration}:domain] {status}")
                    )

            todo["assigned_agent"] = agent_name
            todo["status"] = "running"
            compiled = registry.get_compiled(agent_name)
            if compiled is None:
                msg = f"I don't know — agent '{agent_name}' failed to compile."
                todo["status"] = "blocked"
                todo["result"] = msg
                results[todo_id] = msg
                prior_context_parts.append(f"TODO[{todo_id}] BLOCKED: {msg}")
                _flow("todo", f"TODO[{todo_id}] compile failure for {agent_name}")
                continue

            prior = "\n\n".join(prior_context_parts) if prior_context_parts else "(none yet)"
            prompt = (
                f"Original user task:\n{task}\n\n"
                f"Your TODO (do ONLY this step):\n{todo['description']}\n\n"
                f"Prior context / draft material:\n{prior}\n\n"
                f"Satisfy the ORIGINAL user task constraints (e.g. weekly = cover "
                f"all 7 days including rest). If evidence is insufficient, say "
                f"you don't know — do not invent."
            )
            _flow("todo", f"TODO[{todo_id}] invoking '{agent_name}'...")
            result = compiled.invoke({"messages": [HumanMessage(content=prompt)]})
            output_text = result["messages"][-1].content
            if not isinstance(output_text, str):
                output_text = str(output_text)

            todo["status"] = "done"
            todo["result"] = output_text
            results[todo_id] = output_text
            prior_context_parts.append(
                f"TODO[{todo_id}] agent={agent_name} result:\n{output_text}"
            )
            _flow(
                "todo",
                f"TODO[{todo_id}] done by '{agent_name}' ({len(output_text)} chars)",
            )
            messages.append(
                AIMessage(
                    content=output_text,
                    name=f"{agent_name}::iter{iteration}::todo{todo_id}",
                )
            )

        return {
            "todos": todos,
            "todo_results": results,
            "messages": messages,
        }

    def draft_synthesize_node(state: SupervisorState) -> dict:
        task = state["task"]
        iteration = int(state.get("validation_iteration") or 1)
        todos = state.get("todos") or []
        prior = state.get("best_output") or ""
        _flow("draft", f"iter={iteration} synthesizing draft from TODO results...")
        response = synthesizer.invoke(
            [
                SystemMessage(content=DRAFT_SYNTH_PROMPT),
                HumanMessage(
                    content=(
                        f"Original user task:\n{task}\n\n"
                        f"Prior best draft (may be empty):\n{prior or '(none)'}\n\n"
                        f"TODO results:\n{_todo_results_block(todos)}"
                    )
                ),
            ]
        )
        draft = response.content if isinstance(response.content, str) else str(response.content)
        _flow("draft", f"iter={iteration} draft ready ({len(draft)} chars)")
        return {
            "draft_output": draft,
            "messages": [
                AIMessage(content=draft, name=f"draft_iter{iteration}")
            ],
        }

    def validate_query_node(state: SupervisorState) -> dict:
        task = state["task"]
        draft = state.get("draft_output") or ""
        iteration = int(state.get("validation_iteration") or 1)
        _flow("validate", f"iter={iteration}/{max_iters} checking draft vs user query...")
        verdict: QueryValidation = validator.invoke(
            [
                SystemMessage(content=VALIDATE_PROMPT),
                HumanMessage(
                    content=(
                        f"ORIGINAL user query:\n{task}\n\n"
                        f"DRAFT answer:\n{draft}"
                    )
                ),
            ]
        )
        _flow(
            "validate",
            f"iter={iteration} verdict={verdict.verdict} | gaps={len(verdict.gaps)} | "
            f"{verdict.summary}",
        )
        for i, gap in enumerate(verdict.gaps, 1):
            _flow("validate", f"  gap[{i}]: {gap}")

        best = draft
        # Keep prior best if new draft is empty.
        if not draft.strip():
            best = state.get("best_output") or draft

        update = {
            "validation_verdict": verdict.verdict,
            "validation_gaps": list(verdict.gaps),
            "best_output": best if verdict.verdict == "PASS" else (draft or state.get("best_output")),
            "messages": [
                AIMessage(
                    content=(
                        f"[validate:iter{iteration}] {verdict.verdict}: {verdict.summary}"
                    )
                )
            ],
        }
        # On PASS, promote draft to best.
        if verdict.verdict == "PASS":
            update["best_output"] = draft
        elif draft.strip():
            # Still keep latest draft as working best for refinement input.
            update["best_output"] = draft
        return update

    def route_after_validate(
        state: SupervisorState,
    ) -> Literal["plan_todos_node", "finalize_node"]:
        iteration = int(state.get("validation_iteration") or 1)
        verdict = (state.get("validation_verdict") or "FAIL").upper()
        if verdict == "PASS" or iteration >= max_iters:
            _flow(
                "route",
                f"stopping validation loop (verdict={verdict}, iter={iteration}/{max_iters})",
            )
            return "finalize_node"
        _flow("route", f"refinement needed — looping to plan (next iter {iteration + 1})")
        return "plan_todos_node"

    def finalize_node(state: SupervisorState) -> dict:
        task = state["task"]
        best = state.get("best_output") or state.get("draft_output") or ""
        gaps = state.get("validation_gaps") or []
        verdict = state.get("validation_verdict") or "FAIL"
        iteration = int(state.get("validation_iteration") or 1)
        _flow("finalize", f"producing final answer (verdict={verdict}, iters={iteration})")
        response = synthesizer.invoke(
            [
                SystemMessage(content=FINALIZE_PROMPT),
                HumanMessage(
                    content=(
                        f"Original user query:\n{task}\n\n"
                        f"Validation verdict: {verdict} after {iteration} iteration(s)\n"
                        f"Remaining gaps:\n"
                        + ("\n".join(f"- {g}" for g in gaps) if gaps else "- (none)")
                        + f"\n\nBest draft:\n{best}"
                    )
                ),
            ]
        )
        final = response.content if isinstance(response.content, str) else str(response.content)
        _flow("finalize", f"done ({len(final)} chars)")
        return {
            "final_output": final,
            "messages": [AIMessage(content=final, name="supervisor_final")],
        }

    graph = StateGraph(SupervisorState)
    graph.add_node("analyze_topic_node", analyze_topic_node)
    graph.add_node("plan_todos_node", plan_todos_node)
    graph.add_node("execute_todos_node", execute_todos_node)
    graph.add_node("draft_synthesize_node", draft_synthesize_node)
    graph.add_node("validate_query_node", validate_query_node)
    graph.add_node("finalize_node", finalize_node)

    graph.add_edge(START, "analyze_topic_node")
    graph.add_edge("analyze_topic_node", "plan_todos_node")
    graph.add_edge("plan_todos_node", "execute_todos_node")
    graph.add_edge("execute_todos_node", "draft_synthesize_node")
    graph.add_edge("draft_synthesize_node", "validate_query_node")
    graph.add_conditional_edges(
        "validate_query_node",
        route_after_validate,
        {
            "plan_todos_node": "plan_todos_node",
            "finalize_node": "finalize_node",
        },
    )
    graph.add_edge("finalize_node", END)

    return graph.compile()
