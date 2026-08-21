"""Extract user-facing pipeline steps from a supervisor graph result."""
from __future__ import annotations

import re
from typing import Any, Dict, List


def _msg_text(msg: Any) -> str:
    content = getattr(msg, "content", "") or ""
    return content if isinstance(content, str) else str(content)


def _msg_name(msg: Any) -> str:
    return getattr(msg, "name", None) or msg.__class__.__name__


def extract_run_summary(result: Dict[str, Any]) -> Dict[str, Any]:
    """Pull answer + important create/manifest/routing steps from invoke result."""
    analysis = result.get("topic_analysis") or {}
    todos = result.get("todos") or []
    messages = result.get("messages") or []

    matching_flows = list(analysis.get("matching_flows") or [])
    matching_agents = list(analysis.get("matching_agents") or [])
    created_agents: List[str] = []
    created_or_updated_flows: List[str] = []
    steps: List[Dict[str, str]] = []

    if analysis:
        domain = analysis.get("domain") or "?"
        entities = ", ".join(analysis.get("entities") or []) or "(none)"
        steps.append(
            {
                "kind": "analyze",
                "title": "Topic analysis",
                "detail": (
                    f"domain={domain} | entities={entities} | "
                    f"reuse={analysis.get('reuse_decision') or '?'} | "
                    f"create_needed={bool(analysis.get('create_needed'))}"
                ),
            }
        )

    if matching_flows:
        steps.append(
            {
                "kind": "manifest",
                "title": "Flows matched from manifesto",
                "detail": ", ".join(matching_flows),
            }
        )
    else:
        steps.append(
            {
                "kind": "manifest",
                "title": "Flows matched from manifesto",
                "detail": "(none — may create or use ad-hoc agents)",
            }
        )

    if matching_agents:
        steps.append(
            {
                "kind": "agents",
                "title": "Agents matched",
                "detail": ", ".join(matching_agents),
            }
        )

    for todo in todos:
        agent = todo.get("assigned_agent") or "(unassigned)"
        flow_id = todo.get("flow_id") or ""
        notes = todo.get("notes") or ""
        status = todo.get("status") or ""
        detail = f"agent={agent}"
        if flow_id:
            detail += f" | flow={flow_id}"
        if notes:
            detail += f" | notes={notes}"
        if status:
            detail += f" | status={status}"
        steps.append(
            {
                "kind": "todo",
                "title": f"TODO[{todo.get('id')}] {todo.get('description') or ''}".strip(),
                "detail": detail,
            }
        )
        if (
            flow_id
            and flow_id not in matching_flows
            and flow_id not in created_or_updated_flows
        ):
            created_or_updated_flows.append(flow_id)

    for msg in messages:
        text = _msg_text(msg)
        name = _msg_name(msg)
        lower = text.lower()

        if text.startswith("[create:") or "Created missing-component" in text:
            steps.append(
                {"kind": "create", "title": "Sub-agent creation", "detail": text[:800]}
            )
            for m in re.finditer(r"'([a-zA-Z0-9_]+_agent)'", text):
                agent_name = m.group(1)
                if agent_name not in created_agents:
                    created_agents.append(agent_name)
            if not created_agents:
                for token in text.replace("'", " ").replace('"', " ").split():
                    if token.endswith("_agent") and token not in created_agents:
                        created_agents.append(token)
        elif "upserted flow" in lower or (
            "flow" in lower and "manifest" in lower
        ):
            steps.append(
                {
                    "kind": "manifest",
                    "title": "Flows manifesto update",
                    "detail": text[:800],
                }
            )
            for m in re.finditer(r"'([a-zA-Z0-9_]+)'", text):
                fid = m.group(1)
                if fid not in matching_flows and fid not in created_or_updated_flows:
                    if "flow" in fid or fid.endswith("_pipeline"):
                        created_or_updated_flows.append(fid)
        elif name == "topic_analysis":
            continue
        elif (
            text.startswith("[todo-")
            or text.startswith("[analyze]")
            or text.startswith("[plan]")
        ):
            steps.append({"kind": "trace", "title": name, "detail": text[:600]})

    answer = (
        result.get("final_output")
        or result.get("best_output")
        or result.get("draft_output")
        or ""
    )

    return {
        "answer": answer if isinstance(answer, str) else str(answer),
        "steps": steps,
        "matching_flows": matching_flows,
        "matching_agents": matching_agents,
        "created_agents": created_agents,
        "created_or_updated_flows": created_or_updated_flows,
        "domain": analysis.get("domain") or "",
        "entities": list(analysis.get("entities") or []),
        "create_needed": bool(analysis.get("create_needed")),
        "proposed_agent_name": analysis.get("proposed_agent_name") or "",
        "validation_verdict": result.get("validation_verdict") or "",
        "validation_iteration": result.get("validation_iteration") or 1,
    }
