"""
FlowManifest: tracks multi-sub-agent workflows and the problem classes they solve.

Unlike sub_agents/manifest.json (per-agent metadata), this catalog describes
*compositions*: ordered stages where each sub-agent plays a specific functional
role. It is the source of truth for "what problems can we currently solve with
the mix of sub-agents we have?"
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set


@dataclass
class FlowStage:
    order: int
    role: str
    agent: str
    function: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "order": self.order,
            "role": self.role,
            "agent": self.agent,
            "function": self.function,
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "FlowStage":
        return FlowStage(
            order=int(d["order"]),
            role=str(d["role"]),
            agent=str(d["agent"]),
            function=str(d["function"]),
        )


@dataclass
class FlowDefinition:
    flow_id: str
    name: str
    description: str
    problem_types: List[str]
    example_queries: List[str]
    stages: List[FlowStage]
    # Filled at sync time against the live sub-agent registry.
    status: str = "unknown"  # active | incomplete | unknown
    missing_agents: List[str] = field(default_factory=list)

    @property
    def required_agents(self) -> List[str]:
        return [s.agent for s in sorted(self.stages, key=lambda x: x.order)]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "flow_id": self.flow_id,
            "name": self.name,
            "description": self.description,
            "problem_types": list(self.problem_types),
            "example_queries": list(self.example_queries),
            "stages": [s.to_dict() for s in sorted(self.stages, key=lambda x: x.order)],
            "status": self.status,
            "missing_agents": list(self.missing_agents),
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "FlowDefinition":
        stages = [FlowStage.from_dict(s) for s in d.get("stages", [])]
        return FlowDefinition(
            flow_id=str(d["flow_id"]),
            name=str(d["name"]),
            description=str(d.get("description", "")),
            problem_types=list(d.get("problem_types", [])),
            example_queries=list(d.get("example_queries", [])),
            stages=stages,
            status=str(d.get("status", "unknown")),
            missing_agents=list(d.get("missing_agents", [])),
        )


class FlowManifest:
    """
    Load / query / sync the flows catalog.

    Persistence format (flows/manifest.json):
      {
        "version": 1,
        "description": "...",
        "updated_at": "...",
        "flows": [ {flow_id, name, stages:[{role, agent, function}], ...} ]
      }
    """

    def __init__(self, path: Path):
        self.path = path
        self.version: int = 1
        self.description: str = ""
        self.updated_at: str = ""
        self._flows: Dict[str, FlowDefinition] = {}

    # ------------------------------------------------------------------ #
    # I/O
    # ------------------------------------------------------------------ #
    def load(self) -> "FlowManifest":
        if not self.path.exists():
            self._flows = {}
            return self
        with open(self.path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.version = int(data.get("version", 1))
        self.description = str(data.get("description", ""))
        self.updated_at = str(data.get("updated_at", ""))
        self._flows = {
            item["flow_id"]: FlowDefinition.from_dict(item)
            for item in data.get("flows", [])
        }
        return self

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.updated_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        payload = {
            "version": self.version,
            "description": self.description,
            "updated_at": self.updated_at,
            "flows": [flow.to_dict() for flow in self.all_flows()],
        }
        tmp = self.path.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        tmp.replace(self.path)

    # ------------------------------------------------------------------ #
    # Sync against live sub-agent registry
    # ------------------------------------------------------------------ #
    def sync_with_agents(self, available_agent_names: Iterable[str]) -> Dict[str, str]:
        """
        Mark each flow active/incomplete based on whether all staged agents exist.
        Returns {flow_id: status}.
        """
        available: Set[str] = set(available_agent_names)
        statuses: Dict[str, str] = {}
        for flow in self._flows.values():
            missing = [a for a in flow.required_agents if a not in available]
            flow.missing_agents = missing
            flow.status = "incomplete" if missing else "active"
            statuses[flow.flow_id] = flow.status
        return statuses

    # ------------------------------------------------------------------ #
    # Accessors / coverage
    # ------------------------------------------------------------------ #
    def get(self, flow_id: str) -> Optional[FlowDefinition]:
        return self._flows.get(flow_id)

    def all_flows(self) -> List[FlowDefinition]:
        return sorted(self._flows.values(), key=lambda f: f.flow_id)

    def active_flows(self) -> List[FlowDefinition]:
        return [f for f in self.all_flows() if f.status == "active"]

    def incomplete_flows(self) -> List[FlowDefinition]:
        return [f for f in self.all_flows() if f.status == "incomplete"]

    def solvable_problem_types(self, *, active_only: bool = True) -> List[str]:
        """Flattened list of problem classes covered by (active) flows."""
        flows = self.active_flows() if active_only else self.all_flows()
        seen: Set[str] = set()
        out: List[str] = []
        for flow in flows:
            for p in flow.problem_types:
                if p not in seen:
                    seen.add(p)
                    out.append(p)
        return out

    def flows_using_agent(self, agent_name: str) -> List[FlowDefinition]:
        return [f for f in self.all_flows() if agent_name in f.required_agents]

    def find_flows_for_query(self, query: str) -> List[FlowDefinition]:
        """
        Lightweight keyword overlap over problem_types + example_queries + name.
        Intended as a tracking/routing hint, not a full semantic matcher.
        """
        tokens = {t for t in query.lower().replace("/", " ").split() if len(t) > 2}
        if not tokens:
            return []
        scored: List[tuple[int, FlowDefinition]] = []
        for flow in self.active_flows():
            blob = " ".join(
                [flow.name, flow.description]
                + flow.problem_types
                + flow.example_queries
                + [s.role + " " + s.function for s in flow.stages]
            ).lower()
            score = sum(1 for t in tokens if t in blob)
            if score:
                scored.append((score, flow))
        scored.sort(key=lambda x: (-x[0], x[1].flow_id))
        return [f for _, f in scored]

    def upsert_flow(self, flow: FlowDefinition) -> None:
        self._flows[flow.flow_id] = flow

    def describe_coverage(self) -> str:
        """Human-readable coverage report for logs / router context."""
        lines: List[str] = []
        active = self.active_flows()
        incomplete = self.incomplete_flows()
        lines.append(
            f"Flow coverage: {len(active)} active / {len(self._flows)} total "
            f"({len(incomplete)} incomplete)"
        )
        for flow in active:
            roles = " -> ".join(
                f"{s.role}[{s.agent}]" for s in sorted(flow.stages, key=lambda x: x.order)
            )
            lines.append(f"* {flow.flow_id}: {flow.name}")
            lines.append(f"  pipeline: {roles}")
            for p in flow.problem_types:
                lines.append(f"  - solves: {p}")
        if incomplete:
            lines.append("Incomplete flows (missing agents):")
            for flow in incomplete:
                lines.append(
                    f"* {flow.flow_id}: missing {flow.missing_agents}"
                )
        return "\n".join(lines)

    def describe_for_router(self) -> str:
        """Compact block the supervisor router can use to prefer known flows."""
        if not self.active_flows():
            return "(no active multi-agent flows registered)"
        blocks: List[str] = []
        for flow in self.active_flows():
            stages = "\n".join(
                f"      {s.order}. role={s.role} agent={s.agent} — {s.function}"
                for s in sorted(flow.stages, key=lambda x: x.order)
            )
            problems = "\n".join(f"      - {p}" for p in flow.problem_types)
            examples = "\n".join(f"      - {q}" for q in flow.example_queries[:3])
            blocks.append(
                f"* flow_id: {flow.flow_id}\n"
                f"  name: {flow.name}\n"
                f"  description: {flow.description}\n"
                f"  stages:\n{stages}\n"
                f"  problem_types:\n{problems}\n"
                f"  example_queries:\n{examples}"
            )
        return "\n\n".join(blocks)
