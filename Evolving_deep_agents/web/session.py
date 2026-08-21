"""In-memory session store for Q/A + satisfaction (follow-up memory)."""
from __future__ import annotations

import threading
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class SessionMemory:
    session_id: str
    last_question: Optional[str] = None
    last_answer: Optional[str] = None
    satisfaction: Optional[str] = None  # unsatisfactory | ok | satisfactory
    awaiting_feedback: bool = False
    steps: List[Dict[str, str]] = field(default_factory=list)
    matching_flows: List[str] = field(default_factory=list)
    matching_agents: List[str] = field(default_factory=list)
    created_agents: List[str] = field(default_factory=list)
    created_or_updated_flows: List[str] = field(default_factory=list)
    domain: str = ""
    entities: List[str] = field(default_factory=list)
    history: List[Dict[str, Any]] = field(default_factory=list)

    def to_public(self) -> Dict[str, Any]:
        return asdict(self)


class SessionStore:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._sessions: Dict[str, SessionMemory] = {}

    def get_or_create(self, session_id: Optional[str] = None) -> SessionMemory:
        with self._lock:
            if session_id and session_id in self._sessions:
                return self._sessions[session_id]
            sid = session_id or str(uuid.uuid4())
            mem = SessionMemory(session_id=sid)
            self._sessions[sid] = mem
            return mem

    def get(self, session_id: str) -> Optional[SessionMemory]:
        with self._lock:
            return self._sessions.get(session_id)

    def save_run(
        self,
        session_id: str,
        question: str,
        summary: Dict[str, Any],
    ) -> SessionMemory:
        with self._lock:
            mem = self._sessions.setdefault(
                session_id, SessionMemory(session_id=session_id)
            )
            mem.last_question = question
            mem.last_answer = summary.get("answer") or ""
            mem.satisfaction = None
            mem.awaiting_feedback = True
            mem.steps = list(summary.get("steps") or [])
            mem.matching_flows = list(summary.get("matching_flows") or [])
            mem.matching_agents = list(summary.get("matching_agents") or [])
            mem.created_agents = list(summary.get("created_agents") or [])
            mem.created_or_updated_flows = list(
                summary.get("created_or_updated_flows") or []
            )
            mem.domain = summary.get("domain") or ""
            mem.entities = list(summary.get("entities") or [])
            return mem

    def save_feedback(self, session_id: str, satisfaction: str) -> SessionMemory:
        with self._lock:
            mem = self._sessions.get(session_id)
            if mem is None:
                raise KeyError(session_id)
            if not mem.awaiting_feedback:
                raise ValueError("No answer awaiting feedback for this session.")
            mem.satisfaction = satisfaction
            mem.awaiting_feedback = False
            mem.history.append(
                {
                    "question": mem.last_question,
                    "answer": mem.last_answer,
                    "satisfaction": satisfaction,
                    "domain": mem.domain,
                    "matching_flows": list(mem.matching_flows),
                    "created_agents": list(mem.created_agents),
                }
            )
            return mem


def build_followup_task(mem: SessionMemory, question: str) -> str:
    """Attach prior Q/A/satisfaction so the graph can use it as memory."""
    if not mem.last_question or mem.satisfaction is None:
        return question
    return (
        "FOLLOW-UP CONTEXT (prior turn memory):\n"
        f"- Prior question: {mem.last_question}\n"
        f"- Prior answer: {mem.last_answer}\n"
        f"- User satisfaction with prior answer: {mem.satisfaction}\n"
        f"- Prior domain: {mem.domain or '(unknown)'}\n"
        f"- Prior flows used: {', '.join(mem.matching_flows) or '(none)'}\n"
        f"- Agents created last turn: {', '.join(mem.created_agents) or '(none)'}\n\n"
        "CURRENT QUESTION:\n"
        f"{question}"
    )
