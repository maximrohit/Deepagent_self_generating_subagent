"""
FastAPI UI for Evolving Deep Agents.

    uvicorn web.app:app --host 127.0.0.1 --port 8000

Or:

    python -m web.app
"""
from __future__ import annotations

import asyncio
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, Literal, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from main import bootstrap  # noqa: E402
from web.extract import extract_run_summary  # noqa: E402
from web.session import SessionStore, build_followup_task  # noqa: E402

STATIC_DIR = Path(__file__).resolve().parent / "static"

_registry = None
_graph = None
_flows = None
_sessions = SessionStore()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global _registry, _graph, _flows
    print("[ui] warm boot...", flush=True)
    _registry, _graph, _flows = bootstrap()
    print("[ui] ready at /", flush=True)
    yield


app = FastAPI(title="Evolving Deep Agents UI", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1)
    session_id: Optional[str] = None


class FeedbackRequest(BaseModel):
    session_id: str
    satisfaction: Literal["unsatisfactory", "ok", "satisfactory"]


@app.get("/", response_class=HTMLResponse)
async def index() -> HTMLResponse:
    return HTMLResponse((STATIC_DIR / "index.html").read_text(encoding="utf-8"))


@app.get("/api/health")
async def health() -> Dict[str, Any]:
    return {
        "ok": True,
        "agents": len(_registry.all_metadata()) if _registry else 0,
        "flows": len(_flows.all_flows()) if _flows else 0,
    }


@app.get("/api/session/{session_id}")
async def get_session(session_id: str) -> Dict[str, Any]:
    mem = _sessions.get(session_id)
    if mem is None:
        raise HTTPException(status_code=404, detail="Unknown session")
    return mem.to_public()


@app.post("/api/ask")
async def ask(body: AskRequest) -> JSONResponse:
    if _graph is None:
        raise HTTPException(status_code=503, detail="Graph not ready")

    question = body.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question is empty")

    mem = _sessions.get_or_create(body.session_id)
    if mem.awaiting_feedback:
        raise HTTPException(
            status_code=409,
            detail="Please rate the previous answer before asking another question.",
        )

    task = build_followup_task(mem, question)
    print(f"[ui] ask | session={mem.session_id} | q={question!r}", flush=True)

    try:
        result = await asyncio.to_thread(
            _graph.invoke, {"task": task, "messages": []}
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[ui] ask failed: {exc}", flush=True)
        raise HTTPException(status_code=500, detail=f"Run failed: {exc}") from exc

    if _flows is not None:
        try:
            _flows.load()
        except Exception:  # noqa: BLE001
            pass

    summary = extract_run_summary(result)
    if summary.get("create_needed") and not summary.get("created_agents"):
        proposed = summary.get("proposed_agent_name") or ""
        if proposed:
            summary["created_agents"] = [proposed]

    mem = _sessions.save_run(mem.session_id, question, summary)

    return JSONResponse(
        {
            "session_id": mem.session_id,
            "question": question,
            "answer": mem.last_answer,
            "steps": mem.steps,
            "matching_flows": mem.matching_flows,
            "matching_agents": mem.matching_agents,
            "created_agents": mem.created_agents,
            "created_or_updated_flows": mem.created_or_updated_flows,
            "domain": mem.domain,
            "entities": mem.entities,
            "awaiting_feedback": True,
            "can_ask_again": False,
            "satisfaction": None,
            "validation_verdict": summary.get("validation_verdict"),
            "validation_iteration": summary.get("validation_iteration"),
        }
    )


@app.post("/api/feedback")
async def feedback(body: FeedbackRequest) -> JSONResponse:
    try:
        mem = _sessions.save_feedback(body.session_id, body.satisfaction)
    except KeyError:
        raise HTTPException(status_code=404, detail="Unknown session") from None
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from None

    return JSONResponse(
        {
            "session_id": mem.session_id,
            "satisfaction": mem.satisfaction,
            "awaiting_feedback": False,
            "can_ask_again": True,
            "last_question": mem.last_question,
            "last_answer": mem.last_answer,
            "memory": {
                "question": mem.last_question,
                "answer": mem.last_answer,
                "satisfaction": mem.satisfaction,
            },
        }
    )


def main() -> None:
    import uvicorn

    uvicorn.run("web.app:app", host="127.0.0.1", port=8000, reload=False)


if __name__ == "__main__":
    main()
