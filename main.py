"""
Entrypoint.

    python main.py "some task"          # single-shot
    python main.py                      # interactive loop

Startup sequence (this is the "warm boot"):
  1. Instantiate SubAgentRegistry pointed at the persistent sub_agents/ dir.
  2. registry.warm_boot() -- scans sub_agents/*.py, cross-checks manifest.json,
     dynamically imports + compiles each one, and registers it in memory.
     This is what prevents redundant LLM code-generation across restarts:
     any sub-agent that already exists on disk is simply reloaded, never rewritten.
  3. build_supervisor_graph(registry) -- wires the StateGraph; all of its nodes
     close over this *same* registry instance, so anything registered later
     at runtime (via create_sub_agent_tool) is immediately visible to routing
     on the very next invocation, with no restart of the graph/app.
"""
from __future__ import annotations

import sys

# config loads .env (TAVILY_API_KEY) before any Tavily tools are constructed.
from config import MANIFEST_PATH, SUB_AGENTS_DIR
from graph import build_supervisor_graph
from registry import SubAgentRegistry


def bootstrap() -> tuple[SubAgentRegistry, object]:
    registry = SubAgentRegistry(sub_agents_dir=SUB_AGENTS_DIR, manifest_path=MANIFEST_PATH)
    loaded = registry.warm_boot(eager_compile=True)
    print(f"[warm boot] loaded {len(loaded)} historical sub-agent(s): {loaded}")

    app = build_supervisor_graph(registry)
    return registry, app


def run_task(app, task: str) -> str:
    result = app.invoke({"task": task, "messages": []})
    for m in result["messages"]:
        name = getattr(m, "name", None) or m.__class__.__name__
        print(f"  [{name}] {m.content}\n")
    return result.get("final_output", "")


def main():
    registry, app = bootstrap()

    if len(sys.argv) > 1:
        task = " ".join(sys.argv[1:])
        print(f"\n>>> TASK: {task}\n")
        run_task(app, task)
        return

    print("Interactive mode. Ctrl-C to exit.")
    while True:
        try:
            task = input("\n> ")
        except (EOFError, KeyboardInterrupt):
            break
        if not task.strip():
            continue
        run_task(app, task)


if __name__ == "__main__":
    main()
