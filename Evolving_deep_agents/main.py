"""
Entrypoint.

    python main.py "some task"          # single-shot
    python main.py                      # interactive loop

Startup sequence (this is the "warm boot"):
  1. Instantiate SubAgentRegistry pointed at the persistent sub_agents/ dir.
  2. registry.warm_boot() -- scans sub_agents/*.py, cross-checks manifest.json,
     dynamically imports + compiles each one, and registers it in memory.
  3. Load flows/manifest.json and sync coverage against registered agents.
  4. build_supervisor_graph(registry, flows) -- analyze topic (entities/domain/
     do-don't + manifest check), plan TODOs, match to agents/flows, invoke
     sequentially with prior results as context, then synthesize the final answer.
"""
from __future__ import annotations

import sys

# config loads .env (TAVILY_API_KEY) before any Tavily tools are constructed.
from config import FLOWS_MANIFEST_PATH, MANIFEST_PATH, SUB_AGENTS_DIR
from flow_manifest import FlowManifest
from graph import build_supervisor_graph
from registry import SubAgentRegistry


def bootstrap() -> tuple[SubAgentRegistry, object, FlowManifest]:
    print("[flow:boot] starting warm boot...", flush=True)
    registry = SubAgentRegistry(sub_agents_dir=SUB_AGENTS_DIR, manifest_path=MANIFEST_PATH)
    loaded = registry.warm_boot(eager_compile=True)
    print(f"[flow:boot] loaded {len(loaded)} historical sub-agent(s): {loaded}", flush=True)

    print("[flow:boot] loading flows manifest...", flush=True)
    flows = FlowManifest(FLOWS_MANIFEST_PATH).load()
    statuses = flows.sync_with_agents(m.name for m in registry.all_metadata())
    active = sum(1 for s in statuses.values() if s == "active")
    print(
        f"[flow:boot] flows synced: {active}/{len(statuses)} active",
        flush=True,
    )
    print(flows.describe_coverage(), flush=True)

    print("[flow:boot] compiling supervisor graph...", flush=True)
    app = build_supervisor_graph(registry, flows_manifest=flows)
    print("[flow:boot] ready", flush=True)
    return registry, app, flows


def run_task(app, task: str) -> str:
    print(f"[flow:task] invoke start | task={task!r}", flush=True)
    result = app.invoke({"task": task, "messages": []})
    print("[flow:task] invoke complete — message trace:", flush=True)
    for m in result["messages"]:
        name = getattr(m, "name", None) or m.__class__.__name__
        print(f"  [{name}] {m.content}\n")
    print("[flow:task] done", flush=True)
    return result.get("final_output", "")


def main():
    registry, app, flows = bootstrap()
    _ = (registry, flows)  # available for future interactive commands

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
