# Self-Evolving Multi-Agent System (LangGraph)

## Layout

```
deep_agent_system/
├── config.py            paths + model names
├── state.py              SupervisorState TypedDict
├── registry.py           SubAgentRegistry: manifest I/O, warm boot, hot registration
├── dynamic_loader.py      importlib-based module loading (single code path for
│                          both warm boot and runtime registration)
├── code_generator.py      create_sub_agent_tool: LLM writes + validates + persists
│                          + registers a new sub-agent module
├── graph.py               the StateGraph: router_node → existing_agent_node
│                          | create_and_register_node → execute_new_agent_node
├── main.py                warm boot + run loop entrypoint
└── sub_agents/            PERSISTENT STORAGE — one .py file per sub-agent
    ├── manifest.json      metadata index (name/description/capabilities/hash)
    └── regex_helper_agent.py   example hand-written agent (proves warm boot)
```

## How each requirement is satisfied

**1. Dynamic Router & Evaluator** — `graph.router_node`. The system prompt is
built from `registry.describe_for_router()`, which emits every registered
agent's *full description and capability list*, not just its name. The LLM
returns a structured `RouteDecision` (`action`, `agent_name`, `justification`)
via `with_structured_output`, so the match is auditable — you can log
`justification` to see *why* it did or didn't reuse an agent.

**2. Code Generation & Tool Integration** — `code_generator.create_sub_agent_tool`.
When `route_decision == "create"`, `graph.create_and_register_node` invokes
this tool with the capability-gap description. The tool prompts a codegen
model with a strict contract (`AGENT_NAME`, `AGENT_DESCRIPTION`,
`AGENT_CAPABILITIES`, `build_agent()`), extracts the code block, and
`ast.parse`s it plus checks for the required top-level names *before* writing
anything to disk — a malformed generation fails the tool call instead of
corrupting storage.

**3. Dynamic Compilation & Registration** — `registry.register_from_file()`
calls `dynamic_loader.load_module_from_file()` (an `importlib.util.spec_from_file_location`
load with no dependency on the file being on `sys.path`), then
`instantiate_agent()` calls the module's `build_agent()` to compile it. The
resulting object is stored in `registry._compiled[name]` on the **same
registry instance the running graph closes over** — so the next router turn
in the same process sees it immediately. No app restart, no graph rebuild.

**4. Persistence & Warm Boot** — `registry.warm_boot()`, called once in
`main.bootstrap()`. It diffs `sub_agents/*.py` against `manifest.json` by
content hash: unchanged files reuse cached metadata (cheap), changed/new
files get re-imported to refresh metadata, and it eagerly compiles every
agent so first-use latency is zero. This is exactly why `create_sub_agent_tool`
never needs to regenerate an agent that already exists on disk — restart and
warm boot converge to the same registered state as before the restart.

## Adapting to the real `deepagents` library

This implementation uses `langgraph.prebuilt.create_react_agent` for
`build_agent()` so it's runnable with just `langgraph`+`langchain`. To use
LangChain's `deepagents` package instead (planning tool, virtual filesystem,
built-in sub-agent delegation), swap the body of `build_agent()`:

```python
from deepagents import create_deep_agent

def build_agent(model_name: str = "claude-sonnet-4-6"):
    return create_deep_agent(
        tools=[...],
        instructions="...",
        model=model_name,
    )
```

Everything else — registry, warm boot, dynamic loader, router, codegen tool —
is agnostic to which LangGraph-compatible construction function a sub-agent's
`build_agent()` calls, since the only contract the registry relies on is
`.invoke({"messages": [...]})`.

You could also flip the architecture around: instead of the supervisor
StateGraph shown in `graph.py`, register each compiled sub-agent as a
`deepagents` sub-agent (`subagents=[...]` param on `create_deep_agent`) and
let `deepagents`' own planner do routing — `registry.describe_for_router()`
and `registry.get_compiled()` are exactly what you'd feed into that
`subagents` list at construction time, and `create_sub_agent_tool` would then
need to trigger a rebuild of the top-level deep agent with the new subagent
appended.

## Running

```bash
pip install langgraph langchain langchain-anthropic pydantic
export ANTHROPIC_API_KEY=...
python main.py "Write a regex that validates US zip codes"      # routes to existing regex_helper_agent
python main.py "Summarize this SEC 10-K filing's risk factors"  # creates a new sub-agent, persists it,
                                                                  # then runs it
python main.py "Summarize this other 10-K filing"                # reuses the sub-agent created above
                                                                  # even across a fresh process
```
