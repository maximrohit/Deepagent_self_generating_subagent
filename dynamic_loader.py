"""
Handles turning a .py file on disk into a live, invokable agent object --
used identically by:
  - Registry.warm_boot()        (loading historical sub-agents at startup)
  - Registry.register_from_file() (loading a brand-new sub-agent at runtime)

This is the only place `importlib` is touched, so the "dynamic compilation"
mechanism has one code path regardless of when/why it's invoked.
"""
from __future__ import annotations

import importlib.util
import inspect
import sys
from pathlib import Path
from types import ModuleType

from config import SUB_AGENT_DEFAULT_MODEL
from utility.tavily_tools import select_relevant_search_tools


def load_module_from_file(path: Path, module_name: str, force_reload: bool = False) -> ModuleType:
    """
    Dynamically imports the given .py file as `module_name`, without needing
    it to be on sys.path or part of a package -- this is what lets us treat
    sub_agents/*.py as a plugin directory that grows at runtime.
    """
    if not force_reload and module_name in sys.modules:
        return sys.modules[module_name]

    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not create import spec for {path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module  # register before exec so intra-module imports work
    try:
        spec.loader.exec_module(module)
    except Exception:
        # Don't leave a half-initialized module cached under this name.
        sys.modules.pop(module_name, None)
        raise
    return module


def _relevant_tools_for_module(module: ModuleType):
    text = " ".join(
        [
            str(getattr(module, "AGENT_NAME", "")),
            str(getattr(module, "AGENT_DESCRIPTION", "")),
            " ".join(getattr(module, "AGENT_CAPABILITIES", []) or []),
        ]
    )
    return select_relevant_search_tools(text)


def instantiate_agent(module: ModuleType):
    """
    Calls the module's build_agent() to produce the actual runnable LangGraph
    agent (or deepagents agent). Isolated here so we can add e.g. build-time
    validation, resource limits, or sandboxing later in one place.

    Injects topic-relevant Tavily search tools via `extra_tools` when the
    sub-agent's build_agent() accepts that parameter.
    """
    relevant = _relevant_tools_for_module(module)
    build_agent = module.build_agent
    kwargs = {"model_name": SUB_AGENT_DEFAULT_MODEL}
    if "extra_tools" in inspect.signature(build_agent).parameters:
        kwargs["extra_tools"] = relevant
    return build_agent(**kwargs)
