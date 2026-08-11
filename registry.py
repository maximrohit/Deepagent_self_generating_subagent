"""
SubAgentRegistry is the single source of truth for which sub-agents exist,
what they're capable of, and how to run them.

It backs onto two things:
  1. manifest.json      -- lightweight metadata index (name/description/capabilities/
                            file path/content hash). Cheap to read on startup.
  2. sub_agents/*.py     -- the actual sub-agent code. Only imported (and therefore
                            executed) when we actually need to run/compile that agent.

Both hand-written and LLM-generated sub-agent files must expose this contract:

    AGENT_NAME: str
    AGENT_DESCRIPTION: str            # detailed, capability-level description
    AGENT_CAPABILITIES: list[str]     # bullet-style capability tags/sentences
    def build_agent(model_name: str = ...) -> Runnable: ...

`build_agent()` should return anything invokable with `.invoke({"messages": [...]})`
-- e.g. a compiled LangGraph graph from `create_react_agent` or `deepagents.create_deep_agent`.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from dynamic_loader import load_module_from_file, instantiate_agent


@dataclasses.dataclass
class SubAgentMetadata:
    name: str
    description: str
    capabilities: List[str]
    file_path: str
    content_hash: str
    created_at: float
    module_name: str

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "SubAgentMetadata":
        return SubAgentMetadata(**d)


class SubAgentRegistry:
    def __init__(self, sub_agents_dir: Path, manifest_path: Path):
        self.sub_agents_dir = sub_agents_dir
        self.manifest_path = manifest_path

        # name -> metadata (always populated, cheap)
        self._metadata: Dict[str, SubAgentMetadata] = {}
        # name -> compiled runnable agent (populated lazily / on registration)
        self._compiled: Dict[str, Any] = {}

    # ------------------------------------------------------------------ #
    # Manifest I/O
    # ------------------------------------------------------------------ #
    def _load_manifest(self) -> Dict[str, Any]:
        if not self.manifest_path.exists():
            return {}
        with open(self.manifest_path, "r") as f:
            return json.load(f)

    def _save_manifest(self) -> None:
        data = {name: meta.to_dict() for name, meta in self._metadata.items()}
        tmp_path = self.manifest_path.with_suffix(".tmp")
        with open(tmp_path, "w") as f:
            json.dump(data, f, indent=2)
        tmp_path.replace(self.manifest_path)  # atomic on POSIX

    # ------------------------------------------------------------------ #
    # Warm boot
    # ------------------------------------------------------------------ #
    def warm_boot(self, eager_compile: bool = True) -> List[str]:
        """
        Called once at process startup. Reads manifest.json (if present) and
        cross-checks it against the .py files actually on disk, so:
          - files present on disk but missing from manifest get indexed
          - files referenced in manifest but deleted from disk get dropped
        Then (optionally) eagerly imports + compiles every sub-agent so they're
        immediately invokable with zero cold-start latency on first use.

        Returns the list of sub-agent names that were loaded.
        """
        manifest = self._load_manifest()
        on_disk = {p.stem: p for p in self.sub_agents_dir.glob("*.py")
                   if not p.stem.startswith("_")}

        loaded: List[str] = []

        for stem, path in on_disk.items():
            content_hash = self._hash_file(path)
            cached = manifest.get(stem)

            if cached and cached.get("content_hash") == content_hash:
                # Trust cached metadata, no need to re-import just to read metadata.
                meta = SubAgentMetadata.from_dict(cached)
            else:
                # New or changed file since last run -> (re)import to refresh metadata.
                module = load_module_from_file(path, module_name=f"sub_agents.{stem}")
                meta = self._metadata_from_module(module, path, content_hash)

            self._metadata[meta.name] = meta
            loaded.append(meta.name)

            if eager_compile:
                self._compile_and_cache(meta)

        # Prune manifest entries whose backing file no longer exists.
        for stale_name in set(manifest.keys()) - set(m.name for m in self._metadata.values()):
            self._metadata.pop(stale_name, None)

        self._save_manifest()
        return loaded

    # ------------------------------------------------------------------ #
    # Dynamic registration (called after create_sub_agent_tool writes a file)
    # ------------------------------------------------------------------ #
    def register_from_file(self, path: Path) -> SubAgentMetadata:
        """
        Import a freshly-written sub-agent .py file, compile it, and register
        it into the *live* in-memory registry so the running supervisor graph
        can route to it on the very next turn -- no restart required.
        """
        content_hash = self._hash_file(path)
        module = load_module_from_file(path, module_name=f"sub_agents.{path.stem}")
        meta = self._metadata_from_module(module, path, content_hash)

        self._metadata[meta.name] = meta
        self._compile_and_cache(meta)
        self._save_manifest()
        return meta

    def _compile_and_cache(self, meta: SubAgentMetadata) -> None:
        module = load_module_from_file(
            Path(meta.file_path), module_name=meta.module_name, force_reload=True
        )
        self._compiled[meta.name] = instantiate_agent(module)

    @staticmethod
    def _metadata_from_module(module, path: Path, content_hash: str) -> SubAgentMetadata:
        for required in ("AGENT_NAME", "AGENT_DESCRIPTION", "AGENT_CAPABILITIES", "build_agent"):
            if not hasattr(module, required):
                raise ValueError(
                    f"Sub-agent module at {path} is missing required attribute '{required}'. "
                    f"Every sub-agent must define AGENT_NAME, AGENT_DESCRIPTION, "
                    f"AGENT_CAPABILITIES, and build_agent()."
                )
        return SubAgentMetadata(
            name=module.AGENT_NAME,
            description=module.AGENT_DESCRIPTION,
            capabilities=list(module.AGENT_CAPABILITIES),
            file_path=str(path),
            content_hash=content_hash,
            created_at=time.time(),
            module_name=f"sub_agents.{path.stem}",
        )

    @staticmethod
    def _hash_file(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    # ------------------------------------------------------------------ #
    # Accessors used by the router / execution nodes
    # ------------------------------------------------------------------ #
    def get_compiled(self, name: str) -> Optional[Any]:
        if name not in self._compiled and name in self._metadata:
            # Lazy compile in case eager_compile was False at warm boot.
            self._compile_and_cache(self._metadata[name])
        return self._compiled.get(name)

    def exists(self, name: str) -> bool:
        return name in self._metadata

    def all_metadata(self) -> List[SubAgentMetadata]:
        return list(self._metadata.values())

    def describe_for_router(self) -> str:
        """
        Formats every registered sub-agent's FULL description + capability list
        (never just its name) for injection into the router's system prompt.
        This is what forces the router to evaluate on substance, not naming.
        """
        if not self._metadata:
            return "(no sub-agents currently registered)"

        blocks = []
        for meta in self._metadata.values():
            caps = "\n".join(f"    - {c}" for c in meta.capabilities)
            blocks.append(
                f"* name: {meta.name}\n"
                f"  description: {meta.description}\n"
                f"  capabilities:\n{caps}"
            )
        return "\n\n".join(blocks)
