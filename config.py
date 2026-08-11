"""
Central configuration for the self-evolving multi-agent system.
"""
import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent

# Load secrets (e.g. TAVILY_API_KEY) from project .env if present.
load_dotenv(BASE_DIR / ".env", override=False)

# All persisted sub-agents (LLM-generated AND hand-written) live here as .py files.
# This directory is what gets scanned on "warm boot".
SUB_AGENTS_DIR = BASE_DIR / "sub_agents"
SUB_AGENTS_DIR.mkdir(exist_ok=True, parents=True)

# Manifest = lightweight index of metadata (name/description/capabilities/file/hash).
# Kept separate from the .py files themselves so we can validate/inspect without
# importing (and therefore executing) code we don't need yet.
MANIFEST_PATH = SUB_AGENTS_DIR / "manifest.json"

# All LLM calls use the local Ollama ChatOllama instance in utility/llm.py.
# These env vars are retained only as unused labels for logging/docs compatibility.
ROUTER_MODEL = os.environ.get("DEEP_AGENT_ROUTER_MODEL", "llama3.2:3b-instruct-q4_K_M")
CODEGEN_MODEL = os.environ.get("DEEP_AGENT_CODEGEN_MODEL", "llama3.2:3b-instruct-q4_K_M")
SUB_AGENT_DEFAULT_MODEL = os.environ.get("DEEP_AGENT_SUBAGENT_MODEL", "llama3.2:3b-instruct-q4_K_M")
