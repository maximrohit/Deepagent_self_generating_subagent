"""
Example hand-authored sub-agent, present on disk BEFORE the system starts, to
demonstrate that warm_boot() picks up historical sub-agents without any
LLM code generation.
"""
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

from utility.tavily_tools import merge_tools

AGENT_NAME = "regex_helper_agent"

AGENT_DESCRIPTION = (
    "Handles tasks involving writing, explaining, debugging, or testing regular "
    "expressions (regex) in any common flavor (Python re, PCRE, JavaScript, POSIX). "
    "Given a natural-language description of a matching/extraction/validation "
    "requirement, produces a working regex pattern, explains each component, and "
    "verifies it against example strings. Does not handle general string "
    "parsing that is better solved without regex (e.g. full HTML/JSON parsing)."
)

AGENT_CAPABILITIES = [
    "Writes regex patterns from natural-language matching requirements",
    "Explains what each part of a given regex pattern does",
    "Debugs why a regex fails to match expected input",
    "Tests a candidate regex against sample positive/negative strings",
    "Converts patterns between regex flavors (Python/PCRE/JS/POSIX)",
]


@tool
def test_regex(pattern: str, test_string: str) -> str:
    """Test whether `pattern` matches `test_string` using Python's re module."""
    import re

    match = re.search(pattern, test_string)
    return f"MATCH: {match.group(0)!r} at {match.span()}" if match else "NO MATCH"


def build_agent(model_name: str = "local", extra_tools=None):
    from utility.llm import Model

    return create_react_agent(
        Model,
        tools=merge_tools([test_regex], extra_tools),
        prompt=(
            "You are a regex specialist. Write correct, minimal regex patterns, "
            "explain them clearly, and use the test_regex tool to verify your "
            "pattern against any example strings the user gives before answering."
        ),
    )
