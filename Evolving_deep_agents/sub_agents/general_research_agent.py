from deepagents import create_deep_agent
from utility.llm import Model
from utility.tavily_tools import TIME_RANGE_GUIDANCE, finance_search_tool, merge_tools
from utility.guardrails import HONESTY_AND_GEOGRAPHY_GUIDANCE

AGENT_NAME = "general_research_agent"
AGENT_DESCRIPTION = "A general research and synthesis agent for open-ended questions, providing a foundation for more specialized sub-agents to build upon."
AGENT_CAPABILITIES = [
    "Answers open-ended research questions with live web search",
    "Runs finance-topic web searches for markets and economic data",
    "Runs news-topic searches for current events and headlines",
    "Runs general web searches for factual up-to-date snippets",
    "Synthesizes multi-source findings into a concise answer",
]

def build_agent(model_name: str = "local", extra_tools=None):
    tools = merge_tools([], extra_tools)
    return create_deep_agent(
        model=Model,
        tools=tools,
        system_prompt="..." + "\n\n" + TIME_RANGE_GUIDANCE + "\n\n" + HONESTY_AND_GEOGRAPHY_GUIDANCE,
        name=AGENT_NAME,
    )