"""
Primary deep agent: general-purpose researcher with finance/general/news
Tavily search tools. Persisted under sub_agents/ so warm_boot loads it.
"""
from deepagents import create_deep_agent

from utility.llm import Model
from utility.tavily_tools import PRIMARY_SEARCH_TOOLS, TIME_RANGE_GUIDANCE, merge_tools
from utility.guardrails import HONESTY_AND_GEOGRAPHY_GUIDANCE

AGENT_NAME = "primary_deep_agent"

AGENT_DESCRIPTION = (
    "Primary general-purpose deep agent for research, synthesis, and "
    "open-ended questions that benefit from live web data. Uses topic-scoped "
    "Tavily search tools (finance, general, news). The SAME agent handles "
    "different recency needs (e.g. 1-week vs 1-month vs 1-year questions) by "
    "passing time_range=day|week|month|year on each tool call. Prefer "
    "specialized sub-agents when a narrower skill clearly fits better."
)

AGENT_CAPABILITIES = [
    "Answers open-ended research questions with live web search",
    "Runs finance-topic web searches for markets and economic data",
    "Runs news-topic searches for current events and headlines",
    "Runs general web searches for factual up-to-date snippets",
    "Answers 1-week vs 1-month (and day/year) queries via time_range tool arg",
    "Synthesizes multi-source findings into a concise answer",
]

_SYSTEM_PROMPT = (
    "You are the primary deep agent. Prefer the topic-matched search tool "
    "(finance_web_search, news_web_search, or general_web_search) before "
    "answering questions that need current information.\n"
    "Horizon flexibility: if the user asks about 'this week' use "
    "time_range='week'; if they ask about 'this month' / '1 month' use "
    "'month'; if they ask both, search twice with different time_range values "
    "and contrast the findings. Keep answers grounded in retrieved snippets "
    "and cite source URLs when available.\n\n"
    + TIME_RANGE_GUIDANCE
    + "\n"
    + HONESTY_AND_GEOGRAPHY_GUIDANCE
)


def build_agent(model_name: str = "local", extra_tools=None):
    tools = merge_tools(PRIMARY_SEARCH_TOOLS, extra_tools)
    return create_deep_agent(
        model=Model,
        tools=tools,
        system_prompt=_SYSTEM_PROMPT,
        name=AGENT_NAME,
    )
