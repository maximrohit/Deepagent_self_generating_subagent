"""
US stock trader sub-agent — flexible horizons via time_range tool arg.
"""
from deepagents import create_deep_agent

from utility.llm import Model
from utility.tavily_tools import TIME_RANGE_GUIDANCE, finance_search_tool, merge_tools
from utility.guardrails import HONESTY_AND_GEOGRAPHY_GUIDANCE

AGENT_NAME = "us_stock_trader_3m"

AGENT_DESCRIPTION = (
    "Recommends US equities for near-term / swing setups with high conviction. "
    "The SAME agent answers a 1-week question vs a 1-month (or day/~3-month) "
    "question by passing time_range=day|week|month|year on finance_web_search. "
    "Prefer this agent for short-horizon US stock picks; use market_analyst for "
    "Indian NSE/BSE analysis instead."
)

AGENT_CAPABILITIES = [
    "Analyzes US stock prices, trends, and sector risk",
    "Answers 1-week vs 1-month (and day/year) US trade questions via time_range",
    "Recommends high-conviction US equities for the user-requested horizon",
    "Uses finance_web_search with day/week/month/year for live market data",
]

_SYSTEM_PROMPT = (
    "You are a US equity trader. Use finance_web_search BEFORE recommending, "
    "and ALWAYS pass time_range as a tool argument from the user's horizon:\n"
    "- '1 week' / 'this week' → time_range='week'\n"
    "- '1 month' / 'this month' → time_range='month'\n"
    "- intraday / today → time_range='day'\n"
    "- ~3-month swing when asked → time_range='month' (+ optional 'year' backdrop)\n"
    "Do not lock yourself to a single horizon: the same agent must handle week "
    "AND month questions by changing time_range only. If both are asked, call "
    "search twice and contrast. Keep answers concise: thesis, risks, "
    "BUY/HOLD/SELL with confidence; mention the time_range used.\n\n"
    + TIME_RANGE_GUIDANCE
    + "\n"
    + HONESTY_AND_GEOGRAPHY_GUIDANCE
)


def build_agent(model_name: str = "local", extra_tools=None):
    tools = merge_tools([finance_search_tool], extra_tools)
    return create_deep_agent(
        model=Model,
        tools=tools,
        system_prompt=_SYSTEM_PROMPT,
        name=AGENT_NAME,
    )
