"""
Indian stock trader sub-agent — flexible near-term horizons via time_range.
"""
from deepagents import create_deep_agent

from utility.llm import Model
from utility.tavily_tools import TIME_RANGE_GUIDANCE, finance_search_tool, merge_tools
from utility.guardrails import HONESTY_AND_GEOGRAPHY_GUIDANCE

AGENT_NAME = "indian_stock_trader_3m"

AGENT_DESCRIPTION = (
    "Provides personalized Indian equity trade recommendations with emphasis on "
    "high upside and capital preservation for near-term setups (days to a few "
    "months). The SAME agent handles a 1-week question vs a 1-month (or "
    "same-day / ~3-month) question by passing time_range=day|week|month|year on "
    "finance_web_search. Does NOT replace market_analyst (broad SEBI research), "
    "report_reviewer (critique only), or investment_strategist (CIO multi-horizon "
    "verdicts on finalized reports)."
)

AGENT_CAPABILITIES = [
    "Recommends Indian NSE/BSE stocks for near-term trade setups",
    "Answers 1-week vs 1-month (and day/year) trade questions via time_range",
    "Uses finance_web_search with day/week/month/year for live context",
    "Produces risk-reward analysis including sector, RBI, and SEBI factors",
    "Outputs a concise buy/sell/hold-style recommendation with confidence",
    "Avoids full institutional report writing (owned by market_analyst)",
]

_SYSTEM_PROMPT = (
    "You are an Indian equity trader for near-term setups. Use "
    "finance_web_search BEFORE recommending, and ALWAYS pass time_range as a "
    "tool argument based on the user's horizon:\n"
    "- user says 1 week / this week → time_range='week'\n"
    "- user says 1 month / this month → time_range='month'\n"
    "- same-day / dated trade → time_range='day'\n"
    "- ~3-month swing when asked → time_range='month' (optionally add 'year' "
    "for structural backdrop)\n"
    "You must be able to answer BOTH a 1-week and a 1-month question; only the "
    "time_range (and thesis) change. If both are requested, search twice and "
    "compare. Keep answers concise: thesis, key risks, BUY/HOLD/SELL with "
    "confidence, and state which time_range you used. Prefer Indian numbering "
    "(Crores/Lakhs).\n\n"
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
