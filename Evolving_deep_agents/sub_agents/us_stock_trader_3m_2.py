"""
Duplicate/legacy US stock trader module (same AGENT_NAME as us_stock_trader_3m.py).
Kept for warm-boot compatibility; prefer us_stock_trader_3m.py going forward.
"""
from deepagents import create_deep_agent

from utility.llm import Model
from utility.tavily_tools import TIME_RANGE_GUIDANCE, finance_search_tool, merge_tools
from utility.guardrails import HONESTY_AND_GEOGRAPHY_GUIDANCE

AGENT_NAME = "us_stock_trader_3m"

AGENT_DESCRIPTION = (
    "US equity recommendations with flexible horizons. The SAME agent answers "
    "1-week vs 1-month questions by passing time_range on finance_web_search."
)

AGENT_CAPABILITIES = [
    "Analyzes market trends and news",
    "Answers 1-week vs 1-month US trade questions via time_range",
    "Provides recommendations based on technical analysis",
    "Uses finance_web_search with day/week/month/year time_range",
]


def build_agent(model_name: str = "local", extra_tools=None):
    tools = merge_tools([finance_search_tool], extra_tools)
    return create_deep_agent(
        model=Model,
        tools=tools,
        system_prompt=(
            "Provide a US stock recommendation for the USER'S requested horizon. "
            "Always call finance_web_search with an explicit time_range tool arg: "
            "'week' for 1-week questions, 'month' for 1-month / ~3-month questions, "
            "'day' for same-day. Same agent must handle both week and month.\n\n"
            + TIME_RANGE_GUIDANCE
            + "\n"
            + HONESTY_AND_GEOGRAPHY_GUIDANCE
        ),
        name=AGENT_NAME,
    )
