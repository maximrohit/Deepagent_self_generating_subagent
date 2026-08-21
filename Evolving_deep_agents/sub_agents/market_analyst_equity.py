from deepagents import create_deep_agent
from utility.llm import Model
from utility.tavily_tools import TIME_RANGE_GUIDANCE, finance_search_tool, merge_tools
from utility.llm import Model as LLM
from deepagents import create_deep_agent

AGENT_NAME = "market_analyst_equity"
AGENT_DESCRIPTION = "Provides SEBI-aligned technical and fundamental analysis for Indian equity tickers, including market sizing, CAGR, and growth runways."
AGENT_CAPABILITIES = [
    "Analyzes Indian equity tickers with technical and fundamental methods",
    "Sizes Indian equity markets and sector trends",
    "Computes market-cap, CAGR, and growth-runway metrics",
    "Produces executive summaries with bullish/bearish/neutral verdicts",
    "Persists complex numerical tables to files for context efficiency",
]

def build_agent(model_name: str = "local", extra_tools=None):
    tools = merge_tools([finance_search_tool], extra_tools)
    return create_deep_agent(
        model=LLM,
        tools=tools,
        system_prompt="..." + "\n\n" + TIME_RANGE_GUIDANCE,
        name=AGENT_NAME,
    )