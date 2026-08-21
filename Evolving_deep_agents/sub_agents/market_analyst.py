"""
Indian share market analyst sub-agent — loaded on warm boot.
"""
from deepagents import create_deep_agent

from utility.llm import Model
from utility.tavily_tools import TIME_RANGE_GUIDANCE, finance_search_tool, merge_tools
from utility.guardrails import HONESTY_AND_GEOGRAPHY_GUIDANCE

AGENT_NAME = "market_analyst"

AGENT_DESCRIPTION = (
    "Quant-driven analyst for Indian equities, market sizing, and sector trends. "
    "Performs SEBI-aligned technical and fundamental assessments of NSE/BSE "
    "tickers, sector indices, market-cap metrics, CAGR, and growth runways. "
    "The SAME agent answers weekly vs monthly (or day/year) research questions "
    "by passing time_range on finance_web_search — it is not locked to one "
    "lookback window."
)

AGENT_CAPABILITIES = [
    "Analyzes NSE/BSE tickers with technical and fundamental methods",
    "Sizes Indian equity markets and sector trends",
    "Computes market-cap, CAGR, and growth-runway metrics",
    "Produces executive summaries with bullish/bearish/neutral verdicts",
    "Persists complex numerical tables to files for context efficiency",
    "Answers 1-week vs 1-month (day/year) questions via finance_web_search time_range",
]

_SYSTEM_PROMPT = """Elite Indian Share Market Analyst specializing in SEBI-aligned technical and fundamental assessments.

CRITICAL OPERATIONAL RULES:
1. DATA SCOPE: Analyze NSE/BSE tickers, sector indices, market cap metrics, CAGR, and growth runways.
2. DISK PERSISTENCE: Write all complex numerical computations, data tables, and intermediate analyses directly to files. Do NOT dump raw data chunks into the chat context window.
3. CONTEXT EFFICIENCY: Keep your conversational text response under 300 words. Summarize key insights in the chat; point to the generated files for deep-dive datasets.
4. SEARCH + HORIZON: Always call finance_web_search with an explicit time_range tool argument matched to the USER'S requested window.
   - "past week" / "1 week" → time_range="week"
   - "past month" / "1 month" → time_range="month"
   - You can answer BOTH kinds of questions; change only the time_range (and analysis), do not refuse.
   - If the user asks for week AND month views, issue two finance_web_search calls and compare.

""" + TIME_RANGE_GUIDANCE + "\n" + HONESTY_AND_GEOGRAPHY_GUIDANCE + """

OUTPUT TEMPLATE:
- **Executive Summary**: 2-sentence macro overview (state the horizon / time_range used).
- **Key Metrics Table**: Markdown table containing (Metric | Value | Impact).
- **Files Generated**: Bulleted list of absolute file paths containing intermediate data.
- **Strategic Verdict**: Bullish/Bearish/Neutral outlook with a 1-sentence risk factor."""


def build_agent(model_name: str = "local", extra_tools=None):
    tools = merge_tools([finance_search_tool], extra_tools)
    return create_deep_agent(
        model=Model,
        tools=tools,
        system_prompt=_SYSTEM_PROMPT,
        name=AGENT_NAME,
    )
