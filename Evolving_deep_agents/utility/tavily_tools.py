"""
Domain-specific Tavily web-search tools with aggressive token capping.

Topic-scoped tools call the Tavily client directly (community TavilySearchResults
does not expose `topic` / reliable `time_range`). Agents MUST pass `time_range`
per call according to how recent the data needs to be:
  "day" | "week" | "month" | "year"
"""
from __future__ import annotations

import json
import os
import re
from typing import List, Literal, Optional, Sequence

from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel, Field
from tavily import TavilyClient

Topic = Literal["general", "news", "finance"]
TimeRange = Literal["day", "week", "month", "year"]

TIME_RANGE_VALUES: tuple[TimeRange, ...] = ("day", "week", "month", "year")
DEFAULT_TIME_RANGE: TimeRange = "week"

# Shared guidance injected into agent system prompts / codegen templates.
TIME_RANGE_GUIDANCE = """\
TOOL ARG: time_range (REQUIRED on every finance_web_search / general_web_search / \
news_web_search / web_search call). Allowed values ONLY:
  "day" | "week" | "month" | "year"

SAME AGENT, MULTIPLE HORIZONS:
- You are NOT locked to one horizon. The SAME agent must answer a "1 week" \
question AND a "1 month" question by changing the time_range tool argument \
(and the reasoning), not by refusing or inventing a different agent.
- Read the user's wording and map it:
    "today" / "intraday" / "last 24h" / dated same-day setup  → time_range="day"
    "this week" / "1 week" / "weekly" / "next few days"       → time_range="week"
    "this month" / "1 month" / "~30 days" / "monthly"         → time_range="month"
    "this year" / "12 months" / "annual" / "structural"       → time_range="year"
- If the user asks for BOTH horizons in one task, call the search tool twice \
(or more) with different time_range values and compare the results.
- Never silently default every call to "week". Never ignore an explicit \
horizon in the user question.
- State which time_range you used when you cite live data.
"""

# Shared search settings (token/VRAM-conscious defaults).
_MAX_RESULTS = 20
_SEARCH_DEPTH = "advanced"
_INCLUDE_ANSWER = True


class TopicSearchArgs(BaseModel):
    query: str = Field(description="Natural-language search query.")
    time_range: TimeRange = Field(
        default=DEFAULT_TIME_RANGE,
        description=(
            'Recency window for THIS call: "day", "week", "month", or "year". '
            'Map the user horizon (e.g. "1 week"→"week", "1 month"→"month"). '
            "The same agent should use different time_range values for different "
            "horizons; call the tool multiple times if several windows are needed."
        ),
    )


def _client() -> TavilyClient:
    api_key = os.environ.get("TAVILY_API_KEY")
    if not api_key:
        raise RuntimeError(
            "TAVILY_API_KEY is not set. Add it to the project .env file."
        )
    return TavilyClient(api_key=api_key)


def _normalize_time_range(time_range: Optional[str]) -> TimeRange:
    value = (time_range or DEFAULT_TIME_RANGE).strip().lower()
    if value not in TIME_RANGE_VALUES:
        return DEFAULT_TIME_RANGE
    return value  # type: ignore[return-value]


def _run_topic_search(
    query: str,
    topic: Topic,
    time_range: Optional[str] = DEFAULT_TIME_RANGE,
) -> str:
    """Execute a capped Tavily search for a fixed topic and return compact JSON."""
    tr = _normalize_time_range(time_range)
    result = _client().search(
        query=query,
        topic=topic,
        max_results=_MAX_RESULTS,
        search_depth=_SEARCH_DEPTH,
        time_range=tr,
        include_answer=_INCLUDE_ANSWER,
    )
    # Keep payloads small for local-LLM context windows.
    compact = {
        "query": query,
        "topic": topic,
        "time_range": tr,
        "answer": result.get("answer"),
        "results": [
            {
                "title": r.get("title"),
                "url": r.get("url"),
                "content": r.get("content"),
                "score": r.get("score"),
                "published_date": r.get("published_date"),
            }
            for r in (result.get("results") or [])
        ],
    }
    return json.dumps(compact, ensure_ascii=False)


def _make_topic_tool(topic: Topic, name: str, description: str) -> StructuredTool:
    def _search(
        query: str,
        time_range: TimeRange = DEFAULT_TIME_RANGE,
    ) -> str:
        return _run_topic_search(query, topic, time_range)

    _search.__doc__ = (
        f"{description}\n\n"
        "Args:\n"
        "    query: Natural-language search query.\n"
        '    time_range: Recency window — one of "day", "week", "month", "year". '
        "Choose based on the task horizon (intraday→day, near-term→week, "
        "monthly trends→month, structural/annual→year)."
    )

    return StructuredTool.from_function(
        func=_search,
        name=name,
        description=(
            f"{description} "
            'Always set time_range to "day", "week", "month", or "year" based on '
            "how recent the needed information is."
        ),
        args_schema=TopicSearchArgs,
    )


# --------------------------------------------------------------------------- #
# Topic wrappers: finance / general / news (+ general web_search alias)
# --------------------------------------------------------------------------- #
finance_search_tool = _make_topic_tool(
    topic="finance",
    name="finance_web_search",
    description=(
        "Search finance-focused sources for markets, equities, macro data, "
        "company filings, and investment-related up-to-date information."
    ),
)

general_search_tool = _make_topic_tool(
    topic="general",
    name="general_web_search",
    description=(
        "General-purpose web search for clean, up-to-date factual snippets "
        "that are not specifically finance or breaking-news queries."
    ),
)

news_search_tool = _make_topic_tool(
    topic="news",
    name="news_web_search",
    description=(
        "Search recent news coverage for politics, sports, and major current "
        "events from mainstream media sources."
    ),
)

# Backward-compatible alias used by older prompts / generated agents.
web_search_tool = _make_topic_tool(
    topic="general",
    name="web_search",
    description=(
        "Useful for finding single clean snippets of up-to-date data "
        "(general topic channel)."
    ),
)

ALL_SEARCH_TOOLS: List[BaseTool] = [
    web_search_tool,
    finance_search_tool,
    general_search_tool,
    news_search_tool,
]

PRIMARY_SEARCH_TOOLS: List[BaseTool] = [
    finance_search_tool,
    general_search_tool,
    news_search_tool,
]

_FINANCE_RE = re.compile(
    r"\b(financ|market|stock|equity|equities|trading|invest|portfolio|sec\b|"
    r"10-?k|earnings|revenue|gdp|inflation|forex|crypto|bitcoin|bond|nasdaq|"
    r"s&p|dow\b|valuation|ticker|ipo)\w*",
    re.I,
)
_NEWS_RE = re.compile(
    r"\b(news|headline|breaking|politics|election|sports?|tournament|"
    r"current events?|today'?s|this week|latest report)\w*",
    re.I,
)
_GENERAL_RE = re.compile(
    r"\b(search|web|lookup|research|up-?to-?date|latest|internet|online|"
    r"find info|browse|factual|wikipedia|documentation)\w*",
    re.I,
)


def select_relevant_search_tools(
    text: str,
    *,
    always_include_general: bool = False,
) -> List[BaseTool]:
    """
    Pick finance/general/news wrappers that match an agent description or task.

    Returns an empty list when nothing indicates web research is needed
    (e.g. pure local computation agents).
    """
    blob = text or ""
    selected: List[BaseTool] = []

    if _FINANCE_RE.search(blob):
        selected.append(finance_search_tool)
    if _NEWS_RE.search(blob):
        selected.append(news_search_tool)
    if always_include_general or _GENERAL_RE.search(blob) or selected:
        # If a topical tool matched, also offer general as a fallback channel.
        if general_search_tool not in selected:
            selected.append(general_search_tool)

    # De-dupe while preserving order.
    seen = set()
    unique: List[BaseTool] = []
    for tool in selected:
        if tool.name in seen:
            continue
        seen.add(tool.name)
        unique.append(tool)
    return unique


def merge_tools(
    domain_tools: Optional[Sequence[BaseTool]] = None,
    extra_tools: Optional[Sequence[BaseTool]] = None,
) -> List[BaseTool]:
    """
    Merge domain + injected tools without duplicate tool names.

    Silently drops language-model objects (e.g. ChatOllama `Model`) that
    codegen sometimes mistakenly places in the tools list — those belong in
    model=, never in tools=.
    """
    from langchain_core.language_models.chat_models import BaseChatModel
    from langchain_core.tools import BaseTool as _BaseTool

    merged: List[BaseTool] = []
    seen = set()
    for tool in list(domain_tools or []) + list(extra_tools or []):
        if tool is None:
            continue
        # Guard against codegen putting the LLM into the tools list.
        if isinstance(tool, BaseChatModel):
            continue
        if not (isinstance(tool, _BaseTool) or callable(tool)):
            continue
        name = getattr(tool, "name", None) or getattr(tool, "__name__", None) or id(tool)
        if name in seen:
            continue
        seen.add(name)
        merged.append(tool)
    return merged
