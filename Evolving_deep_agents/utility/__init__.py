"""Shared utilities for the self-evolving multi-agent system."""

from utility.llm import Model
from utility.domains import (
    DOMAIN_IDS,
    classify_task_domain,
    domain_catalog_for_prompt,
    proposed_agent_for_domain,
)
from utility.guardrails import HONESTY_AND_GEOGRAPHY_GUIDANCE, MACRO_DOMAIN_GUIDANCE
from utility.tavily_tools import (
    ALL_SEARCH_TOOLS,
    DEFAULT_TIME_RANGE,
    PRIMARY_SEARCH_TOOLS,
    TIME_RANGE_GUIDANCE,
    TIME_RANGE_VALUES,
    finance_search_tool,
    general_search_tool,
    news_search_tool,
    select_relevant_search_tools,
    web_search_tool,
)

__all__ = [
    "Model",
    "DOMAIN_IDS",
    "classify_task_domain",
    "domain_catalog_for_prompt",
    "proposed_agent_for_domain",
    "ALL_SEARCH_TOOLS",
    "PRIMARY_SEARCH_TOOLS",
    "TIME_RANGE_GUIDANCE",
    "TIME_RANGE_VALUES",
    "DEFAULT_TIME_RANGE",
    "HONESTY_AND_GEOGRAPHY_GUIDANCE",
    "MACRO_DOMAIN_GUIDANCE",
    "web_search_tool",
    "finance_search_tool",
    "general_search_tool",
    "news_search_tool",
    "select_relevant_search_tools",
]
