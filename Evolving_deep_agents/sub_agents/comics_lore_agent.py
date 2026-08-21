"""
Comics lore sub-agent: MACRO domain specialist for comic-book characters,
storylines, and versus/prep-time style analysis (Raj Comics, Marvel, DC, etc.).
"""
from deepagents import create_deep_agent

from utility.guardrails import HONESTY_AND_GEOGRAPHY_GUIDANCE
from utility.llm import Model
from utility.tavily_tools import (
    TIME_RANGE_GUIDANCE,
    general_search_tool,
    merge_tools,
    news_search_tool,
)

AGENT_NAME = "comics_lore_agent"

AGENT_DESCRIPTION = (
    "MACRO comics / superhero fiction specialist. Answers lore, character, "
    "storyline, and hypothetical matchup questions across comic universes "
    "(e.g. Batman, Superman, Super Commando Dhruv, Marvel, DC, Raj Comics). "
    "Uses general/news web search with time_range for recent coverage. "
    "Does NOT handle finance, equities, or workout planning."
)

AGENT_CAPABILITIES = [
    "Analyzes comic-book characters' abilities, feats, and weaknesses",
    "Compares peers across universes (e.g. Batman vs Super Commando Dhruv)",
    "Reasons about prep-time vs no-prep hypothetical confrontations",
    "Summarizes relevant storylines and continuity with cited search snippets",
    "Answers open-ended comics lore questions via general/news search",
]

_SYSTEM_PROMPT = (
    "You are comics_lore_agent, a MACRO domain specialist for comic-book lore "
    "and superhero fiction. Cover peer characters in the same domain — never "
    "refuse a Superman question because a prior ask was about Batman.\n"
    "Use general_web_search / news_web_search with an appropriate time_range. "
    "Ground claims in retrieved snippets; if evidence is thin, say you don't "
    "know rather than inventing feats.\n"
    "For who-would-win questions, explicitly address both with-prep and "
    "without-prep scenarios when the user asks for both.\n\n"
    + TIME_RANGE_GUIDANCE
    + "\n"
    + HONESTY_AND_GEOGRAPHY_GUIDANCE
)


def build_agent(model_name: str = "local", extra_tools=None):
    tools = merge_tools([general_search_tool, news_search_tool], extra_tools)
    return create_deep_agent(
        model=Model,
        tools=tools,
        system_prompt=_SYSTEM_PROMPT,
        name=AGENT_NAME,
    )
