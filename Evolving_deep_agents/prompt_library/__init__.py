"""
Central prompt library for the supervisor graph.

Organized by stage type:
  analyze         — topic / entity / domain briefing
  plan            — TODO planning
  create_research — holistic system-prompt research for new agents
  draft           — draft synthesis
  validate        — draft vs original query
  finalize        — final user-facing answer
"""

from prompt_library.analyze import ANALYZE_TOPIC_PROMPT
from prompt_library.create_research import SYNTHESIZE_AGENT_PROMPT_RESEARCH
from prompt_library.draft import DRAFT_SYNTH_PROMPT
from prompt_library.finalize import FINALIZE_PROMPT
from prompt_library.plan import PLAN_TODOS_PROMPT
from prompt_library.validate import VALIDATE_PROMPT

__all__ = [
    "ANALYZE_TOPIC_PROMPT",
    "PLAN_TODOS_PROMPT",
    "DRAFT_SYNTH_PROMPT",
    "VALIDATE_PROMPT",
    "FINALIZE_PROMPT",
    "SYNTHESIZE_AGENT_PROMPT_RESEARCH",
]
