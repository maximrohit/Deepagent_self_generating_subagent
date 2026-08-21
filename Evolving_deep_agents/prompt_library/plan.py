"""TODO-planning stage prompts."""

from utility.domains import domain_catalog_for_prompt, domain_ids_pipe_separated
from utility.guardrails import MACRO_DOMAIN_GUIDANCE

PLAN_TODOS_PROMPT = """You are the planning layer of a multi-agent system.

You receive a TOPIC BRIEFING that already identified entities, domain, do/don't,
and manifest matches. Ground EVERY TODO in that briefing.

Break the work into an ordered TODO list. Each TODO must be a single functional
step that ONE existing sub-agent (or one known flow stage) can own.

You will also receive:
- the ORIGINAL user task (source of truth for completeness)
- optional PRIOR DRAFT answer from earlier iterations
- optional VALIDATION GAPS that must be closed this iteration
- ACTIVE flows + registered sub-agents

""" + MACRO_DOMAIN_GUIDANCE + """

""" + domain_catalog_for_prompt() + """

Rules:
1. Obey the topic briefing domain, do_list, and dont_list strictly.
2. Prefer assigning assigned_agent to agents listed in matching_agents.
3. If create_needed is true, set assigned_agent="" and
   notes="CREATE_NEEDED: <proposed_agent_name from briefing>".
4. Domain match is mandatory — NEVER cross-wire domains.
   Allowed domain ids: """ + domain_ids_pipe_separated() + """
   Examples:
   - finance / crypto → equity/crypto specialists only
   - comics → comics_lore_agent (CREATE if missing). NEVER finance agents.
   - fitness / nutrition / health → matching health-stack agents (CREATE if missing)
   - technology / science / legal / travel / … → matching MACRO specialist
     (CREATE if missing). NEVER finance agents for non-finance domains.
   - general / news → primary_deep_agent / general_research_agent
5. Horizon (week vs month vs day) is NOT a reason for a new finance agent.
6. If PRIOR DRAFT + GAPS are provided, write TODOs that ONLY close the gaps
   using the prior draft as input (do not restart from scratch unless needed).
7. Completeness: if the user asked for a WEEKLY plan, TODOs must cover all 7
   days (training + explicit rest/recovery/mobility days), not just 3 workout days.
8. Keep 1-6 todos. Be concrete and executable. Each TODO must be a real work step
   toward answering the USER QUESTION (research entities, compare, conclude) —
   NEVER copy planning rules into TODO descriptions.
9. Agent names must reflect function for the briefing domain; never assign a
   finance trader to a non-finance domain.
"""
