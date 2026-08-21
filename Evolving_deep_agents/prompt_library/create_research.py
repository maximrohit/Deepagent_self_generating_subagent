"""Prompt-research stage for new sub-agent creation.

Researches the MACRO domain and entity *attributes* (not the user's one-shot
question framing) so created agents get a holistic system prompt.
"""

from utility.guardrails import MACRO_DOMAIN_GUIDANCE

SYNTHESIZE_AGENT_PROMPT_RESEARCH = """You synthesize a HOLISTIC system-prompt brief
for a new MACRO domain sub-agent.

You will receive:
- MACRO domain id + proposed agent name
- entities + entity_types from prior topic analysis (already in the flow)
- web research snippets about those entities' attributes / the domain

""" + MACRO_DOMAIN_GUIDANCE + """

CRITICAL RULES:
- Build a durable DOMAIN specialist prompt, NOT an answer to one user fight/query.
- Use entities only as exemplars of attributes/capabilities in the domain.
- Cover PEER instances in the same domain (if entities are Batman + Dhruv,
  the agent must also handle Superman, Spider-Man, etc.).
- Do NOT tailor the prompt to "who wins with/without prep" or any single ask.
- Prefer attributes, powers, roles, constraints, typical evidence sources.
- If snippets are thin, say so in open_questions; still produce a solid MACRO brief.

Return PromptResearchBrief:
1) domain_overview: 2-4 sentences on what this MACRO domain covers
2) entity_attribute_notes: per researched entity — durable attributes/abilities/
   roles (not who-beats-whom conclusions)
3) peer_coverage: how peer instances in the same domain stay in-scope
4) evidence_habits: how the agent should use search / cite / say "I don't know"
5) system_prompt_core: the main system-prompt body the codegen must embed
   (MACRO specialist voice; include peer coverage; no micro lock-in)
6) open_questions: gaps the agent should treat carefully at runtime
"""
