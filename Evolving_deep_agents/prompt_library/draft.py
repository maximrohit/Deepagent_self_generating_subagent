"""Draft-synthesis stage prompts."""

DRAFT_SYNTH_PROMPT = """Combine the ordered TODO results into a single DRAFT answer
for the user. Do not drop required parts of the original request. If a TODO said
"I don't know", preserve that uncertainty rather than inventing content.
"""
