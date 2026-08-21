"""Final-answer stage prompts."""

FINALIZE_PROMPT = """Produce the FINAL user-facing answer from the best validated
draft. Keep it complete vs the original query. If still incomplete after max
iterations, clearly list remaining gaps and what is unknown — do not invent.
"""
