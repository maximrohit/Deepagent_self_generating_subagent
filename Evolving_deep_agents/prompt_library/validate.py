"""Draft-validation stage prompts."""

VALIDATE_PROMPT = """You validate a DRAFT answer against the ORIGINAL user query.

Be strict about completeness and fidelity:
- Weekly plan requests must cover all 7 days (name each day). If only 3 workout
  days appear with no rest/recovery/active-recovery days for the other 4, FAIL.
- Multi-year / progressive goals must mention progression across time, not only
  one static week, when the user asked for a multi-year approach.
- Equipment constraints (dumbbells, bands, home-only) must be respected.
- Geography/domain constraints must be respected (India ≠ LSE/US equities).
- If the draft invents unsupported claims or wrong domain, FAIL.
- PASS only if the draft substantially satisfies every explicit ask in the query.

Return gaps as actionable TODO seeds for the next iteration.
"""
