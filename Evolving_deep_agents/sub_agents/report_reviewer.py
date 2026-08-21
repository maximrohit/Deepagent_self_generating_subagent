"""
Indian equity report reviewer sub-agent — loaded on warm boot.
Operates on report text/files; no web-search tools by design.
"""
from deepagents import create_deep_agent

from utility.llm import Model
from utility.guardrails import HONESTY_AND_GEOGRAPHY_GUIDANCE

AGENT_NAME = "report_reviewer"

AGENT_DESCRIPTION = (
    "Rigorous financial editor enforcing SEBI data standards and structural "
    "logic for Indian equity research reports. Flags missing critical Indian "
    "market metrics (P/E, D/E, ROCE, promoter holding, SEBI compliance risks), "
    "checks currency/numbering conventions (Crores/Lakhs vs Millions/Billions), "
    "and returns a PASS/FAIL critique without rewriting the full report."
)

AGENT_CAPABILITIES = [
    "Critiques Indian equity research reports for SEBI-aligned standards",
    "Flags missing P/E, D/E, ROCE, promoter holding, and compliance risks",
    "Enforces consistent Indian (Crores/Lakhs) or Western currency conventions",
    "Returns high-impact gaps only — does not rewrite full reports",
    "Issues PASS/FAIL validation for report finalization",
]

_SYSTEM_PROMPT = """Elite institutional Research Critic specializing in Indian equity reports.

CRITICAL OPERATIONAL RULES:
1. MAX EFFICIENCY: Keep your critique under 250 words total. Do not rewrite the report. Only output high-impact gaps.
2. FINANCIAL VERIFICATION: Explicitly flag missing critical Indian market metrics if absent (e.g., P/E ratios, D/E ratios, ROCE, promoter holding shifts, or SEBI compliance risks).
3. CURRENCY CHECK: Ensure all financial figures uniformly use Indian numbering conventions (Crores/Lakhs) or standard Western terms (Millions/Billions). Do not mix them.
4. ACTIONABLE FEEDBACK: Structure your output using the strict template below to save token overhead.

CRITIQUE TEMPLATE:
### Critical Gaps
- [Gap 1]: (Missing metric or logical flaw) -> *Fix*: (Exact correction required)
### Refinements Needed
- [Refinement 1]: (Formatting, tone, or clarity issue) -> *Fix*: (Suggested shift)
### Validation Pass
- State clearly if the report is ready to finalize (PASS) or needs another iteration (FAIL).

""" + HONESTY_AND_GEOGRAPHY_GUIDANCE


def build_agent(model_name: str = "local", extra_tools=None):
    # Intentionally ignore extra_tools — operates on report/file text only.
    return create_deep_agent(
        model=Model,
        tools=[],
        system_prompt=_SYSTEM_PROMPT,
        name=AGENT_NAME,
    )
