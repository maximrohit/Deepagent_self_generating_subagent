"""
MACRO domain catalog — single source of truth for topic analysis, routing,
agent creation names, and prompt guidance.

Domains are intentionally broader than one character/ticker/product, but far
richer than the old finance|comics|fitness|general quartet.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple  # noqa: F401

# Ordered for classification priority (first match wins). Put specific niches
# before broad ones (e.g. comics before entertainment, crypto before finance).
DOMAIN_SPECS: Tuple[Dict[str, object], ...] = (
    {
        "id": "comics",
        "label": "Comics / superhero fiction",
        "proposed_agent": "comics_lore_agent",
        "markers": (
            "batman", "superman", "spiderman", "spider-man", "wonder woman",
            "marvel", "dc comic", "dc comics", "superhero", "super hero",
            "superheroes", "comics", "comic book", "comic-book",
            "commando dhruv", "super commando", "dhruv", "rajnagar", "raj comics",
            "avenger", "x-men", "manga", "anime",
            "graphic novel", "caped crusader",
        ),
        "agent_keywords": (
            "comic", "comics", "superhero", "lore", "manga", "anime",
        ),
    },
    {
        "id": "fitness",
        "label": "Fitness / strength training",
        "proposed_agent": "home_strength_training_agent",
        "markers": (
            "workout", "exercise", "fitness", "gym", "dumbbell", "resistance band",
            "muscle", "training plan", "yoga", "cardio", "calisthenics",
            "bodybuilding", "crossfit", "hiit", "stretching", "mobility routine",
        ),
        "agent_keywords": (
            "fitness", "workout", "training", "strength", "exercise", "gym",
        ),
    },
    {
        "id": "nutrition",
        "label": "Nutrition / meal planning",
        "proposed_agent": "nutrition_meal_planning_agent",
        "markers": (
            "nutrition", "diet", "meal plan", "calorie", "macros", "protein intake",
            "vegan diet", "keto", "intermittent fasting", "recipe", "cooking",
            "food prep", "grocery list",
        ),
        "agent_keywords": (
            "nutrition", "meal", "diet", "recipe", "cooking", "food",
        ),
    },
    {
        "id": "health",
        "label": "Health / wellness (non-diagnostic)",
        "proposed_agent": "health_wellness_agent",
        "markers": (
            "symptoms", "wellness", "mental health", "sleep hygiene", "stress",
            "meditation", "physiotherapy", "rehab", "injury recovery",
            "blood pressure", "diabetes management",
        ),
        "agent_keywords": (
            "health", "wellness", "medical", "symptom", "rehab",
        ),
    },
    {
        "id": "crypto",
        "label": "Crypto / digital assets",
        "proposed_agent": "crypto_markets_agent",
        "markers": (
            "cryptocurrency", "crypto", "bitcoin", "ethereum", "solana", "defi",
            "nft", "blockchain", "altcoin", "web3", "stablecoin",
        ),
        "agent_keywords": (
            "crypto", "bitcoin", "ethereum", "blockchain", "defi", "nft",
        ),
    },
    {
        "id": "finance",
        "label": "Equities / markets / investing",
        "proposed_agent": "market_research_agent",
        "markers": (
            "stock", "stocks", "equity", "equities", "nse", "bse", "nasdaq", "nyse",
            "nifty", "sensex", "portfolio", "dividend", "ipo", "sebi", "ticker",
            "share price", "intraday", "mutual fund", "options trading", "forex",
            "market cap", "earnings", "buy/sell", "trade setup", "trading setup",
            "investment verdict", "bullish", "bearish", "bond", "etf",
            "interest rate", "rbi", "fed rate", "valuation",
        ),
        "agent_keywords": (
            "stock", "equity", "trader", "market", "finance", "investment",
            "portfolio", "sebi", "nse", "bse", "analyst", "strategist",
        ),
    },
    {
        "id": "technology",
        "label": "Technology / software / engineering",
        "proposed_agent": "technology_research_agent",
        "markers": (
            "software", "programming", "python", "javascript", "typescript",
            "api design", "kubernetes", "docker", "machine learning", "llm",
            "devops", "cloud", "aws", "azure", "gcp", "database", "sql",
            "cybersecurity", "coding interview", "system design", "git",
            "react", "pytorch", "tensorflow",
        ),
        "agent_keywords": (
            "technology", "software", "engineering", "programming", "devops",
            "cloud", "cyber", "ml", "ai research",
        ),
    },
    {
        "id": "science",
        "label": "Science / research explainers",
        "proposed_agent": "science_explainer_agent",
        "markers": (
            "physics", "chemistry", "biology", "astronomy", "quantum",
            "climate science", "scientific paper", "peer review", "genome",
            "neuroscience", "space exploration",
        ),
        "agent_keywords": (
            "science", "physics", "biology", "chemistry", "research explainer",
        ),
    },
    {
        "id": "history",
        "label": "History / civilizations",
        "proposed_agent": "history_research_agent",
        "markers": (
            "history", "historical", "ancient", "medieval", "world war",
            "civilization", "empire", "archaeology", "timeline of",
            "who was", "when did",
        ),
        "agent_keywords": ("history", "historical", "civilization", "archaeology"),
    },
    {
        "id": "politics",
        "label": "Politics / policy / geopolitics",
        "proposed_agent": "politics_policy_agent",
        "markers": (
            "election", "parliament", "congress", "senate", "geopolitics",
            "foreign policy", "diplomacy", "sanctions", "legislation",
            "political party", "prime minister", "president",
        ),
        "agent_keywords": (
            "politics", "policy", "geopolitics", "election", "government",
        ),
    },
    {
        "id": "legal",
        "label": "Legal / compliance (informational)",
        "proposed_agent": "legal_research_agent",
        "markers": (
            "lawsuit", "contract law", "compliance", "regulation", "gdpr",
            "intellectual property", "patent", "trademark", "court ruling",
            "legal advice", "terms of service",
        ),
        "agent_keywords": ("legal", "law", "compliance", "regulation", "patent"),
    },
    {
        "id": "travel",
        "label": "Travel / itineraries",
        "proposed_agent": "travel_planning_agent",
        "markers": (
            "itinerary", "travel plan", "visa", "flight", "hotel", "backpack",
            "tourist", "sightseeing", "road trip", "places to visit",
        ),
        "agent_keywords": ("travel", "itinerary", "tourism", "visa"),
    },
    {
        "id": "education",
        "label": "Education / tutoring / study plans",
        "proposed_agent": "education_tutoring_agent",
        "markers": (
            "study plan", "exam prep", "tutoring", "curriculum", "homework help",
            "sat", "gre", "jee", "neet", "learn math", "flashcards",
            "online course",
        ),
        "agent_keywords": (
            "education", "tutoring", "study", "curriculum", "exam",
        ),
    },
    {
        "id": "sports",
        "label": "Sports analysis / fantasy",
        "proposed_agent": "sports_analysis_agent",
        "markers": (
            "cricket", "football", "soccer", "nba", "ipl", "premier league",
            "tennis", "olympics", "match prediction", "fantasy league",
            "player stats", "world cup", "who wins the match", "scorecard",
        ),
        "agent_keywords": ("sports", "cricket", "football", "nba", "athletics"),
    },
    {
        "id": "entertainment",
        "label": "Film / TV / music / pop culture",
        "proposed_agent": "entertainment_culture_agent",
        "markers": (
            "movie", "film", "bollywood", "hollywood", "netflix", "tv series",
            "oscar", "song", "album", "concert", "celebrity gossip",
            "box office", "trailer",
        ),
        "agent_keywords": (
            "entertainment", "cinema", "film", "movie", "music", "tv",
        ),
    },
    {
        "id": "business",
        "label": "Business / startups / strategy",
        "proposed_agent": "business_strategy_agent",
        "markers": (
            "startup", "business model", "go to market", "saas pricing",
            "pitch deck", "competitive analysis", "market sizing",
            "unit economics", "fundraising", "okr",
        ),
        "agent_keywords": (
            "business", "startup", "strategy", "saas", "gtm",
        ),
    },
    {
        "id": "real_estate",
        "label": "Real estate / housing",
        "proposed_agent": "real_estate_research_agent",
        "markers": (
            "real estate", "property", "mortgage", "rent vs buy", "housing",
            "apartment", "commercial lease", "cap rate",
        ),
        "agent_keywords": (
            "real_estate", "property", "housing", "mortgage", "realty",
        ),
    },
    {
        "id": "news",
        "label": "Current events / news briefing",
        "proposed_agent": "news_briefing_agent",
        "markers": (
            "breaking news", "today's news", "headlines", "current events",
            "what happened today", "news summary",
        ),
        "agent_keywords": ("news", "headline", "current events", "briefing"),
    },
    {
        "id": "environment",
        "label": "Environment / climate / sustainability",
        "proposed_agent": "environment_climate_agent",
        "markers": (
            "climate change", "carbon", "sustainability", "renewable energy",
            "pollution", "biodiversity", "esg", "net zero",
        ),
        "agent_keywords": (
            "environment", "climate", "sustainability", "carbon", "esg",
        ),
    },
    {
        "id": "general",
        "label": "General research / open-ended",
        "proposed_agent": "general_research_agent",
        "markers": (),  # fallback — never matched by markers
        "agent_keywords": (
            "general-purpose", "general purpose", "open-ended", "primary",
        ),
    },
)

DOMAIN_IDS: Tuple[str, ...] = tuple(str(s["id"]) for s in DOMAIN_SPECS)

_DOMAIN_BY_ID: Dict[str, Dict[str, object]] = {
    str(s["id"]): s for s in DOMAIN_SPECS
}

# Domains that must NEVER be served by equity/finance specialists.
NON_FINANCE_DOMAINS = frozenset(d for d in DOMAIN_IDS if d not in {"finance", "crypto"})

# Domains where a missing specialist should trigger CREATE (not silent primary).
SPECIALIST_CREATE_DOMAINS = frozenset(
    d for d in DOMAIN_IDS if d not in {"general", "news"}
)

GENERIC_AGENT_NAMES = frozenset({"primary_deep_agent", "general_research_agent"})

FINANCE_ONLY_AGENT_NAMES = frozenset(
    {
        "indian_stock_trader_3m",
        "us_stock_trader_3m",
        "us_stock_trader_3m_2",
        "market_analyst",
        "market_analyst_equity",
        "finance_analyst",
        "investment_strategist",
        "report_reviewer",
    }
)


def domain_ids_pipe_separated() -> str:
    return " | ".join(DOMAIN_IDS)


def domain_catalog_for_prompt() -> str:
    """Human-readable catalog injected into analyze/plan prompts."""
    lines = [
        "ALLOWED MACRO DOMAINS (pick exactly one id; prefer the most specific match):"
    ]
    for spec in DOMAIN_SPECS:
        lines.append(
            f"- {spec['id']}: {spec['label']} "
            f"(specialist name if creating: {spec['proposed_agent']})"
        )
    lines.append(
        "If none fit cleanly, use domain=general and prefer primary_deep_agent / "
        "general_research_agent rather than inventing a micro agent."
    )
    return "\n".join(lines)


def classify_task_domain(text: str) -> str:
    """Return the best-matching domain id from free text (first match wins)."""
    blob = (text or "").lower()
    for spec in DOMAIN_SPECS:
        domain_id = str(spec["id"])
        if domain_id == "general":
            continue
        markers: Sequence[str] = spec["markers"]  # type: ignore[assignment]
        if any(m in blob for m in markers):
            return domain_id
    return "general"


def proposed_agent_for_domain(domain: str) -> str:
    spec = _DOMAIN_BY_ID.get((domain or "").lower())
    if spec:
        return str(spec["proposed_agent"])
    return "general_research_agent"


def normalize_domain(domain: str) -> str:
    d = (domain or "").strip().lower().replace(" ", "_").replace("-", "_")
    aliases = {
        "comic": "comics",
        "superhero": "comics",
        "superheroes": "comics",
        "equity": "finance",
        "equities": "finance",
        "investing": "finance",
        "markets": "finance",
        "stocks": "finance",
        "workout": "fitness",
        "training": "fitness",
        "food": "nutrition",
        "cooking": "nutrition",
        "tech": "technology",
        "software": "technology",
        "programming": "technology",
        "film": "entertainment",
        "movies": "entertainment",
        "music": "entertainment",
        "pop_culture": "entertainment",
        "geo_politics": "politics",
        "geopolitics": "politics",
        "law": "legal",
        "realty": "real_estate",
        "property": "real_estate",
        "climate": "environment",
        "sustainability": "environment",
        "current_events": "news",
        "healthcare": "health",
        "medical": "health",
        "startup": "business",
        "cryptocurrency": "crypto",
    }
    d = aliases.get(d, d)
    return d if d in _DOMAIN_BY_ID else "general"


def infer_agent_domain(
    name: str, description: str = "", capabilities: Optional[List[str]] = None
) -> str:
    """Infer MACRO domain from an agent's name/description/capabilities."""
    n = (name or "").strip().lower()
    hay = f"{n} {description} {' '.join(capabilities or [])}".lower()
    if n in GENERIC_AGENT_NAMES or "general-purpose" in hay or "general purpose" in hay:
        return "general"
    if n in FINANCE_ONLY_AGENT_NAMES:
        return "finance"
    # Prefer non-general domains; skip empty-keyword general until end.
    for spec in DOMAIN_SPECS:
        domain_id = str(spec["id"])
        if domain_id == "general":
            continue
        keywords: Sequence[str] = spec["agent_keywords"]  # type: ignore[assignment]
        if any(k in hay for k in keywords):
            # Avoid classifying primary via incidental "finance" tool wording:
            if domain_id == "finance" and n in GENERIC_AGENT_NAMES:
                continue
            return domain_id
    return "general"


def domains_compatible(task_domain: str, agent_domain: str) -> bool:
    task_domain = normalize_domain(task_domain)
    agent_domain = normalize_domain(agent_domain)
    if task_domain == agent_domain:
        return True
    # Closely related pairs that may share specialists.
    related = {
        frozenset({"fitness", "nutrition", "health"}),
        frozenset({"finance", "crypto", "business"}),
        frozenset({"entertainment", "comics"}),
        frozenset({"news", "politics"}),
        frozenset({"science", "environment", "technology"}),
    }
    for group in related:
        if task_domain in group and agent_domain in group:
            return True
    # Generic research may backstop non-finance/crypto tasks.
    if agent_domain == "general" and task_domain not in {"finance", "crypto"}:
        return True
    if task_domain in {"finance", "crypto"} and agent_domain == "general":
        return True  # last-resort only; callers decide create vs primary
    return False


def is_finance_like_domain(domain: str) -> bool:
    return normalize_domain(domain) in {"finance", "crypto"}


def needs_specialist_create(domain: str) -> bool:
    return normalize_domain(domain) in SPECIALIST_CREATE_DOMAINS
