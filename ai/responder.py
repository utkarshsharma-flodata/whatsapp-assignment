"""Customer-facing reply generation.

The reply is written by the LLM, but *what* to ask is decided here in Python:
``plan_questions`` picks at most two missing topics, in a priority order that
depends on the service intent, and never a topic that is already known. The
model only sees facts the customer gave plus business details from the
business profile: never scores, categories or other developer-panel data.

Every reply is checked before it is accepted: no prices that are neither
configured nor stated by the customer, and no internal scoring language.
A failing reply is regenerated once with feedback; if it fails again an
error is raised and nothing is shown or saved.
"""

import re
from dataclasses import dataclass, field

from ai.errors import ResponseValidationError
from ai.groq_client import complete_text, create_client
from ai.schemas import LeadAnalysis
from core.business import BusinessProfile
from core.config import Settings, get_settings
from core.logging_setup import get_logger

log = get_logger("responder")

MAX_HISTORY_MESSAGES = 20
NON_SERVICE_INTENTS = {"Pricing Enquiry", "General Enquiry", "Support"}

# Topic -> how the model should understand the question.
TOPICS = {
    "confirm_conflict": "confirm which of two conflicting details is correct",
    "issue_details": "what exactly is going wrong (which system/page, since when, any error message)",
    "requirement": "what they need and the main goal of the project",
    "features": "the key features or modules they need",
    "budget": "their approximate budget for this (with currency)",
    "timeline": "when they would like it ready or to start",
    "company": "their company or business name",
    "name": "their name",
    "contact": "an email address to send details to",
    "decision": "whether anyone else is involved in the decision",
}
_PRIORITY = {
    "service": ["requirement", "features", "budget", "timeline", "company", "name", "contact", "decision"],
    "Pricing Enquiry": ["requirement", "features", "timeline", "budget", "company", "name", "contact"],
    "Support": ["issue_details", "company", "name", "contact"],
    "General Enquiry": ["requirement", "name", "company"],
}


@dataclass
class AssistantReply:
    text: str
    asked_topics: list[str]
    model: str
    regenerated: bool = False
    warnings: list[str] = field(default_factory=list)


# --------------------------------------------------------------- planning

def _known_topics(profile: LeadAnalysis, contact_known: bool) -> set[str]:
    e = profile.entities
    known = set()
    if e.requirement_description.value or e.interested_service.value:
        known |= {"requirement", "issue_details"}
    if e.requested_features.values:
        known.add("features")
    if e.budget.is_known:
        known.add("budget")
    if e.timeline.value:
        known.add("timeline")
    if e.company_name.value:
        known.add("company")
    if e.customer_name.value:
        known.add("name")
    if contact_known or e.email.value or e.whatsapp_number.value:
        known.add("contact")
    if e.decision_authority.value != "unknown":
        known.add("decision")
    return known


def plan_questions(
    profile: LeadAnalysis | None,
    latest_message: str,
    contact_known: bool = False,
    recently_asked: list[list[str]] | None = None,
) -> list[str]:
    """Pick at most two topics to ask about next (one if the customer asked something).

    ``recently_asked`` holds the topics of the last assistant replies (most
    recent last). A topic asked in each of the last two replies and still
    unanswered is skipped, so an ignored question is not repeated a third time.
    """
    recent = (recently_asked or [])[-2:]
    ignored = set(recent[0]).intersection(*recent[1:]) if len(recent) == 2 else set()
    if profile is None:
        return [] if "requirement" in ignored else ["requirement"]
    if profile.conflicts:
        return ["confirm_conflict"]

    category = profile.intent.category
    order = _PRIORITY.get(category, _PRIORITY["service"]) if category in NON_SERVICE_INTENTS else _PRIORITY["service"]
    known = _known_topics(profile, contact_known)
    missing = [t for t in order if t not in known and t not in ignored]

    # Ask about the decision process only once the core requirement, budget and timeline are known.
    if "decision" in missing and not {"requirement", "budget", "timeline"} <= known:
        missing.remove("decision")
    limit = 1 if "?" in latest_message else 2
    return missing[:limit]


# ----------------------------------------------------------------- prompt

def _known_facts(profile: LeadAnalysis | None) -> list[str]:
    if profile is None:
        return []
    e = profile.entities
    facts = [
        ("Name", e.customer_name.value),
        ("Company", e.company_name.value),
        ("Location", e.location.value),
        ("Industry", e.industry.value),
        ("Business size", e.business_size.value),
        ("Service of interest", e.interested_service.value),
        ("Requirement", e.requirement_description.value),
        ("Requested features", ", ".join(e.requested_features.values) or None),
        ("Budget", e.budget.display()),
        ("Timeline", e.timeline.value),
        ("Email", e.email.value),
    ]
    return [f"- {label}: {value}" for label, value in facts if value]


def _business_section(business: BusinessProfile, services: list[str]) -> str:
    name = business.name or 'not configured (refer to the company as "our team")'
    lines = [f"Company name: {name}"]
    if business.description:
        lines.append(f"About: {business.description}")
    if services:
        lines.append("Services offered: " + ", ".join(services))
    for label, value in (("Office location", business.office_location), ("Working hours", business.working_hours),
                         ("Sales email", business.contact_email), ("Support contact", business.support_contact)):
        lines.append(f"{label}: {value or 'not configured (say the team will share it)'}")
    if business.pricing:
        lines.append("Configured prices (quote ONLY these, word for word, and only for the matching service):")
        lines += [f"- {service}: {price}" for service, price in business.pricing.items()]
    else:
        lines.append("Configured prices: NONE. Do not state any price, range, rate or estimate.")
    return "\n".join(lines)


def build_system_prompt(
    profile: LeadAnalysis | None,
    questions: list[str],
    business: BusinessProfile,
    services: list[str],
) -> str:
    known = _known_facts(profile)
    ask = [f"- {TOPICS[t]}" for t in questions]
    if "confirm_conflict" in questions and profile:
        ask += [f"  ({c.field}: '{c.previous}' or '{c.new}'?)" for c in profile.conflicts]
    customer_questions = profile.entities.questions.values if profile else []
    return f"""You are the WhatsApp assistant of a technology services company. Write the next reply to the customer.

## How to reply
- Respond to the customer's latest message first, naturally and specifically. Use the whole conversation for context.
- WhatsApp style: friendly, professional, concise (usually 2-5 short sentences), plain text, no markdown headings or tables, at most one emoji.
- Reply in the customer's language.
- Ask ONLY the follow-up questions listed under "Ask next", phrased naturally. Do not add other questions. If "Ask next" is empty, do not ask for more details: confirm the next step (the team will review and get back to them).
- NEVER ask for anything listed under "Already known". You may use those details to personalise the reply.
- Pricing: if prices are configured below, quote only those exact texts. Otherwise explain honestly that the cost depends on their requirements (scope, features, timeline) and that the team will share a quote once they understand the needs. Never invent numbers, ranges, discounts or packages.
- Support requests: acknowledge the problem with empathy, collect the details listed under "Ask next", and say the support team will look into it. Do not promise fixes, refunds or resolution times.
- General questions: answer only from the business information below. If something is not configured, say the team will confirm it. Never invent addresses, hours, staff names, clients, delivery dates or capabilities.
- You are a virtual assistant; if asked, do not claim to be human.
- Never mention lead scores, qualification, categories (hot/warm/cold), purchase intent, internal notes, or that the conversation is being analysed or extracted.

## Business information
{_business_section(business, services)}

## Already known (provided by the customer)
{chr(10).join(known) or "- nothing yet"}

## Customer questions so far
{chr(10).join(f"- {q}" for q in customer_questions) or "- none"}

## Ask next
{chr(10).join(ask) or "- nothing: do not ask for more details"}
"""


# ------------------------------------------------------------- validation

_MONEY = re.compile(
    r"(?:₹|rs\.?|inr|usd|us\$|\$|€|eur|£|gbp|aed)\s?\d[\d,.]*\s?(?:k|lakh|lakhs|lac|crore|cr|thousand|million|m)?\b"
    r"|\d[\d,.]*\s?(?:k|lakh|lakhs|lac|crore|thousand|million)?\s?(?:₹|rs\.?|inr|usd|dollars|rupees|eur|euros|gbp|aed)\b"
    r"|\d[\d,.]*\s?(?:lakh|lakhs|lac|crore)\b",
    re.IGNORECASE,
)
_INTERNAL = re.compile(
    r"\b(lead score|your score|qualification score|hot lead|warm lead|cold lead|purchase intent|"
    r"lead (?:category|qualification)|\d{1,3}\s?/\s?100)\b",
    re.IGNORECASE,
)
# Developer-only vocabulary and artefacts that must never reach the customer.
_DEVELOPER = re.compile(
    r"\b(developer (?:panel|sidebar)|system prompt|extract(?:ed|ion)|confidence (?:score|level)|"
    r"entities|json|is_correction|interested_service|requirement_description)\b|^\s*[{\[]",
    re.IGNORECASE,
)


_AMOUNT = re.compile(r"(\d[\d,]*(?:\.\d+)?)\s?(k|lakhs?|lacs?|crores?|cr|thousand|million|m)?\b", re.IGNORECASE)
_MULTIPLIER = {"k": 1e3, "thousand": 1e3, "lakh": 1e5, "lakhs": 1e5, "lac": 1e5, "lacs": 1e5,
               "crore": 1e7, "crores": 1e7, "cr": 1e7, "million": 1e6, "m": 1e6}


def _amounts(text: str) -> set[float]:
    """Numeric values in the text, with k/lakh/crore expanded ("50k" and "50,000" -> 50000)."""
    values = set()
    for number, unit in _AMOUNT.findall(text):
        try:
            values.add(round(float(number.replace(",", "")) * _MULTIPLIER.get(unit.lower(), 1), 2))
        except ValueError:
            continue
    return values


def validate_reply(text: str, customer_text: str, business: BusinessProfile) -> list[str]:
    """Return a list of problems; empty means the reply is acceptable."""
    problems = []
    if not text.strip():
        return ["The reply is empty."]
    allowed = _amounts(customer_text + " " + " ".join(business.pricing.values()))
    for match in _MONEY.finditer(text):
        if _amounts(match.group()) - allowed:
            problems.append(f"Mentions the amount '{match.group().strip()}', which is neither configured nor stated "
                            "by the customer. Do not state prices or amounts.")
    if _INTERNAL.search(text):
        problems.append("Mentions internal lead scoring or qualification. Never mention these to the customer.")
    if _DEVELOPER.search(text):
        problems.append("Contains internal or technical details (extraction, JSON, field names). "
                        "Write a natural customer-facing message only.")
    return problems


# ------------------------------------------------------------- generation

def generate_reply(
    messages: list[dict],
    profile: LeadAnalysis | None,
    business: BusinessProfile,
    services: list[str],
    contact_known: bool = False,
    recently_asked: list[list[str]] | None = None,
    settings: Settings | None = None,
    client=None,
) -> AssistantReply:
    """Write the next assistant reply. Raises an NLPError subclass on failure."""
    settings = settings or get_settings()
    client = client or create_client(settings)

    history = messages[-MAX_HISTORY_MESSAGES:]
    latest = next((m["content"] for m in reversed(history) if m["role"] == "user"), "")
    customer_text = " ".join(m["content"] for m in history if m["role"] == "user")
    questions = plan_questions(profile, latest, contact_known, recently_asked)

    chat = [{"role": "system", "content": build_system_prompt(profile, questions, business, services)}]
    chat += [{"role": "user" if m["role"] == "user" else "assistant", "content": m["content"]} for m in history]

    problems: list[str] = []
    for attempt in range(2):
        text = complete_text(client, settings, chat).strip()
        problems = validate_reply(text, customer_text, business)
        if not problems:
            return AssistantReply(text=text, asked_topics=questions, model=settings.groq_model,
                                  regenerated=attempt > 0)
        log.warning("Reply draft %d rejected by safety checks: %d problem(s)", attempt + 1, len(problems))
        chat += [
            {"role": "assistant", "content": text},
            {"role": "system", "content": "That reply was rejected: " + " ".join(problems)
                                          + " Rewrite the reply following all the rules."},
        ]
    raise ResponseValidationError("The generated reply was rejected after a retry: " + " ".join(problems))
