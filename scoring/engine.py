"""Deterministic, rule-based lead scoring.

The LLM only extracts facts; this module decides the score and category.
It depends on the extraction schema (plain Pydantic data) and the raw
customer messages, and on nothing from Streamlit or Groq.

Evidence levels
---------------
A field counts as **explicit** when it has a value, confidence >= 0.8 and its
evidence (quote, raw text or value) appears verbatim in the customer's
messages. A field with a value that fails either check is **inferred** and
earns reduced points. A field with no value is **missing** and earns 0.

Rules (maximum points in brackets)
----------------------------------
Requirement clarity [20]
    Service identified     6 explicit interested_service; 3 if inferred or only
                           a specific intent category (confidence >= 0.7)
    Requirement described  6 explicit; 3 inferred
    Features/modules       8 for >= 3 features, 6 for 2, 4 for 1;
                           halved (rounded down) if list confidence < 0.5
Budget [20]
    20 explicit numeric amount with currency
    16 explicit numeric amount, currency unknown
     8 budget mentioned without a number ("flexible", "reasonable")
     8 numeric amount but inferred (low confidence or not found in messages)
     0 missing
Timeline [15]
    15 explicit, urgent or within 30 days
    12 explicit, 31-90 days
     8 explicit, more than 90 days
     8 explicit, duration not parseable ("after Diwali")
     4 vague ("no rush", "not sure")
     5 inferred (any duration)
     0 missing
Purchase intent [20]
    20 high, confidence >= 0.7 and corroborated by an explicit budget,
       explicit timeline or contact details shared
    14 high but low confidence or uncorroborated
    12 medium (confidence >= 0.5); 8 medium with lower confidence
     4 low
     0 unknown
Business information [10]
    Company 4 explicit / 2 inferred; industry, location, business size
    2 explicit / 1 inferred each
Engagement [10]
    Customer messages      1 for 1 message, 2 for 2, 3 for 3+
    Contact details        2 email or phone found in the messages
    Name shared            1 explicit customer name
    Questions asked        2 at least one question extracted
    Buying signals         1 per extracted engagement indicator, max 2
Decision-making authority [5]
     5 explicit "yes"; 3 inferred "yes"; 0 "no" or unknown

Category: HOT 80-100, WARM 50-79, COLD 0-49.
"""

import re

from ai.schemas import Entities, IntentResult, PurchaseIntent, TextField
from scoring.models import CRITERIA_LABELS, CRITERIA_MAX, CriterionScore, EvidenceLevel, LeadScore, classify

EXPLICIT_CONFIDENCE = 0.8
NON_SERVICE_INTENTS = {"Pricing Enquiry", "General Enquiry", "Support"}

_URGENT = re.compile(
    r"\b(asap|urgent(ly)?|immediately|right away|today|tomorrow|this week|next week|"
    r"end of (the |this )?month|this month|at the earliest)\b"
)
_VAGUE = re.compile(
    r"\b(no rush|no hurry|not sure|unsure|undecided|not decided|someday|eventually|"
    r"sometime|maybe|later|flexible|no fixed|tbd|not urgent)\b"
)
_NUMBER_WORDS = {
    "a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
}
_NUM = r"(\d+(?:\.\d+)?|" + "|".join(_NUMBER_WORDS) + r")"
_DURATION = re.compile(r"\b" + _NUM + r"(?:\s*(?:-|–|to)\s*" + _NUM + r")?\s*(day|week|month|year)s?\b")
_UNIT_DAYS = {"day": 1, "week": 7, "month": 30, "year": 365}


# ---------------------------------------------------------------- evidence

def _norm(text: str) -> str:
    text = text.casefold().replace("’", "'").replace("“", '"').replace("”", '"')
    return " ".join(text.split()).strip(" .,!?\"'")


def _appears(snippet: str | None, haystack: str) -> bool:
    return bool(snippet) and _norm(snippet) in haystack


def _level(field: TextField, haystack: str) -> EvidenceLevel:
    if field.value is None:
        return "missing"
    verified = _appears(field.evidence, haystack) or _appears(field.value, haystack)
    return "explicit" if field.confidence >= EXPLICIT_CONFIDENCE and verified else "inferred"


def _digits(text: str) -> str:
    return re.sub(r"\D", "", text)


def _contact_shared(entities: Entities, haystack: str) -> bool:
    if entities.email.value and entities.email.value.casefold() in haystack:
        return True
    number = _digits(entities.whatsapp_number.value or "")
    return len(number) >= 7 and number[-10:] in _digits(haystack)


def timeline_days(text: str) -> int | None:
    """Days until the stated deadline, 0 for urgent phrases, None if unparseable."""
    t = text.casefold()
    if _URGENT.search(t):
        return 0
    if "next month" in t:
        return 45
    match = _DURATION.search(t)
    if not match:
        return None
    lo, hi, unit = match.groups()
    amount = hi or lo  # use the upper bound of a range
    value = _NUMBER_WORDS.get(amount) or float(amount)
    return round(value * _UNIT_DAYS[unit])


# ---------------------------------------------------------------- criteria

def _criterion(key: str, earned: int, reason: str, evidence: EvidenceLevel) -> CriterionScore:
    return CriterionScore(
        key=key,
        label=CRITERIA_LABELS[key],
        earned=min(earned, CRITERIA_MAX[key]),
        maximum=CRITERIA_MAX[key],
        reason=reason,
        evidence=evidence,
    )


def _overall(levels: list[EvidenceLevel]) -> EvidenceLevel:
    if "explicit" in levels:
        return "explicit"
    return "inferred" if "inferred" in levels else "missing"


def _requirement_clarity(e: Entities, intent: IntentResult, hay: str, missing: list[str]):
    points, notes, levels = 0, [], []

    service = _level(e.interested_service, hay)
    specific_intent = intent.category not in NON_SERVICE_INTENTS and intent.confidence >= 0.7
    if service == "explicit":
        points += 6
        notes.append(f"service stated ({e.interested_service.value})")
    elif service == "inferred" or specific_intent:
        points += 3
        notes.append(f"service inferred ({e.interested_service.value or intent.category})")
        levels.append("inferred")
    else:
        missing.append("Service/product of interest")
    levels.append(service)

    desc = _level(e.requirement_description, hay)
    if desc == "explicit":
        points += 6
        notes.append("requirement described")
    elif desc == "inferred":
        points += 3
        notes.append("requirement partly described")
    else:
        missing.append("Requirement description")
    levels.append(desc)

    n = len(e.requested_features.values)
    feature_points = {0: 0, 1: 4, 2: 6}.get(n, 8)
    if n and e.requested_features.confidence < 0.5:
        feature_points //= 2
    points += feature_points
    if n:
        notes.append(f"{n} feature(s) listed")
        levels.append("explicit" if e.requested_features.confidence >= EXPLICIT_CONFIDENCE else "inferred")
    else:
        missing.append("Specific modules/features")

    reason = "; ".join(notes) if notes else "No requirement details yet"
    reason = reason[0].upper() + reason[1:]
    return _criterion("requirement_clarity", points, reason, _overall(levels))


def _budget(e: Entities, hay: str, missing: list[str]):
    b = e.budget
    if not b.is_known:
        missing.append("Budget (amount and currency)")
        return _criterion("budget", 0, "No budget mentioned", "missing")

    has_amount = b.amount_min is not None or b.amount_max is not None
    verified = _appears(b.raw_text, hay)
    explicit = b.confidence >= EXPLICIT_CONFIDENCE and verified

    if has_amount and explicit and b.currency:
        return _criterion("budget", 20, f"Explicit budget: {b.display()}", "explicit")
    if has_amount and explicit:
        missing.append("Budget currency")
        return _criterion("budget", 16, f"Explicit budget {b.display()}, currency not stated", "explicit")
    if has_amount:
        missing.append("Confirm budget amount")
        return _criterion("budget", 8, f"Budget {b.display()} inferred, not confirmed in the customer's words", "inferred")
    missing.append("Specific budget amount")
    return _criterion("budget", 8, f"Budget mentioned without an amount ('{b.raw_text}')",
                      "explicit" if verified else "inferred")


def _timeline(e: Entities, hay: str, missing: list[str]):
    t = e.timeline
    level = _level(t, hay)
    if level == "missing":
        missing.append("Expected timeline")
        return _criterion("timeline", 0, "No timeline mentioned", "missing")
    if level == "inferred":
        missing.append("Confirm timeline")
        return _criterion("timeline", 5, f"Timeline inferred ('{t.value}'), not stated clearly", "inferred")

    if _VAGUE.search(t.value.casefold()) and timeline_days(t.value) is None:
        missing.append("Specific timeline/deadline")
        return _criterion("timeline", 4, f"Vague timeline ('{t.value}')", "explicit")
    days = timeline_days(t.value)
    if days is None:
        missing.append("Specific timeline/deadline")
        return _criterion("timeline", 8, f"Timeline stated but duration unclear ('{t.value}')", "explicit")
    if days <= 30:
        return _criterion("timeline", 15, f"Urgent timeline ('{t.value}')", "explicit")
    if days <= 90:
        return _criterion("timeline", 12, f"Near-term timeline ('{t.value}', ~{days} days)", "explicit")
    return _criterion("timeline", 8, f"Long-term timeline ('{t.value}', ~{days} days)", "explicit")


def _purchase_intent(pi: PurchaseIntent, corroborated: bool, missing: list[str]):
    if pi.level == "high":
        if pi.confidence >= 0.7 and corroborated:
            return _criterion("purchase_intent", 20, "High intent, backed by budget/timeline/contact details", "inferred")
        missing.append("Concrete buying signal (budget, timeline or contact details)")
        return _criterion("purchase_intent", 14, "High intent signalled but not yet corroborated", "inferred")
    if pi.level == "medium":
        missing.append("Confirmation of readiness to proceed")
        points = 12 if pi.confidence >= 0.5 else 8
        return _criterion("purchase_intent", points, "Medium intent: interested but still exploring", "inferred")
    if pi.level == "low":
        missing.append("Confirmation of readiness to proceed")
        return _criterion("purchase_intent", 4, "Low intent: early research or just browsing", "inferred")
    missing.append("Indication of buying intent")
    return _criterion("purchase_intent", 0, "Purchase intent unknown", "missing")


def _business_information(e: Entities, hay: str, missing: list[str]):
    parts = [("company_name", "Company name", 4), ("industry", "Industry", 2),
             ("location", "Location", 2), ("business_size", "Business size", 2)]
    points, known, levels = 0, [], []
    for field, label, full in parts:
        level = _level(getattr(e, field), hay)
        levels.append(level)
        if level == "explicit":
            points += full
            known.append(label.lower())
        elif level == "inferred":
            points += full // 2
            known.append(f"{label.lower()} (inferred)")
        else:
            missing.append(label)
    reason = ("Known: " + ", ".join(known)) if known else "No business details yet"
    return _criterion("business_information", points, reason, _overall(levels))


def _engagement(e: Entities, customer_messages: list[str], hay: str, missing: list[str]):
    points, notes = 0, []
    count = len(customer_messages)
    if count:
        points += min(count, 3)
        notes.append(f"{count} message(s)")
    if _contact_shared(e, hay):
        points += 2
        notes.append("shared contact details")
    else:
        missing.append("Contact details (email or phone)")
    if _level(e.customer_name, hay) == "explicit":
        points += 1
        notes.append("shared name")
    if e.questions.values:
        points += 2
        notes.append(f"asked {len(e.questions.values)} question(s)")
    signals = min(len(e.engagement_indicators.values), 2)
    if signals:
        points += signals
        notes.append(f"{signals} buying signal(s)")
    reason = ("Customer " + ", ".join(notes)) if notes else "No engagement yet"
    return _criterion("engagement", points, reason, "explicit" if count else "missing")


def _decision_authority(e: Entities, hay: str, missing: list[str]):
    d = e.decision_authority
    if d.value == "yes":
        verified = _appears(d.evidence, hay)
        if d.confidence >= EXPLICIT_CONFIDENCE and verified:
            return _criterion("decision_authority", 5, "Customer stated they make the decision", "explicit")
        missing.append("Confirm decision-maker")
        return _criterion("decision_authority", 3, "Likely decision-maker (inferred)", "inferred")
    missing.append("Decision-maker contact" if d.value == "no" else "Decision-maker confirmation")
    if d.value == "no":
        return _criterion("decision_authority", 0, "Someone else makes the decision", "explicit")
    return _criterion("decision_authority", 0, "Decision authority unknown", "missing")


# ---------------------------------------------------------------- public

def score_lead(
    entities: Entities,
    intent: IntentResult,
    purchase_intent: PurchaseIntent,
    customer_messages: list[str],
) -> LeadScore:
    """Score a lead from validated extraction data and the customer's messages."""
    hay = _norm(" \n ".join(customer_messages))
    missing: list[str] = []

    budget = _budget(entities, hay, missing)
    timeline = _timeline(entities, hay, missing)
    corroborated = (
        budget.evidence == "explicit" and budget.earned >= 16
        or timeline.evidence == "explicit" and timeline.earned >= 8
        or _contact_shared(entities, hay)
    )
    criteria = [
        _requirement_clarity(entities, intent, hay, missing),
        budget,
        timeline,
        _purchase_intent(purchase_intent, corroborated, missing),
        _business_information(entities, hay, missing),
        _engagement(entities, customer_messages, hay, missing),
        _decision_authority(entities, hay, missing),
    ]
    total = sum(c.earned for c in criteria)
    return LeadScore(
        total=total,
        category=classify(total),
        criteria=criteria,
        missing_information=list(dict.fromkeys(missing)),
    )
