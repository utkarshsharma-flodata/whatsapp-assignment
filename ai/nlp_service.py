"""Intent detection and entity extraction over the accumulated conversation.

Flow: build transcript -> call Groq (JSON mode) -> validate with Pydantic
(one corrective retry) -> drop unsupported contact details -> merge with the
previous analysis (preserve, correct, or flag conflicts) -> list missing info.
"""

import re

from pydantic import ValidationError

from ai import prompts
from ai.errors import ExtractionValidationError
from ai.groq_client import complete_json, create_client
from ai.schemas import (
    BudgetField,
    Entities,
    FieldConflict,
    LeadAnalysis,
    ListField,
    LLMExtraction,
    TextField,
)
from core.config import Settings, get_settings
from core.logging_setup import get_logger

log = get_logger("nlp")

# Older messages are summarised by the previously extracted entities.
MAX_CONTEXT_MESSAGES = 30

TEXT_FIELDS = [
    "customer_name", "whatsapp_number", "company_name", "email", "location",
    "industry", "business_size", "requirement_description", "interested_service",
    "timeline", "lead_source",
]
LIST_FIELDS = ["requested_features", "engagement_indicators", "questions", "objections"]

FIELD_LABELS = {
    "customer_name": "Customer name",
    "whatsapp_number": "WhatsApp number",
    "company_name": "Company name",
    "email": "Email",
    "location": "Location",
    "industry": "Industry",
    "business_size": "Business size",
    "requirement_description": "Requirement",
    "requested_features": "Requested features",
    "interested_service": "Interested service",
    "budget": "Budget",
    "timeline": "Timeline",
    "lead_source": "Lead source",
    "decision_authority": "Decision authority",
    "engagement_indicators": "Engagement indicators",
    "questions": "Questions",
    "objections": "Objections",
}


def analyze_conversation(
    messages: list[dict],
    previous: LeadAnalysis | None = None,
    settings: Settings | None = None,
    client=None,
) -> LeadAnalysis:
    """Analyse the conversation and return a validated, merged LeadAnalysis.

    Raises an NLPError subclass on configuration, API or validation failure.
    """
    settings = settings or get_settings()
    client = client or create_client(settings)

    relevant = [m for m in messages if not m.get("placeholder")][-MAX_CONTEXT_MESSAGES:]
    customer_text = " ".join(m["content"] for m in relevant if m["role"] == "user")

    extraction = _extract(client, settings, relevant, previous)
    warnings = _drop_unsupported_contacts(extraction.entities, customer_text)
    if warnings:
        log.warning("Discarded %d contact detail(s) not present in the customer's messages", len(warnings))

    entities, conflicts, corrections = _merge(
        previous.entities if previous else None,
        extraction.entities,
        previous.conflicts if previous else [],
    )
    return LeadAnalysis(
        intent=extraction.intent,
        purchase_intent=extraction.purchase_intent,
        entities=entities,
        missing_information=missing_information(entities),
        conflicts=conflicts,
        corrections=(previous.corrections if previous else []) + corrections,
        warnings=warnings,
        model=settings.groq_model,
        analysed_messages=sum(1 for m in messages if m["role"] == "user"),
    )


def _extract(client, settings, messages, previous) -> LLMExtraction:
    transcript = "\n".join(
        f"{'Customer' if m['role'] == 'user' else 'Business'}: {m['content']}" for m in messages
    )
    previous_entities = previous.entities.model_dump(exclude_defaults=True) if previous else None
    chat = [
        {"role": "system", "content": prompts.SYSTEM_PROMPT},
        {"role": "user", "content": prompts.build_user_prompt(transcript, previous_entities)},
    ]

    last_error = None
    for _attempt in range(2):
        try:
            raw = complete_json(client, settings, chat)
            return LLMExtraction.model_validate_json(raw)
        except ValidationError as exc:
            last_error = _summarise_validation(exc)
            log.warning("Extraction output failed validation (attempt %d): %s", _attempt + 1, last_error)
            chat += [
                {"role": "assistant", "content": raw},
                {"role": "user", "content": prompts.build_retry_prompt(last_error)},
            ]
        except ExtractionValidationError as exc:
            last_error = str(exc)
            log.warning("Extraction returned invalid JSON (attempt %d)", _attempt + 1)
            chat.append({"role": "user", "content": prompts.build_retry_prompt(last_error)})

    raise ExtractionValidationError(
        f"The model's response did not match the extraction schema after a retry: {last_error}"
    )


def _summarise_validation(exc: ValidationError, limit: int = 8) -> str:
    lines = [
        f"{'.'.join(str(p) for p in err['loc']) or '<root>'}: {err['msg']}"
        for err in exc.errors()[:limit]
    ]
    return "; ".join(lines)


def _digits(text: str) -> str:
    return re.sub(r"\D", "", text)


_PHONE_LIKE = re.compile(r"\+?\d[\d\s\-().]{5,}\d")


def _number_in_text(number: str, text: str) -> bool:
    """True if the number appears in the text, ignoring formatting.

    Matches on the last 10 digits so an added/omitted country code still matches.
    """
    digits = _digits(number)
    if len(digits) < 7:
        return False
    for candidate in (_digits(m) for m in _PHONE_LIKE.findall(text)):
        if candidate == digits or (min(len(candidate), len(digits)) >= 10 and candidate[-10:] == digits[-10:]):
            return True
    return False


def _drop_unsupported_contacts(entities: Entities, customer_text: str) -> list[str]:
    """Discard emails/numbers that do not literally appear in customer messages."""
    warnings = []
    email = entities.email.value
    if email and email.lower() not in customer_text.lower():
        warnings.append(f"Discarded email '{email}': not found in the customer's messages.")
        entities.email = TextField()
    number = entities.whatsapp_number.value
    if number:
        if not _number_in_text(number, customer_text):
            warnings.append(f"Discarded number '{number}': not found in the customer's messages.")
            entities.whatsapp_number = TextField()
    return warnings


def _normalise(field: str, value: str) -> str:
    if field == "whatsapp_number":
        return _digits(value)
    return " ".join(value.casefold().split())


def _merge(
    old: Entities | None,
    new: Entities,
    old_conflicts: list[FieldConflict],
) -> tuple[Entities, list[FieldConflict], list[str]]:
    """Preserve earlier values, apply explicit corrections, and flag conflicts."""
    if old is None:
        return new, [], []

    merged = old.model_copy(deep=True)
    conflicts = list(old_conflicts)
    corrections = []

    def flag(field: str, previous: str, value: str) -> None:
        conflict = FieldConflict(field=FIELD_LABELS[field], previous=previous, new=value)
        if conflict not in conflicts:
            conflicts.append(conflict)

    def resolve(field: str) -> None:
        conflicts[:] = [c for c in conflicts if c.field != FIELD_LABELS[field]]

    for field in TEXT_FIELDS:
        prev: TextField = getattr(old, field)
        cur: TextField = getattr(new, field)
        if cur.value is None:
            continue
        if prev.value is None:
            setattr(merged, field, cur)
        elif _normalise(field, prev.value) == _normalise(field, cur.value):
            setattr(merged, field, cur.model_copy(update={"confidence": max(prev.confidence, cur.confidence)}))
        elif cur.is_correction:
            setattr(merged, field, cur)
            resolve(field)
            corrections.append(f"{FIELD_LABELS[field]}: '{prev.value}' → '{cur.value}'")
        elif field in ("requirement_description", "interested_service"):
            # Free-text summaries naturally get refined as the conversation grows.
            setattr(merged, field, cur)
        else:
            flag(field, prev.value, cur.value)

    for field in LIST_FIELDS:
        prev_list: ListField = getattr(old, field)
        cur_list: ListField = getattr(new, field)
        seen = {v.casefold() for v in prev_list.values}
        combined = prev_list.values + [v for v in cur_list.values if v.casefold() not in seen]
        setattr(merged, field, ListField(values=combined, confidence=max(prev_list.confidence, cur_list.confidence)))

    merged.budget = _merge_budget(old.budget, new.budget, flag, resolve, corrections)

    if new.decision_authority.value != "unknown":
        prev_auth = old.decision_authority.value
        if prev_auth in ("unknown", new.decision_authority.value):
            merged.decision_authority = new.decision_authority
        else:
            flag("decision_authority", prev_auth, new.decision_authority.value)

    return merged, conflicts, corrections


def _merge_budget(prev: BudgetField, cur: BudgetField, flag, resolve, corrections) -> BudgetField:
    if not cur.is_known:
        return prev
    if not prev.is_known:
        return cur
    same = (prev.amount_min, prev.amount_max, prev.currency) == (cur.amount_min, cur.amount_max, cur.currency)
    if same:
        return cur.model_copy(update={"confidence": max(prev.confidence, cur.confidence)})
    if cur.is_correction:
        resolve("budget")
        corrections.append(f"Budget: '{prev.display()}' → '{cur.display()}'")
        return cur
    if prev.currency is None and cur.currency and (prev.amount_min, prev.amount_max) == (cur.amount_min, cur.amount_max):
        return cur  # only added a currency
    flag("budget", prev.display(), cur.display())
    return prev


def merge_analyses(base: LeadAnalysis, new: LeadAnalysis) -> LeadAnalysis:
    """Merge a new analysis into an existing lead profile with the same rules
    used within a conversation (preserve, correct, or flag conflicts)."""
    old_conflicts = base.conflicts + [c for c in new.conflicts if c not in base.conflicts]
    entities, conflicts, corrections = _merge(base.entities, new.entities, old_conflicts)
    all_corrections = base.corrections + [c for c in new.corrections if c not in base.corrections]
    all_corrections += [c for c in corrections if c not in all_corrections]
    return new.model_copy(update={
        "entities": entities,
        "conflicts": conflicts,
        "corrections": all_corrections,
        "missing_information": missing_information(entities),
    })


def missing_information(entities: Entities) -> list[str]:
    """Fields still needed to qualify the lead."""
    missing = []
    if not entities.customer_name.value:
        missing.append("Customer name")
    if not entities.company_name.value:
        missing.append("Company name")
    if not (entities.email.value or entities.whatsapp_number.value):
        missing.append("Contact details (email or phone)")
    if not (entities.interested_service.value or entities.requirement_description.value):
        missing.append("Requirement / service")
    if not entities.budget.is_known:
        missing.append("Budget")
    if not entities.timeline.value:
        missing.append("Timeline")
    if entities.decision_authority.value == "unknown":
        missing.append("Decision-making authority")
    return missing
