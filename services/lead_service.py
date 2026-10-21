"""Lead persistence workflow: identify the lead, merge, re-score, record history.

Independent of Streamlit. Every public function takes an open SQLite
connection and performs its writes in a single transaction.

Lead identity (in priority order)
---------------------------------
1. Normalised phone: the conversation's sender phone (WhatsApp metadata or
   the number entered in the simulator), else a phone the customer typed.
2. Normalised email, only if the matching lead has no phone or the same phone.
3. The lead already linked to the conversation (an explicitly reopened lead).
Otherwise a new lead is created. Two customers without phone or email are
never assumed to be the same person.

If an identified conversation turns out to belong to an existing lead, the
provisional lead is merged into it (history moved, no duplicate kept) unless
both leads have different phone numbers, in which case both are kept and a
possible duplicate is logged.
"""

import json
import re
import sqlite3
import uuid
from dataclasses import dataclass, field
from typing import Literal

from ai.nlp_service import merge_analyses
from ai.schemas import LeadAnalysis
from database import repository as repo
from database.connection import transaction
from core.logging_setup import get_logger
from scoring import LeadScore, score_lead

log = get_logger("leads")

_EMAIL = re.compile(r"[^@\s]+@[^@\s]+\.[^@\s]+")

_COLUMN_LABELS = {
    "customer_name": "Customer name", "phone": "Phone", "email": "Email", "company": "Company",
    "location": "Location", "industry": "Industry", "business_size": "Business size",
    "requirement": "Requirement", "requested_modules": "Requested modules",
    "service_interest": "Service/product", "budget_min": "Budget", "budget_max": "Budget",
    "budget_text": "Budget", "currency": "Currency", "timeline": "Timeline",
    "service_intent": "Service intent", "purchase_intent": "Purchase intent",
    "decision_authority": "Decision authority", "lead_source": "Lead source",
    "score": "Score", "category": "Category",
}


@dataclass
class PersistOutcome:
    lead_id: int
    action: Literal["created", "updated", "unchanged", "merged"]
    matched_by: Literal["phone", "email", "conversation"] | None
    changed_fields: list[str]
    score_saved: bool
    previous_score: int | None
    profile: LeadAnalysis
    score: LeadScore
    notes: list[str] = field(default_factory=list)

    def summary(self) -> str:
        if self.action == "created":
            return f"New lead #{self.lead_id} created"
        if self.action == "merged":
            return f"Returning customer: merged into existing lead #{self.lead_id} (matched by {self.matched_by})"
        if self.matched_by in ("phone", "email"):
            prefix = f"Returning customer matched by {self.matched_by}: lead #{self.lead_id}"
        else:
            prefix = f"Existing lead #{self.lead_id}"
        if self.action == "unchanged":
            return f"{prefix}, no profile changes"
        return f"{prefix} updated: {', '.join(self.changed_fields) or 'score'}"


# ------------------------------------------------------------------ helpers

def normalize_phone(raw: str | None, default_country_code: str = "91") -> str | None:
    """Return an E.164-style number (``+<digits>``) or None if not a plausible phone."""
    if not raw:
        return None
    digits = re.sub(r"\D", "", raw)
    if raw.strip().startswith("00"):
        digits = digits[2:]
    elif len(digits) == 11 and digits.startswith("0"):
        digits = digits[1:]  # national trunk prefix
    if len(digits) == 10 and default_country_code:
        digits = default_country_code + digits
    return f"+{digits}" if 8 <= len(digits) <= 15 else None


def normalize_email(raw: str | None) -> str | None:
    if not raw:
        return None
    email = raw.strip().lower()
    return email if _EMAIL.fullmatch(email) else None


def _profile(lead: dict | None) -> LeadAnalysis | None:
    if lead and lead.get("profile_json"):
        return LeadAnalysis.model_validate_json(lead["profile_json"])
    return None


def _score_from_row(row: dict | None) -> LeadScore | None:
    if row is None:
        return None
    return LeadScore(
        total=row["total_score"],
        category=row["category"],
        criteria=json.loads(row["breakdown_json"]),
        missing_information=json.loads(row["missing_json"]),
    )


def find_matching_lead(conn: sqlite3.Connection, phone: str | None, email: str | None):
    """Return (lead, matched_by) using phone first, then email."""
    if phone:
        lead = repo.find_lead_by_phone(conn, phone)
        if lead:
            return lead, "phone"
    if email:
        for lead in repo.find_leads_by_email(conn, email):
            if phone is None or lead["phone_normalized"] in (None, phone):
                return lead, "email"
    return None, None


def _lead_fields(conn, profile: LeadAnalysis, score: LeadScore, phone: str | None, phone_display: str | None) -> dict:
    e = profile.entities
    budget = e.budget
    return {
        "customer_name": e.customer_name.value,
        "phone": phone_display if phone else None,
        "phone_normalized": phone,
        "email": e.email.value,
        "email_normalized": normalize_email(e.email.value),
        "company": e.company_name.value,
        "location": e.location.value,
        "industry": e.industry.value,
        "business_size": e.business_size.value,
        "requirement": e.requirement_description.value,
        "requested_modules": json.dumps(e.requested_features.values) if e.requested_features.values else None,
        "service_interest": e.interested_service.value,
        "service_id": repo.get_service_id(conn, profile.intent.category),
        "budget_min": budget.amount_min,
        "budget_max": budget.amount_max,
        "budget_text": budget.raw_text,
        "currency": budget.currency,
        "timeline": e.timeline.value,
        "service_intent": profile.intent.category,
        "purchase_intent": profile.purchase_intent.level,
        "decision_authority": e.decision_authority.value,
        "lead_source": e.lead_source.value,
        "score": score.total,
        "category": score.category,
        "profile_json": profile.model_dump_json(),
    }


def _explanation(score: LeadScore) -> str:
    return "; ".join(f"{c.label} {c.earned}/{c.maximum}: {c.reason}" for c in score.criteria)


# --------------------------------------------------------------- workflows

def start_conversation(
    conn: sqlite3.Connection,
    sender_phone: str | None = None,
    lead_id: int | None = None,
    channel: str = "simulator",
    session_identifier: str | None = None,
    default_country_code: str = "91",
) -> int:
    """Create a conversation, optionally for a known sender phone or an existing lead."""
    phone = normalize_phone(sender_phone, default_country_code)
    if sender_phone and phone is None:
        raise ValueError(f"'{sender_phone}' is not a valid phone number.")
    session_identifier = session_identifier or f"{channel}-{uuid.uuid4().hex}"
    with transaction(conn):
        if lead_id is None and phone:
            existing = repo.find_lead_by_phone(conn, phone)
            lead_id = existing["id"] if existing else None
        conversation_id = repo.create_conversation(
            conn, session_identifier, channel=channel, sender_phone=phone, lead_id=lead_id
        )
        if lead_id:
            repo.log_activity(conn, lead_id, "conversation_started",
                              f"Conversation #{conversation_id} started ({channel})")
    return conversation_id


def record_customer_message(conn: sqlite3.Connection, conversation_id: int, text: str,
                            sent_at: str | None = None) -> int:
    """Store a customer message (pending analysis) and update contact dates."""
    sent_at = sent_at or repo.utc_now()
    with transaction(conn):
        message_id = repo.add_message(conn, conversation_id, "customer", text, sent_at)
        conversation = repo.get_conversation(conn, conversation_id)
        if conversation["lead_id"]:
            repo.touch_lead_contact(conn, conversation["lead_id"], sent_at)
    return message_id


def record_assistant_message(conn: sqlite3.Connection, conversation_id: int, text: str,
                             asked_topics: list[str] | None = None, sent_at: str | None = None) -> int:
    """Store an assistant reply (not analysed) with the follow-up topics it asked about."""
    with transaction(conn):
        return repo.add_message(conn, conversation_id, "assistant", text, sent_at, processing_status="skipped",
                                asked_topics=asked_topics)


def record_analysis_failure(conn: sqlite3.Connection, conversation_id: int, error: str) -> None:
    with transaction(conn):
        repo.mark_pending_messages(conn, conversation_id, "failed", error)


def persist_analysis(
    conn: sqlite3.Connection,
    conversation_id: int,
    analysis: LeadAnalysis,
    default_country_code: str = "91",
    now: str | None = None,
) -> PersistOutcome:
    """Create or update the lead for a conversation from a validated analysis."""
    now = now or repo.utc_now()
    with transaction(conn):
        conversation = repo.get_conversation(conn, conversation_id)
        if conversation is None:
            raise ValueError(f"Conversation {conversation_id} does not exist")
        current = repo.get_lead(conn, conversation["lead_id"]) if conversation["lead_id"] else None

        typed_phone = analysis.entities.whatsapp_number.value
        phone = conversation["sender_phone"] or normalize_phone(typed_phone, default_country_code)
        phone_display = conversation["sender_phone"] or typed_phone
        email = normalize_email(analysis.entities.email.value)
        match, matched_by = find_matching_lead(conn, phone, email)
        notes: list[str] = []

        if current and match and match["id"] != current["id"]:
            if current["phone_normalized"] and match["phone_normalized"] \
                    and current["phone_normalized"] != match["phone_normalized"]:
                note = f"Possible duplicate of lead #{match['id']} (same {matched_by}, different phone); not merged"
                repo.log_activity(conn, current["id"], "possible_duplicate", note, now)
                notes.append(note)
                match = None

        if current is None and match is None:
            action, profile = "created", analysis
            lead_id = repo.create_lead(conn, {}, contact_time=conversation["started_at"], now=now)
            repo.link_conversation(conn, conversation_id, lead_id, now)
            repo.log_activity(conn, lead_id, "lead_created", f"Lead created from conversation #{conversation_id}", now)
            before_profile = None
        elif current is None:
            action, lead_id = "updated", match["id"]
            before_profile = _profile(match)
            profile = merge_analyses(before_profile, analysis) if before_profile else analysis
            repo.link_conversation(conn, conversation_id, lead_id, now)
            repo.log_activity(conn, lead_id, "returning_customer",
                              f"Conversation #{conversation_id} matched to this lead by {matched_by}", now)
        elif match is None or match["id"] == current["id"]:
            # The analysis was seeded with this lead's profile, so it is already merged.
            action, lead_id, profile = "updated", current["id"], analysis
            before_profile = _profile(current)
            # Already linked (earlier message, reopened lead, or sender phone at start).
            matched_by = "conversation"
        else:
            action, lead_id = "merged", match["id"]
            before_profile = _profile(match)
            profile = merge_analyses(before_profile, analysis) if before_profile else analysis
            repo.reassign_lead(conn, current["id"], lead_id)
            repo.log_activity(conn, lead_id, "lead_merged",
                              f"Lead #{current['id']} merged into this lead (matched by {matched_by})", now)

        # Never move a phone number that another lead already owns.
        owner = repo.find_lead_by_phone(conn, phone) if phone else None
        if owner and owner["id"] != lead_id:
            notes.append(f"Phone {phone} belongs to lead #{owner['id']}; not copied")
            phone = None

        customer_messages = repo.get_customer_messages_for_lead(conn, lead_id)
        score = score_lead(profile.entities, profile.intent, profile.purchase_intent, customer_messages)

        changed = repo.update_lead(conn, lead_id, _lead_fields(conn, profile, score, phone, phone_display), now)
        repo.touch_lead_contact(conn, lead_id, conversation["last_message_at"], now)
        changed_labels = list(dict.fromkeys(_COLUMN_LABELS.get(c, c) for c in changed if c not in ("score", "category")))

        previous = repo.latest_score(conn, lead_id)
        breakdown = [c.model_dump() for c in score.criteria]
        score_saved = previous is None or previous["total_score"] != score.total \
            or json.loads(previous["breakdown_json"]) != breakdown
        if score_saved:
            repo.add_score(conn, lead_id, score.total, score.category, breakdown, score.missing_information,
                           _explanation(score), conversation_id, now)
            if previous is None:
                repo.log_activity(conn, lead_id, "score_calculated", f"Score {score.total} ({score.category})", now)
            elif previous["total_score"] != score.total:
                repo.log_activity(conn, lead_id, "score_changed",
                                  f"Score {previous['total_score']} → {score.total} ({score.category})", now)

        if action != "created" and changed_labels:
            repo.log_activity(conn, lead_id, "lead_updated", "Updated: " + ", ".join(changed_labels), now)
        old_corrections = set(before_profile.corrections) if before_profile else set()
        for correction in profile.corrections:
            if correction not in old_corrections:
                repo.log_activity(conn, lead_id, "field_corrected", correction, now)
        old_conflicts = before_profile.conflicts if before_profile else []
        for conflict in profile.conflicts:
            if conflict not in old_conflicts:
                repo.log_activity(conn, lead_id, "conflict_detected",
                                  f"{conflict.field}: kept '{conflict.previous}', customer also said '{conflict.new}'", now)

        repo.save_conversation_analysis(conn, conversation_id, profile.model_dump_json(), now)
        repo.mark_pending_messages(conn, conversation_id, "processed")

        if action == "updated" and not changed_labels and not score_saved and matched_by == "conversation":
            action = "unchanged"
        log.info("Lead #%s %s (conversation #%s, matched_by=%s, changed=%s, score=%s %s, history_row=%s)",
                 lead_id, action, conversation_id, matched_by, changed_labels, score.total, score.category,
                 score_saved)
        return PersistOutcome(
            lead_id=lead_id,
            action=action,
            matched_by=matched_by if action != "created" else None,
            changed_fields=changed_labels,
            score_saved=score_saved,
            previous_score=previous["total_score"] if previous else None,
            profile=profile,
            score=score,
            notes=notes,
        )


def load_conversation(conn: sqlite3.Connection, conversation_id: int) -> dict:
    """Everything needed to reopen a conversation in the UI."""
    conversation = repo.get_conversation(conn, conversation_id)
    if conversation is None:
        raise ValueError(f"Conversation {conversation_id} does not exist")
    lead = repo.get_lead(conn, conversation["lead_id"]) if conversation["lead_id"] else None
    profile = _profile(lead)
    if profile is None and conversation["analysis_json"]:
        profile = LeadAnalysis.model_validate_json(conversation["analysis_json"])
    return {
        "conversation": conversation,
        "lead": lead,
        "messages": repo.get_messages(conn, conversation_id),
        "profile": profile,
        "score": _score_from_row(repo.latest_score(conn, lead["id"])) if lead else None,
    }


def load_lead(conn: sqlite3.Connection, lead_id: int) -> dict:
    """Lead record with its stored profile and latest score (for a new conversation)."""
    lead = repo.get_lead(conn, lead_id)
    if lead is None:
        raise ValueError(f"Lead {lead_id} does not exist")
    return {"lead": lead, "profile": _profile(lead), "score": _score_from_row(repo.latest_score(conn, lead_id))}

