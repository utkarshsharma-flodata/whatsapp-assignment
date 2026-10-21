"""One customer turn: analyse -> merge/score/persist -> reply -> save reply.

Independent of Streamlit. The conversation history is always read from the
database, so a reopened conversation has exactly the same context as a live
one. The reply is generated only after the lead has been updated, from the
stored profile.
"""

import json
import sqlite3
from dataclasses import dataclass
from typing import Literal

from ai.errors import NLPError
from ai.groq_client import create_client
from ai.nlp_service import analyze_conversation
from ai.responder import AssistantReply, generate_reply
from ai.schemas import LeadAnalysis
from core.business import BusinessProfile, load_business_profile
from core.config import Settings
from core.logging_setup import get_logger
from database import repository as repo
from services import lead_service
from services.lead_service import PersistOutcome

log = get_logger("turns")

Stage = Literal["analysis", "save", "reply"]


@dataclass
class TurnResult:
    outcome: PersistOutcome | None = None
    reply: AssistantReply | None = None
    error: str | None = None
    failed_stage: Stage | None = None

    @property
    def ok(self) -> bool:
        return self.error is None


def history(conn: sqlite3.Connection, conversation_id: int) -> list[dict]:
    """Conversation messages in the role/content shape the AI layer uses."""
    return [
        {"role": "user" if m["sender_role"] == "customer" else "assistant", "content": m["message_text"]}
        for m in repo.get_messages(conn, conversation_id)
    ]


def _reply(conn, conversation_id: int, profile: LeadAnalysis | None, settings: Settings,
           business: BusinessProfile | None, client) -> AssistantReply:
    conversation = repo.get_conversation(conn, conversation_id)
    lead = repo.get_lead(conn, conversation["lead_id"]) if conversation["lead_id"] else None
    contact_known = bool(conversation["sender_phone"] or (lead and (lead["phone"] or lead["email"])))
    services = [s["name"] for s in repo.list_services(conn)]
    business = business or load_business_profile(settings.business_profile_path)
    recently_asked = [json.loads(m["asked_topics"]) for m in repo.get_messages(conn, conversation_id)
                      if m["sender_role"] == "assistant" and m["asked_topics"] is not None][-2:]
    reply = generate_reply(history(conn, conversation_id), profile, business, services,
                           contact_known=contact_known, recently_asked=recently_asked,
                           settings=settings, client=client)
    lead_service.record_assistant_message(conn, conversation_id, reply.text, asked_topics=reply.asked_topics)
    return reply


def process_turn(
    conn: sqlite3.Connection,
    conversation_id: int,
    settings: Settings,
    business: BusinessProfile | None = None,
    client=None,
    generate: bool = True,
) -> TurnResult:
    """Analyse the latest customer message, update the lead, then reply.

    The previous profile is read from the database (the linked lead's stored
    profile), so live, reopened and returning conversations behave the same.
    Set ``generate=False`` to re-run the analysis without writing a reply.
    """
    try:
        previous = lead_service.load_conversation(conn, conversation_id)["profile"]
        client = client or create_client(settings)
        analysis = analyze_conversation(history(conn, conversation_id), previous=previous,
                                        settings=settings, client=client)
    except (NLPError, ValueError) as exc:  # ValueError: unknown conversation or corrupt stored profile
        log.warning("Turn for conversation #%s failed at analysis: %s", conversation_id, type(exc).__name__)
        try:
            lead_service.record_analysis_failure(conn, conversation_id, str(exc))
        except sqlite3.Error:
            log.exception("Could not record the analysis failure")
        return TurnResult(error=f"Analysis failed: {exc}", failed_stage="analysis")

    try:
        outcome = lead_service.persist_analysis(conn, conversation_id, analysis,
                                                default_country_code=settings.default_country_code)
    except (sqlite3.Error, ValueError) as exc:
        log.exception("Turn for conversation #%s failed while saving the lead", conversation_id)
        return TurnResult(error=f"The lead could not be saved (database error: {exc})", failed_stage="save")

    if not generate:
        return TurnResult(outcome=outcome)
    try:
        reply = _reply(conn, conversation_id, outcome.profile, settings, business, client)
    except (NLPError, ValueError, sqlite3.Error) as exc:
        log.warning("Turn for conversation #%s failed at reply: %s", conversation_id, type(exc).__name__)
        return TurnResult(outcome=outcome, error=f"Reply could not be generated: {exc}", failed_stage="reply")
    return TurnResult(outcome=outcome, reply=reply)


def retry_reply(
    conn: sqlite3.Connection,
    conversation_id: int,
    settings: Settings,
    business: BusinessProfile | None = None,
    client=None,
) -> TurnResult:
    """Generate the reply again from the stored lead profile (after a reply failure)."""
    try:
        profile = lead_service.load_conversation(conn, conversation_id)["profile"]
        reply = _reply(conn, conversation_id, profile, settings, business, client or create_client(settings))
    except (NLPError, ValueError, sqlite3.Error) as exc:
        log.warning("Reply retry for conversation #%s failed: %s", conversation_id, type(exc).__name__)
        return TurnResult(error=f"Reply could not be generated: {exc}", failed_stage="reply")
    return TurnResult(reply=reply)
