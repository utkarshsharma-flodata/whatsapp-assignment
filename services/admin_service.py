"""Admin CRM workflows: lifecycle changes, assignment, notes and follow-ups.

Independent of Streamlit. Admin actions only touch admin-managed columns
(status, salesperson, follow-up dates, conversion) plus history tables, so
conversations, messages, the extracted profile and score history are never
modified. Every action is recorded in ``activities``.
"""

import json
import sqlite3
from datetime import date

from ai.schemas import LeadAnalysis
from database import repository as repo
from database.connection import transaction
from core.logging_setup import get_logger
from scoring import LeadScore

log = get_logger("admin")

STAGES = ["new", "contacted", "qualified", "proposal", "negotiation", "converted", "lost"]
STAGE_LABELS = {s: s.capitalize() for s in STAGES}
CLOSED_STAGES = {"converted", "lost"}
FOLLOW_UP_OUTCOMES = ("done", "missed", "cancelled")


def change_status(conn: sqlite3.Connection, lead_id: int, new_status: str, note: str | None = None,
                  now: str | None = None) -> bool:
    """Move a lead to a lifecycle stage. Returns False if it was already there.

    Any move is allowed (so mistakes can be corrected); moving a closed lead
    back to an open stage reopens it.
    """
    if new_status not in STAGES:
        raise ValueError(f"Unknown status '{new_status}'")
    now = now or repo.utc_now()
    note = (note or "").strip() or None
    with transaction(conn):
        lead = repo.get_lead(conn, lead_id)
        if lead is None:
            raise ValueError(f"Lead {lead_id} does not exist")
        old_status = lead["status"]
        if old_status == new_status:
            return False
        repo.update_lead_admin(conn, lead_id, {
            "status": new_status,
            "status_changed_at": now,
            "conversion_status": {"converted": "won", "lost": "lost"}.get(new_status, "open"),
            "closed_at": now if new_status in CLOSED_STAGES else None,
        }, now)
        repo.add_status_history(conn, lead_id, old_status, new_status, note, now)
        verb = "reopened" if old_status in CLOSED_STAGES and new_status not in CLOSED_STAGES else "status_changed"
        description = f"{STAGE_LABELS[old_status]} → {STAGE_LABELS[new_status]}" + (f": {note}" if note else "")
        repo.log_activity(conn, lead_id, verb, description, now)
    log.info("Lead #%s status %s -> %s", lead_id, old_status, new_status)
    return True


def mark_converted(conn, lead_id: int, note: str | None = None, now: str | None = None) -> bool:
    return change_status(conn, lead_id, "converted", note, now)


def mark_lost(conn, lead_id: int, reason: str | None = None, now: str | None = None) -> bool:
    return change_status(conn, lead_id, "lost", reason, now)


def assign_salesperson(conn: sqlite3.Connection, lead_id: int, name: str | None, now: str | None = None) -> bool:
    name = (name or "").strip() or None
    with transaction(conn):
        changes = repo.update_lead_admin(conn, lead_id, {"assigned_salesperson": name}, now)
        if not changes:
            return False
        old = changes["assigned_salesperson"][0]
        description = f"Assigned to {name}" if name else f"Unassigned (was {old})"
        if name and old:
            description += f" (was {old})"
        repo.log_activity(conn, lead_id, "assigned" if name else "unassigned", description, now)
    return True


def add_note(conn: sqlite3.Connection, lead_id: int, text: str, now: str | None = None) -> int:
    text = (text or "").strip()
    if not text:
        raise ValueError("A note cannot be empty.")
    with transaction(conn):
        if repo.get_lead(conn, lead_id) is None:
            raise ValueError(f"Lead {lead_id} does not exist")
        return repo.log_activity(conn, lead_id, "note", text, now)


def _sync_next_follow_up(conn, lead_id: int, now: str | None) -> None:
    repo.update_lead_admin(conn, lead_id, {"next_follow_up_date": repo.earliest_pending_follow_up(conn, lead_id)},
                           now)


def schedule_follow_up(conn: sqlite3.Connection, lead_id: int, due_date: date | str, next_action: str | None = None,
                       now: str | None = None) -> int:
    due = due_date.isoformat() if isinstance(due_date, date) else date.fromisoformat(due_date).isoformat()
    next_action = (next_action or "").strip() or None
    with transaction(conn):
        if repo.get_lead(conn, lead_id) is None:
            raise ValueError(f"Lead {lead_id} does not exist")
        follow_up_id = repo.add_follow_up(conn, lead_id, due, next_action, now)
        _sync_next_follow_up(conn, lead_id, now)
        repo.log_activity(conn, lead_id, "follow_up_scheduled",
                          f"Follow-up due {due}" + (f": {next_action}" if next_action else ""), now)
    return follow_up_id


def complete_follow_up(conn: sqlite3.Connection, follow_up_id: int, outcome: str,
                       customer_response: str | None = None, next_action: str | None = None,
                       now: str | None = None) -> None:
    """Record a follow-up as done, missed or cancelled."""
    if outcome not in FOLLOW_UP_OUTCOMES:
        raise ValueError(f"Unknown follow-up outcome '{outcome}'")
    now = now or repo.utc_now()
    with transaction(conn):
        follow_up = repo.get_follow_up(conn, follow_up_id)
        if follow_up is None:
            raise ValueError(f"Follow-up {follow_up_id} does not exist")
        repo.update_follow_up(conn, follow_up_id, outcome, (customer_response or "").strip() or None,
                              (next_action or "").strip() or None, now if outcome == "done" else None, now)
        _sync_next_follow_up(conn, follow_up["lead_id"], now)
        detail = f"Follow-up #{follow_up['follow_up_count']} (due {follow_up['due_date']}) {outcome}"
        if customer_response and customer_response.strip():
            detail += f". Response: {customer_response.strip()}"
        repo.log_activity(conn, follow_up["lead_id"], f"follow_up_{outcome}", detail, now)


def lead_detail(conn: sqlite3.Connection, lead_id: int) -> dict | None:
    """Everything the lead detail view shows."""
    lead = repo.get_lead(conn, lead_id)
    if lead is None:
        return None
    latest = repo.latest_score(conn, lead_id)
    conversations = repo.list_conversations(conn, lead_id)
    return {
        "lead": lead,
        "profile": LeadAnalysis.model_validate_json(lead["profile_json"]) if lead["profile_json"] else None,
        "score": LeadScore(total=latest["total_score"], category=latest["category"],
                           criteria=json.loads(latest["breakdown_json"]),
                           missing_information=json.loads(latest["missing_json"])) if latest else None,
        "score_history": repo.score_history(conn, lead_id),
        "conversations": [{**c, "messages": repo.get_messages(conn, c["id"])} for c in conversations],
        "activities": repo.list_activities(conn, lead_id, limit=200),
        "follow_ups": repo.list_follow_ups(conn, lead_id),
        "status_history": repo.list_status_history(conn, lead_id),
    }
