"""Data-access functions. All SQL lives here and is parameterised.

Functions take an open connection (see ``database.connection``) and return
plain dicts, so callers never handle cursors or SQL.
"""

import json
import sqlite3
from datetime import datetime, timezone


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _row(row: sqlite3.Row | None) -> dict | None:
    return dict(row) if row is not None else None


def _rows(rows) -> list[dict]:
    return [dict(r) for r in rows]


# Columns callers may set through create_lead / update_lead.
LEAD_COLUMNS = (
    "customer_name", "phone", "phone_normalized", "email", "email_normalized", "company",
    "location", "industry", "business_size", "requirement", "requested_modules",
    "service_interest", "service_id", "budget_min", "budget_max", "budget_text", "currency",
    "timeline", "service_intent", "purchase_intent", "decision_authority", "lead_source",
    "score", "category", "status", "assigned_salesperson", "next_follow_up_date",
    "conversion_status", "profile_json",
)
# Bookkeeping columns that are not reported as "changed fields".
_SILENT_COLUMNS = {"profile_json", "phone_normalized", "email_normalized", "service_id"}


def _check_columns(fields: dict) -> None:
    unknown = set(fields) - set(LEAD_COLUMNS)
    if unknown:
        raise ValueError(f"Unknown lead columns: {sorted(unknown)}")


# ------------------------------------------------------------------ services

def list_services(conn: sqlite3.Connection, active_only: bool = True) -> list[dict]:
    sql = "SELECT * FROM services" + (" WHERE is_active = 1" if active_only else "") + " ORDER BY name"
    return _rows(conn.execute(sql))


def get_service_id(conn: sqlite3.Connection, name: str | None) -> int | None:
    if not name:
        return None
    row = conn.execute("SELECT id FROM services WHERE name = ? AND is_active = 1", (name,)).fetchone()
    return row["id"] if row else None


# --------------------------------------------------------------------- leads

def create_lead(conn: sqlite3.Connection, fields: dict, contact_time: str, now: str | None = None) -> int:
    _check_columns(fields)
    now = now or utc_now()
    data = {**fields, "first_contact_date": contact_time, "last_contact_date": contact_time,
            "created_at": now, "updated_at": now}
    columns = ", ".join(data)
    placeholders = ", ".join("?" for _ in data)
    cur = conn.execute(f"INSERT INTO leads ({columns}) VALUES ({placeholders})", tuple(data.values()))
    return cur.lastrowid


def get_lead(conn: sqlite3.Connection, lead_id: int) -> dict | None:
    return _row(conn.execute("SELECT * FROM leads WHERE id = ?", (lead_id,)).fetchone())


def find_lead_by_phone(conn: sqlite3.Connection, phone_normalized: str) -> dict | None:
    return _row(conn.execute(
        "SELECT * FROM leads WHERE phone_normalized = ?", (phone_normalized,)
    ).fetchone())


def find_leads_by_email(conn: sqlite3.Connection, email_normalized: str) -> list[dict]:
    return _rows(conn.execute(
        "SELECT * FROM leads WHERE email_normalized = ? ORDER BY last_contact_date DESC", (email_normalized,)
    ))


def update_lead(conn: sqlite3.Connection, lead_id: int, fields: dict, now: str | None = None) -> list[str]:
    """Update the given columns. ``None`` values never overwrite stored values.

    Returns the names of user-visible columns whose value changed.
    """
    _check_columns(fields)
    current = get_lead(conn, lead_id)
    if current is None:
        raise ValueError(f"Lead {lead_id} does not exist")
    changes = {k: v for k, v in fields.items() if v is not None and current[k] != v}
    if changes:
        assignments = ", ".join(f"{k} = ?" for k in changes)
        conn.execute(
            f"UPDATE leads SET {assignments}, updated_at = ? WHERE id = ?",
            (*changes.values(), now or utc_now(), lead_id),
        )
    return [k for k in changes if k not in _SILENT_COLUMNS]


def touch_lead_contact(conn: sqlite3.Connection, lead_id: int, contact_time: str, now: str | None = None) -> None:
    conn.execute(
        "UPDATE leads SET last_contact_date = MAX(last_contact_date, ?), updated_at = ? WHERE id = ?",
        (contact_time, now or utc_now(), lead_id),
    )


def list_leads(conn: sqlite3.Connection, limit: int = 200) -> list[dict]:
    return _rows(conn.execute(
        """
        SELECT l.*, (SELECT COUNT(*) FROM conversations c WHERE c.lead_id = l.id) AS conversation_count
        FROM leads l
        ORDER BY l.last_contact_date DESC
        LIMIT ?
        """,
        (limit,),
    ))


def reassign_lead(conn: sqlite3.Connection, source_id: int, target_id: int) -> None:
    """Move everything owned by ``source_id`` to ``target_id`` and delete the source lead."""
    for table in ("conversations", "lead_scores", "follow_ups", "activities", "lead_status_history"):
        conn.execute(f"UPDATE {table} SET lead_id = ? WHERE lead_id = ?", (target_id, source_id))
    conn.execute(
        """
        UPDATE leads SET first_contact_date = MIN(first_contact_date,
                             (SELECT first_contact_date FROM leads WHERE id = ?))
        WHERE id = ?
        """,
        (source_id, target_id),
    )
    conn.execute("DELETE FROM leads WHERE id = ?", (source_id,))


# ------------------------------------------------------------- conversations

def create_conversation(
    conn: sqlite3.Connection,
    session_identifier: str,
    channel: str = "simulator",
    sender_phone: str | None = None,
    lead_id: int | None = None,
    now: str | None = None,
) -> int:
    now = now or utc_now()
    cur = conn.execute(
        """
        INSERT INTO conversations
            (lead_id, channel, session_identifier, sender_phone, started_at, last_message_at, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (lead_id, channel, session_identifier, sender_phone, now, now, now, now),
    )
    return cur.lastrowid


def get_conversation(conn: sqlite3.Connection, conversation_id: int) -> dict | None:
    return _row(conn.execute("SELECT * FROM conversations WHERE id = ?", (conversation_id,)).fetchone())


def link_conversation(conn: sqlite3.Connection, conversation_id: int, lead_id: int, now: str | None = None) -> None:
    conn.execute(
        "UPDATE conversations SET lead_id = ?, updated_at = ? WHERE id = ?",
        (lead_id, now or utc_now(), conversation_id),
    )


def save_conversation_analysis(conn: sqlite3.Connection, conversation_id: int, analysis_json: str,
                               now: str | None = None) -> None:
    conn.execute(
        "UPDATE conversations SET analysis_json = ?, updated_at = ? WHERE id = ?",
        (analysis_json, now or utc_now(), conversation_id),
    )


def list_conversations(conn: sqlite3.Connection, lead_id: int | None) -> list[dict]:
    """Conversations of a lead, or unlinked conversations when ``lead_id`` is None."""
    where = "c.lead_id = ?" if lead_id is not None else "c.lead_id IS NULL"
    params = (lead_id,) if lead_id is not None else ()
    return _rows(conn.execute(
        f"""
        SELECT c.*, COUNT(m.id) AS message_count
        FROM conversations c LEFT JOIN messages m ON m.conversation_id = c.id
        WHERE {where}
        GROUP BY c.id
        ORDER BY c.last_message_at DESC
        """,
        params,
    ))


# ------------------------------------------------------------------ messages

def add_message(
    conn: sqlite3.Connection,
    conversation_id: int,
    sender_role: str,
    text: str,
    sent_at: str | None = None,
    processing_status: str = "pending",
    asked_topics: list[str] | None = None,
) -> int:
    sent_at = sent_at or utc_now()
    cur = conn.execute(
        """
        INSERT INTO messages (conversation_id, sender_role, message_text, sent_at, processing_status, asked_topics)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (conversation_id, sender_role, text, sent_at, processing_status,
         json.dumps(asked_topics) if asked_topics is not None else None),
    )
    conn.execute(
        "UPDATE conversations SET last_message_at = MAX(last_message_at, ?), updated_at = ? WHERE id = ?",
        (sent_at, utc_now(), conversation_id),
    )
    return cur.lastrowid


def get_messages(conn: sqlite3.Connection, conversation_id: int) -> list[dict]:
    return _rows(conn.execute(
        "SELECT * FROM messages WHERE conversation_id = ? ORDER BY sent_at, id", (conversation_id,)
    ))


def mark_pending_messages(conn: sqlite3.Connection, conversation_id: int, status: str,
                          error: str | None = None) -> int:
    """Set the status of all pending/failed customer messages in a conversation."""
    cur = conn.execute(
        """
        UPDATE messages SET processing_status = ?, processing_error = ?
        WHERE conversation_id = ? AND sender_role = 'customer' AND processing_status IN ('pending', 'failed')
        """,
        (status, error, conversation_id),
    )
    return cur.rowcount


def get_customer_messages_for_lead(conn: sqlite3.Connection, lead_id: int) -> list[str]:
    return [r["message_text"] for r in conn.execute(
        """
        SELECT m.message_text FROM messages m JOIN conversations c ON c.id = m.conversation_id
        WHERE c.lead_id = ? AND m.sender_role = 'customer'
        ORDER BY m.sent_at, m.id
        """,
        (lead_id,),
    )]


# -------------------------------------------------------------------- scores

def add_score(
    conn: sqlite3.Connection,
    lead_id: int,
    total: int,
    category: str,
    breakdown: list[dict],
    missing: list[str],
    explanation: str,
    conversation_id: int | None = None,
    now: str | None = None,
) -> int:
    cur = conn.execute(
        """
        INSERT INTO lead_scores
            (lead_id, conversation_id, total_score, category, breakdown_json, missing_json, explanation, scored_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (lead_id, conversation_id, total, category, json.dumps(breakdown), json.dumps(missing),
         explanation, now or utc_now()),
    )
    return cur.lastrowid


def latest_score(conn: sqlite3.Connection, lead_id: int) -> dict | None:
    return _row(conn.execute(
        "SELECT * FROM lead_scores WHERE lead_id = ? ORDER BY scored_at DESC, id DESC LIMIT 1", (lead_id,)
    ).fetchone())


def score_history(conn: sqlite3.Connection, lead_id: int) -> list[dict]:
    return _rows(conn.execute(
        "SELECT * FROM lead_scores WHERE lead_id = ? ORDER BY scored_at, id", (lead_id,)
    ))


# ---------------------------------------------------------------- follow-ups

def add_follow_up(conn: sqlite3.Connection, lead_id: int, due_date: str, next_action: str | None = None,
                  now: str | None = None) -> int:
    """Schedule a follow-up; ``follow_up_count`` numbers the lead's follow-ups 1, 2, 3..."""
    now = now or utc_now()
    cur = conn.execute(
        """
        INSERT INTO follow_ups (lead_id, due_date, next_action, follow_up_count, created_at, updated_at)
        VALUES (?, ?, ?, (SELECT COUNT(*) + 1 FROM follow_ups WHERE lead_id = ?), ?, ?)
        """,
        (lead_id, due_date, next_action, lead_id, now, now),
    )
    return cur.lastrowid


def get_follow_up(conn: sqlite3.Connection, follow_up_id: int) -> dict | None:
    return _row(conn.execute("SELECT * FROM follow_ups WHERE id = ?", (follow_up_id,)).fetchone())


def update_follow_up(
    conn: sqlite3.Connection,
    follow_up_id: int,
    status: str,
    customer_response: str | None = None,
    next_action: str | None = None,
    completed_at: str | None = None,
    now: str | None = None,
) -> None:
    conn.execute(
        """
        UPDATE follow_ups
        SET status = ?, customer_response = COALESCE(?, customer_response),
            next_action = COALESCE(?, next_action), completed_at = ?, updated_at = ?
        WHERE id = ?
        """,
        (status, customer_response, next_action, completed_at, now or utc_now(), follow_up_id),
    )


def list_follow_ups(conn: sqlite3.Connection, lead_id: int) -> list[dict]:
    return _rows(conn.execute("SELECT * FROM follow_ups WHERE lead_id = ? ORDER BY due_date, id", (lead_id,)))


def earliest_pending_follow_up(conn: sqlite3.Connection, lead_id: int) -> str | None:
    row = conn.execute(
        "SELECT MIN(due_date) AS due FROM follow_ups WHERE lead_id = ? AND status = 'pending'", (lead_id,)
    ).fetchone()
    return row["due"]


# ------------------------------------------------------------- lead admin

# Columns the admin workflow may set, including back to NULL.
ADMIN_COLUMNS = ("status", "assigned_salesperson", "next_follow_up_date", "conversion_status",
                 "status_changed_at", "closed_at")
# Columns offered as dashboard filters / distinct-value lists.
FILTER_COLUMNS = ("category", "status", "service_intent", "industry", "lead_source", "assigned_salesperson")


def update_lead_admin(conn: sqlite3.Connection, lead_id: int, fields: dict, now: str | None = None) -> dict:
    """Set admin-managed columns (``None`` clears). Returns {column: (old, new)} for changed columns."""
    unknown = set(fields) - set(ADMIN_COLUMNS)
    if unknown:
        raise ValueError(f"Not an admin column: {sorted(unknown)}")
    current = get_lead(conn, lead_id)
    if current is None:
        raise ValueError(f"Lead {lead_id} does not exist")
    changes = {k: v for k, v in fields.items() if current[k] != v}
    if changes:
        assignments = ", ".join(f"{k} = ?" for k in changes)
        conn.execute(f"UPDATE leads SET {assignments}, updated_at = ? WHERE id = ?",
                     (*changes.values(), now or utc_now(), lead_id))
    return {k: (current[k], v) for k, v in changes.items()}


def add_status_history(conn: sqlite3.Connection, lead_id: int, from_status: str | None, to_status: str,
                       note: str | None = None, now: str | None = None) -> int:
    cur = conn.execute(
        "INSERT INTO lead_status_history (lead_id, from_status, to_status, note, changed_at) VALUES (?, ?, ?, ?, ?)",
        (lead_id, from_status, to_status, note, now or utc_now()),
    )
    return cur.lastrowid


def list_status_history(conn: sqlite3.Connection, lead_id: int) -> list[dict]:
    return _rows(conn.execute(
        "SELECT * FROM lead_status_history WHERE lead_id = ? ORDER BY changed_at, id", (lead_id,)
    ))


def distinct_lead_values(conn: sqlite3.Connection, column: str) -> list[str]:
    if column not in FILTER_COLUMNS:
        raise ValueError(f"Not a filter column: {column}")
    return [r[0] for r in conn.execute(
        f"SELECT DISTINCT {column} FROM leads WHERE {column} IS NOT NULL AND {column} <> '' ORDER BY {column}"
    )]


def _like(term: str) -> str:
    """LIKE pattern matching ``term`` literally ("!" is the escape character)."""
    escaped = term.replace("!", "!!").replace("%", "!%").replace("_", "!_")
    return f"%{escaped}%"


SEARCH_COLUMNS = ("customer_name", "company", "phone", "phone_normalized", "email", "service_intent",
                  "service_interest", "requirement")


def search_leads(
    conn: sqlite3.Connection,
    text: str | None = None,
    filters: dict[str, list[str]] | None = None,
    created_from: str | None = None,
    created_to: str | None = None,
    limit: int = 500,
) -> list[dict]:
    """Search leads by free text and filter by column values and local creation date (YYYY-MM-DD)."""
    where, params = [], []
    if text and text.strip():
        clause = " OR ".join(f"{c} LIKE ? ESCAPE '!'" for c in SEARCH_COLUMNS)
        where.append(f"({clause})")
        params += [_like(text.strip())] * len(SEARCH_COLUMNS)
    for column, values in (filters or {}).items():
        if column not in FILTER_COLUMNS:
            raise ValueError(f"Not a filter column: {column}")
        if values:
            where.append(f"{column} IN ({', '.join('?' for _ in values)})")
            params += list(values)
    if created_from:
        where.append("date(first_contact_date, 'localtime') >= ?")
        params.append(created_from)
    if created_to:
        where.append("date(first_contact_date, 'localtime') <= ?")
        params.append(created_to)
    sql = "SELECT * FROM leads" + (" WHERE " + " AND ".join(where) if where else "")
    sql += " ORDER BY last_contact_date DESC LIMIT ?"
    return _rows(conn.execute(sql, (*params, limit)))


# ---------------------------------------------------------------- activities

def log_activity(conn: sqlite3.Connection, lead_id: int, activity_type: str, description: str,
                 now: str | None = None) -> int:
    cur = conn.execute(
        "INSERT INTO activities (lead_id, activity_type, description, created_at) VALUES (?, ?, ?, ?)",
        (lead_id, activity_type, description, now or utc_now()),
    )
    return cur.lastrowid


def list_activities(conn: sqlite3.Connection, lead_id: int, limit: int = 50) -> list[dict]:
    return _rows(conn.execute(
        "SELECT * FROM activities WHERE lead_id = ? ORDER BY created_at DESC, id DESC LIMIT ?", (lead_id, limit)
    ))
