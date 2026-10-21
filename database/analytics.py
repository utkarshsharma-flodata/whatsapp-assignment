"""Read-only aggregate queries for the admin dashboard.

Every number on the dashboard comes from one of these functions. Dates are
grouped by the local calendar day (SQLite ``'localtime'``); ``today`` is a
local ``YYYY-MM-DD`` string supplied by the caller so results are testable.
"""

import sqlite3

# Lifecycle order; "lost" can follow any open stage, so it is not a funnel step.
FUNNEL_STAGES = ["new", "contacted", "qualified", "proposal", "negotiation", "converted"]
_STAGE_RANK = {stage: i for i, stage in enumerate(FUNNEL_STAGES)}

NOT_STATED = "Not stated"
_GROUPABLE = {"service_intent", "industry", "category", "lead_source", "status", "assigned_salesperson"}


def kpis(conn: sqlite3.Connection, today: str) -> dict:
    row = conn.execute(
        """
        SELECT COUNT(*)                                        AS total,
               SUM(status = 'new')                             AS new,
               SUM(category = 'HOT')                           AS hot,
               SUM(category = 'WARM')                          AS warm,
               SUM(category = 'COLD')                          AS cold,
               SUM(category IS NULL)                           AS unscored,
               SUM(status = 'converted')                       AS converted,
               SUM(status = 'lost')                            AS lost,
               AVG(score)                                      AS avg_score
        FROM leads
        """
    ).fetchone()
    follow = conn.execute(
        """
        SELECT SUM(status = 'pending')                   AS pending,
               SUM(status = 'pending' AND due_date < ?)  AS overdue
        FROM follow_ups
        """,
        (today,),
    ).fetchone()
    total = row["total"] or 0
    converted, lost = row["converted"] or 0, row["lost"] or 0
    return {
        "total": total,
        "new": row["new"] or 0,
        "hot": row["hot"] or 0,
        "warm": row["warm"] or 0,
        "cold": row["cold"] or 0,
        "unscored": row["unscored"] or 0,
        "converted": converted,
        "lost": lost,
        "pending_follow_ups": follow["pending"] or 0,
        "overdue_follow_ups": follow["overdue"] or 0,
        # Converted / all leads. None when there are no leads (not 0%).
        "conversion_rate": converted / total if total else None,
        # Converted / closed (converted + lost). None until a lead is closed.
        "win_rate": converted / (converted + lost) if converted + lost else None,
        "avg_score": round(row["avg_score"], 1) if row["avg_score"] is not None else None,
    }


def leads_by_date(conn: sqlite3.Connection) -> list[dict]:
    """New leads per local day of first contact."""
    return [dict(r) for r in conn.execute(
        """
        SELECT date(first_contact_date, 'localtime') AS day, COUNT(*) AS leads
        FROM leads GROUP BY day ORDER BY day
        """
    )]


def leads_by(conn: sqlite3.Connection, column: str) -> list[dict]:
    """Lead counts per value of ``column``; missing values are grouped as "Not stated"."""
    if column not in _GROUPABLE:
        raise ValueError(f"Cannot group by {column}")
    return [dict(r) for r in conn.execute(
        f"""
        SELECT COALESCE(NULLIF(TRIM({column}), ''), ?) AS value, COUNT(*) AS leads
        FROM leads GROUP BY value ORDER BY leads DESC, value
        """,
        (NOT_STATED,),
    )]


def funnel(conn: sqlite3.Connection) -> list[dict]:
    """Leads that ever reached each lifecycle stage (from status history and current status).

    A lead counts for a stage if it reached that stage or a later one, so
    the counts never increase down the funnel. Lost leads count up to the
    furthest stage they reached before being lost.
    """
    furthest: dict[int, int] = {}
    rows = conn.execute(
        "SELECT lead_id, to_status AS status FROM lead_status_history "
        "UNION ALL SELECT id, status FROM leads"
    )
    for r in rows:
        rank = _STAGE_RANK.get(r["status"])
        if rank is not None:
            furthest[r["lead_id"]] = max(furthest.get(r["lead_id"], 0), rank)
    # Leads with no history row at all still count as "new".
    for (lead_id,) in conn.execute("SELECT id FROM leads"):
        furthest.setdefault(lead_id, 0)
    return [
        {"stage": stage, "leads": sum(1 for rank in furthest.values() if rank >= i)}
        for i, stage in enumerate(FUNNEL_STAGES)
    ]


def source_performance(conn: sqlite3.Connection) -> list[dict]:
    return [dict(r) for r in conn.execute(
        """
        SELECT COALESCE(NULLIF(TRIM(lead_source), ''), ?) AS source,
               COUNT(*)                     AS leads,
               SUM(status = 'converted')    AS converted,
               SUM(status = 'lost')         AS lost,
               ROUND(AVG(score), 1)         AS avg_score
        FROM leads GROUP BY source ORDER BY leads DESC, source
        """,
        (NOT_STATED,),
    )]


def follow_up_performance(conn: sqlite3.Connection, today: str) -> dict:
    row = conn.execute(
        """
        SELECT SUM(status = 'pending' AND due_date >= ?)                                   AS upcoming,
               SUM(status = 'pending' AND due_date < ?)                                    AS overdue,
               SUM(status = 'done' AND date(completed_at, 'localtime') <= due_date)        AS done_on_time,
               SUM(status = 'done' AND date(completed_at, 'localtime') > due_date)         AS done_late,
               SUM(status = 'missed')                                                      AS missed,
               SUM(status = 'cancelled')                                                   AS cancelled,
               COUNT(*)                                                                    AS total
        FROM follow_ups
        """,
        (today, today),
    ).fetchone()
    return {k: row[k] or 0 for k in row.keys()}


def conversion_trend(conn: sqlite3.Connection) -> list[dict]:
    """Leads closed per local day, split into converted and lost."""
    return [dict(r) for r in conn.execute(
        """
        SELECT date(closed_at, 'localtime') AS day,
               SUM(status = 'converted') AS converted,
               SUM(status = 'lost')      AS lost
        FROM leads WHERE closed_at IS NOT NULL AND status IN ('converted', 'lost')
        GROUP BY day ORDER BY day
        """
    )]


def data_gaps(conn: sqlite3.Connection) -> dict:
    """How many leads are missing key qualification fields."""
    row = conn.execute(
        """
        SELECT COUNT(*) AS total,
               SUM(budget_min IS NULL AND budget_max IS NULL AND budget_text IS NULL) AS no_budget,
               SUM(timeline IS NULL)                                                   AS no_timeline,
               SUM(company IS NULL)                                                    AS no_company,
               SUM(phone IS NULL AND email IS NULL)                                    AS no_contact
        FROM leads
        """
    ).fetchone()
    return {k: row[k] or 0 for k in row.keys()}


def attention_counts(conn: sqlite3.Connection) -> dict:
    row = conn.execute(
        """
        SELECT SUM(category = 'HOT' AND assigned_salesperson IS NULL
                   AND status NOT IN ('converted', 'lost'))            AS hot_unassigned,
               SUM(category = 'HOT' AND status = 'new')                AS hot_still_new
        FROM leads
        """
    ).fetchone()
    return {k: row[k] or 0 for k in row.keys()}
