"""Database schema and versioned migrations.

The schema version is stored in ``PRAGMA user_version``. Each entry in
``MIGRATIONS`` upgrades the database by one version and runs in a single
transaction. To change the schema, append a new migration; never edit an
existing one.
"""

import logging
import sqlite3
from datetime import datetime, timezone

log = logging.getLogger("leadbot.schema")

SERVICE_NAMES = [
    "Website Development",
    "Software Development",
    "Power BI Dashboard",
    "Data Analytics",
    "Business Automation",
    "Digital Marketing",
    "Graphic Design",
    "Cybersecurity",
    "SaaS Development",
]

_V1 = """
CREATE TABLE services (
    id          INTEGER PRIMARY KEY,
    name        TEXT    NOT NULL UNIQUE,
    is_active   INTEGER NOT NULL DEFAULT 1 CHECK (is_active IN (0, 1)),
    created_at  TEXT    NOT NULL,
    updated_at  TEXT    NOT NULL
);

CREATE TABLE leads (
    id                    INTEGER PRIMARY KEY,
    customer_name         TEXT,
    phone                 TEXT,
    phone_normalized      TEXT,
    email                 TEXT,
    email_normalized      TEXT,
    company               TEXT,
    location              TEXT,
    industry              TEXT,
    business_size         TEXT,
    requirement           TEXT,
    requested_modules     TEXT    NOT NULL DEFAULT '[]' CHECK (json_valid(requested_modules)),
    service_interest      TEXT,
    service_id            INTEGER REFERENCES services(id) ON DELETE SET NULL,
    budget_min            REAL,
    budget_max            REAL,
    budget_text           TEXT,
    currency              TEXT,
    timeline              TEXT,
    service_intent        TEXT,
    purchase_intent       TEXT    CHECK (purchase_intent IN ('high', 'medium', 'low', 'unknown')),
    decision_authority    TEXT    CHECK (decision_authority IN ('yes', 'no', 'unknown')),
    lead_source           TEXT,
    score                 INTEGER CHECK (score BETWEEN 0 AND 100),
    category              TEXT    CHECK (category IN ('HOT', 'WARM', 'COLD')),
    status                TEXT    NOT NULL DEFAULT 'new'
                                  CHECK (status IN ('new', 'contacted', 'qualified', 'proposal', 'negotiation', 'closed')),
    assigned_salesperson  TEXT,
    first_contact_date    TEXT    NOT NULL,
    last_contact_date     TEXT    NOT NULL,
    next_follow_up_date   TEXT,
    conversion_status     TEXT    NOT NULL DEFAULT 'open' CHECK (conversion_status IN ('open', 'won', 'lost')),
    profile_json          TEXT    CHECK (profile_json IS NULL OR json_valid(profile_json)),
    created_at            TEXT    NOT NULL,
    updated_at            TEXT    NOT NULL
);
-- A phone number identifies at most one lead.
CREATE UNIQUE INDEX ux_leads_phone ON leads(phone_normalized) WHERE phone_normalized IS NOT NULL;
CREATE INDEX ix_leads_email ON leads(email_normalized) WHERE email_normalized IS NOT NULL;
CREATE INDEX ix_leads_category ON leads(category);
CREATE INDEX ix_leads_status ON leads(status);
CREATE INDEX ix_leads_last_contact ON leads(last_contact_date);
CREATE INDEX ix_leads_next_follow_up ON leads(next_follow_up_date);

CREATE TABLE conversations (
    id                  INTEGER PRIMARY KEY,
    lead_id             INTEGER REFERENCES leads(id) ON DELETE SET NULL,
    channel             TEXT    NOT NULL DEFAULT 'simulator' CHECK (channel IN ('simulator', 'whatsapp')),
    session_identifier  TEXT    NOT NULL UNIQUE,
    sender_phone        TEXT,
    analysis_json       TEXT    CHECK (analysis_json IS NULL OR json_valid(analysis_json)),
    started_at          TEXT    NOT NULL,
    last_message_at     TEXT    NOT NULL,
    created_at          TEXT    NOT NULL,
    updated_at          TEXT    NOT NULL
);
CREATE INDEX ix_conversations_lead ON conversations(lead_id);
CREATE INDEX ix_conversations_last_message ON conversations(last_message_at);

CREATE TABLE messages (
    id                 INTEGER PRIMARY KEY,
    conversation_id    INTEGER NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    sender_role        TEXT    NOT NULL CHECK (sender_role IN ('customer', 'assistant', 'agent', 'system')),
    message_text       TEXT    NOT NULL,
    sent_at            TEXT    NOT NULL,
    processing_status  TEXT    NOT NULL DEFAULT 'pending'
                               CHECK (processing_status IN ('pending', 'processed', 'failed', 'skipped')),
    processing_error   TEXT
);
CREATE INDEX ix_messages_conversation ON messages(conversation_id, sent_at);
CREATE INDEX ix_messages_status ON messages(processing_status);

CREATE TABLE lead_scores (
    id                    INTEGER PRIMARY KEY,
    lead_id               INTEGER NOT NULL REFERENCES leads(id) ON DELETE CASCADE,
    conversation_id       INTEGER REFERENCES conversations(id) ON DELETE SET NULL,
    total_score           INTEGER NOT NULL CHECK (total_score BETWEEN 0 AND 100),
    category              TEXT    NOT NULL CHECK (category IN ('HOT', 'WARM', 'COLD')),
    breakdown_json        TEXT    NOT NULL CHECK (json_valid(breakdown_json)),
    missing_json          TEXT    NOT NULL DEFAULT '[]' CHECK (json_valid(missing_json)),
    explanation           TEXT    NOT NULL,
    scored_at             TEXT    NOT NULL
);
CREATE INDEX ix_lead_scores_lead ON lead_scores(lead_id, scored_at);

CREATE TABLE follow_ups (
    id                 INTEGER PRIMARY KEY,
    lead_id            INTEGER NOT NULL REFERENCES leads(id) ON DELETE CASCADE,
    due_date           TEXT    NOT NULL,
    status             TEXT    NOT NULL DEFAULT 'pending'
                               CHECK (status IN ('pending', 'done', 'cancelled', 'missed')),
    follow_up_count    INTEGER NOT NULL DEFAULT 0 CHECK (follow_up_count >= 0),
    customer_response  TEXT,
    next_action        TEXT,
    created_at         TEXT    NOT NULL,
    updated_at         TEXT    NOT NULL
);
CREATE INDEX ix_follow_ups_lead ON follow_ups(lead_id);
CREATE INDEX ix_follow_ups_due ON follow_ups(status, due_date);

CREATE TABLE activities (
    id             INTEGER PRIMARY KEY,
    lead_id        INTEGER NOT NULL REFERENCES leads(id) ON DELETE CASCADE,
    activity_type  TEXT    NOT NULL,
    description    TEXT    NOT NULL,
    created_at     TEXT    NOT NULL
);
CREATE INDEX ix_activities_lead ON activities(lead_id, created_at);
"""


def _seed_services(conn: sqlite3.Connection) -> None:
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    conn.executemany(
        "INSERT OR IGNORE INTO services (name, is_active, created_at, updated_at) VALUES (?, 1, ?, ?)",
        [(name, now, now) for name in SERVICE_NAMES],
    )


_V2 = """
-- Follow-up topics an assistant reply asked about (JSON list), used to avoid repeating questions.
ALTER TABLE messages ADD COLUMN asked_topics TEXT CHECK (asked_topics IS NULL OR json_valid(asked_topics));
"""

_LEAD_COPY_COLUMNS = (
    "id, customer_name, phone, phone_normalized, email, email_normalized, company, location, industry, "
    "business_size, requirement, requested_modules, service_interest, service_id, budget_min, budget_max, "
    "budget_text, currency, timeline, service_intent, purchase_intent, decision_authority, lead_source, score, "
    "category, assigned_salesperson, first_contact_date, last_contact_date, next_follow_up_date, "
    "conversion_status, profile_json, created_at, updated_at"
)

# v3: sales lifecycle New -> Contacted -> Qualified -> Proposal -> Negotiation -> Converted/Lost.
# SQLite cannot alter a CHECK constraint, so the leads table is rebuilt (foreign keys are
# switched off for this migration; see REBUILD_VERSIONS).
_V3 = f"""
CREATE TABLE leads_v3 (
    id                    INTEGER PRIMARY KEY,
    customer_name         TEXT,
    phone                 TEXT,
    phone_normalized      TEXT,
    email                 TEXT,
    email_normalized      TEXT,
    company               TEXT,
    location              TEXT,
    industry              TEXT,
    business_size         TEXT,
    requirement           TEXT,
    requested_modules     TEXT    NOT NULL DEFAULT '[]' CHECK (json_valid(requested_modules)),
    service_interest      TEXT,
    service_id            INTEGER REFERENCES services(id) ON DELETE SET NULL,
    budget_min            REAL,
    budget_max            REAL,
    budget_text           TEXT,
    currency              TEXT,
    timeline              TEXT,
    service_intent        TEXT,
    purchase_intent       TEXT    CHECK (purchase_intent IN ('high', 'medium', 'low', 'unknown')),
    decision_authority    TEXT    CHECK (decision_authority IN ('yes', 'no', 'unknown')),
    lead_source           TEXT,
    score                 INTEGER CHECK (score BETWEEN 0 AND 100),
    category              TEXT    CHECK (category IN ('HOT', 'WARM', 'COLD')),
    status                TEXT    NOT NULL DEFAULT 'new'
                                  CHECK (status IN ('new', 'contacted', 'qualified', 'proposal', 'negotiation',
                                                    'converted', 'lost')),
    assigned_salesperson  TEXT,
    first_contact_date    TEXT    NOT NULL,
    last_contact_date     TEXT    NOT NULL,
    next_follow_up_date   TEXT,
    conversion_status     TEXT    NOT NULL DEFAULT 'open' CHECK (conversion_status IN ('open', 'won', 'lost')),
    profile_json          TEXT    CHECK (profile_json IS NULL OR json_valid(profile_json)),
    status_changed_at     TEXT,
    closed_at             TEXT,
    created_at            TEXT    NOT NULL,
    updated_at            TEXT    NOT NULL
);
INSERT INTO leads_v3 ({_LEAD_COPY_COLUMNS}, status, status_changed_at, closed_at)
SELECT {_LEAD_COPY_COLUMNS},
       CASE WHEN status <> 'closed' THEN status
            WHEN conversion_status = 'won' THEN 'converted'
            WHEN conversion_status = 'lost' THEN 'lost'
            ELSE 'negotiation' END,
       created_at,
       CASE WHEN conversion_status IN ('won', 'lost') THEN updated_at END
FROM leads;
DROP TABLE leads;
ALTER TABLE leads_v3 RENAME TO leads;
CREATE UNIQUE INDEX ux_leads_phone ON leads(phone_normalized) WHERE phone_normalized IS NOT NULL;
CREATE INDEX ix_leads_email ON leads(email_normalized) WHERE email_normalized IS NOT NULL;
CREATE INDEX ix_leads_category ON leads(category);
CREATE INDEX ix_leads_status ON leads(status);
CREATE INDEX ix_leads_last_contact ON leads(last_contact_date);
CREATE INDEX ix_leads_next_follow_up ON leads(next_follow_up_date);
CREATE INDEX ix_leads_first_contact ON leads(first_contact_date);
CREATE INDEX ix_leads_assigned ON leads(assigned_salesperson);

CREATE TABLE lead_status_history (
    id           INTEGER PRIMARY KEY,
    lead_id      INTEGER NOT NULL REFERENCES leads(id) ON DELETE CASCADE,
    from_status  TEXT,
    to_status    TEXT    NOT NULL,
    note         TEXT,
    changed_at   TEXT    NOT NULL
);
CREATE INDEX ix_status_history_lead ON lead_status_history(lead_id, changed_at);
INSERT INTO lead_status_history (lead_id, from_status, to_status, changed_at)
SELECT id, NULL, status, created_at FROM leads;

ALTER TABLE follow_ups ADD COLUMN completed_at TEXT;
"""

# (version, DDL script, optional Python step run in the same transaction)
MIGRATIONS = [
    (1, _V1, _seed_services),
    (2, _V2, None),
    (3, _V3, None),
]
SCHEMA_VERSION = MIGRATIONS[-1][0]
# Migrations that rebuild a referenced table must run with foreign keys off,
# otherwise DROP TABLE would cascade to child rows.
REBUILD_VERSIONS = {3}


def get_version(conn: sqlite3.Connection) -> int:
    return conn.execute("PRAGMA user_version").fetchone()[0]


def migrate(conn: sqlite3.Connection) -> int:
    """Apply pending migrations. Returns the resulting schema version."""
    current = get_version(conn)
    if current > SCHEMA_VERSION:
        raise RuntimeError(
            f"Database schema version {current} is newer than this app supports ({SCHEMA_VERSION})."
        )
    for version, script, step in MIGRATIONS:
        if version <= current:
            continue
        log.info("Applying database migration %d", version)
        rebuild = version in REBUILD_VERSIONS
        fks_were_on = conn.execute("PRAGMA foreign_keys").fetchone()[0]
        if rebuild:
            conn.execute("PRAGMA foreign_keys = OFF")  # has no effect inside a transaction
        try:
            conn.execute("BEGIN")
            try:
                for statement in _split(script):
                    conn.execute(statement)
                if step:
                    step(conn)
                if rebuild and conn.execute("PRAGMA foreign_key_check").fetchall():
                    raise RuntimeError(f"Migration {version} left foreign-key violations")
                conn.execute(f"PRAGMA user_version = {int(version)}")
                conn.execute("COMMIT")
            except Exception:
                conn.execute("ROLLBACK")
                raise
        finally:
            if rebuild and fks_were_on:
                conn.execute("PRAGMA foreign_keys = ON")
    return get_version(conn)


def _split(script: str) -> list[str]:
    """Split a DDL script into statements (the DDL has no semicolons in literals)."""
    lines = [line for line in script.splitlines() if not line.strip().startswith("--")]
    return [s.strip() for s in "\n".join(lines).split(";") if s.strip()]
