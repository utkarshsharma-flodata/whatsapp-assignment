"""Print a quick overview of the leads database (read-only).

Run from the project root:
    python -m scripts.inspect_db            # table counts + recent leads
    python -m scripts.inspect_db 3          # full detail for lead #3
"""

import json
import sqlite3
import sys
from pathlib import Path

from core.config import get_settings

TABLES = ["leads", "conversations", "messages", "lead_scores", "lead_status_history", "follow_ups", "activities",
          "services"]


def main() -> int:
    path = get_settings().database_path
    if not Path(path).exists():
        print(f"No database at {path} yet. Start the app and send a message first.")
        return 1
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    print(f"Database: {path}  (schema version {conn.execute('PRAGMA user_version').fetchone()[0]})\n")

    if len(sys.argv) > 1:
        return show_lead(conn, int(sys.argv[1]))

    for table in TABLES:
        print(f"{table:<20}{conn.execute(f'SELECT COUNT(*) FROM {table}').fetchone()[0]:>6}")
    print("\nRecent leads:")
    for r in conn.execute(
        "SELECT id, customer_name, company, phone, email, service_intent, score, category, last_contact_date "
        "FROM leads ORDER BY last_contact_date DESC LIMIT 20"
    ):
        print(f"  #{r['id']:<4} {r['customer_name'] or '—':<18} {r['company'] or '—':<18} "
              f"{r['phone'] or r['email'] or '—':<22} {r['service_intent'] or '—':<22} "
              f"{r['score'] if r['score'] is not None else '—':>3} {r['category'] or ''}")
    return 0


def show_lead(conn: sqlite3.Connection, lead_id: int) -> int:
    lead = conn.execute("SELECT * FROM leads WHERE id = ?", (lead_id,)).fetchone()
    if lead is None:
        print(f"Lead #{lead_id} not found.")
        return 1
    for key in lead.keys():
        if key != "profile_json" and lead[key] is not None:
            print(f"  {key:<22} {lead[key]}")

    print("\nScore history:")
    for s in conn.execute("SELECT * FROM lead_scores WHERE lead_id = ? ORDER BY scored_at, id", (lead_id,)):
        parts = ", ".join(f"{c['label']} {c['earned']}/{c['maximum']}" for c in json.loads(s["breakdown_json"]))
        print(f"  {s['scored_at']}  {s['total_score']:>3} {s['category']:<4}  {parts}")

    print("\nConversations:")
    for c in conn.execute("SELECT * FROM conversations WHERE lead_id = ? ORDER BY started_at", (lead_id,)):
        print(f"  Conversation #{c['id']} ({c['channel']}, started {c['started_at']})")
        for m in conn.execute("SELECT * FROM messages WHERE conversation_id = ? ORDER BY sent_at, id", (c["id"],)):
            print(f"    [{m['sent_at']}] {m['sender_role']} ({m['processing_status']}): {m['message_text']}")

    print("\nActivities:")
    for a in conn.execute("SELECT * FROM activities WHERE lead_id = ? ORDER BY created_at, id", (lead_id,)):
        print(f"  {a['created_at']}  {a['activity_type']:<20} {a['description']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
