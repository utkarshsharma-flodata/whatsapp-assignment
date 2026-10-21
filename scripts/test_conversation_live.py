"""Live multi-turn conversation against the Groq model configured in .env.

Uses a temporary database, so your real CRM data is not touched. Ends with
checks for duplicates, score history, dashboard figures and reopening. Prints each
assistant reply, the follow-up topics chosen, the score, and any errors.
Uses your Groq quota: two to four requests per message.

Run from the project root:  python -m scripts.test_conversation_live
"""

import dataclasses
import sys
import tempfile
from datetime import date
from pathlib import Path

from core.config import get_settings
from database import repository as repo
from database.connection import connect
from services import conversation_service as cs
from services import lead_service as ls
from services.analytics_service import build_dashboard

SCRIPT = [
    "Hi, how much does a website cost?",
    "We need an e-commerce site for our bakery with online ordering and payments.",
    "Budget is around 1.5 lakh rupees, and we'd like it live within 2 months.",
    "Bloom Bakery",
    "Actually, we need it sooner, within 1 month if possible.",
]
RETURNING = "Hi, it's me again. Can you also add a blog to the website?"


def run_turn(conn, conversation_id: int, settings, text: str) -> bool:
    print(f"\nCustomer: {text}")
    ls.record_customer_message(conn, conversation_id, text)
    result = cs.process_turn(conn, conversation_id, settings)
    if result.outcome:
        print(f"  [{result.outcome.summary()} | score {result.outcome.score.total} {result.outcome.score.category}]")
    if result.error:
        print(f"  ERROR ({result.failed_stage}): {result.error}")
        return False
    print(f"  [asked about: {', '.join(result.reply.asked_topics) or 'nothing'}"
          f"{' | regenerated after safety check' if result.reply.regenerated else ''}]")
    print(f"Assistant: {result.reply.text}")
    return True


def main() -> int:
    tmp = tempfile.TemporaryDirectory()
    settings = dataclasses.replace(get_settings(), database_path=str(Path(tmp.name) / "live.db"))
    ok = True
    with connect(settings.database_path) as conn:
        conversation_id = ls.start_conversation(conn, sender_phone="+91 98765 43210")
        replies = 0
        for text in SCRIPT:
            turn_ok = run_turn(conn, conversation_id, settings, text)
            replies += turn_ok
            ok &= turn_ok

        print("\n--- Returning customer: new conversation with the same number ---")
        second = ls.start_conversation(conn, sender_phone="09876543210")
        ok &= run_turn(conn, second, settings, RETURNING)
        lead = repo.list_leads(conn)[0]
        print(f"\nLeads in database: {len(repo.list_leads(conn))} | lead #{lead['id']}: "
              f"{lead['company']}, budget {lead['budget_text']}, timeline {lead['timeline']}, "
              f"score {lead['score']} {lead['category']}")
        history = repo.score_history(conn, lead["id"])
        print(f"Score history: {[h['total_score'] for h in history]}")

        checks = {
            "one lead (no duplicates)": len(repo.list_leads(conn)) == 1,
            "lead score equals latest history row": lead["score"] == history[-1]["total_score"],
        }
        dashboard = build_dashboard(conn, date.today().isoformat())
        checks["dashboard total = 1"] = dashboard.kpis["total"] == 1
        checks["dashboard average = lead score"] = dashboard.kpis["avg_score"] == float(lead["score"])
        reopened = ls.load_conversation(conn, conversation_id)
        roles = [m["sender_role"] for m in reopened["messages"]]
        # Every customer message is stored; an assistant message exists only for replies that succeeded.
        checks["reopened conversation has all messages"] = (roles.count("customer"), roles.count("assistant")) \
            == (len(SCRIPT), replies)
        checks["reopened profile kept the company"] = reopened["profile"].entities.company_name.value is not None
        print("\nChecks:")
        for name, passed in checks.items():
            print(f"  {'PASS' if passed else 'FAIL'}  {name}")
        ok &= all(checks.values())
    tmp.cleanup()
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
