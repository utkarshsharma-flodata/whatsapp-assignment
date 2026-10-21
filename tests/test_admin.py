"""Tests for the admin CRM: migration v3, lead management, KPIs, analytics, insights.

Leads are created through the real chat pipeline (store message -> analyse
with a fake Groq client -> persist), never inserted as fake rows.

Run:  python -m unittest discover tests
"""

import sqlite3
import tempfile
import unittest
from pathlib import Path

from database import analytics
from database import repository as repo
from database.connection import open_connection
from database.schema import MIGRATIONS, SCHEMA_VERSION, _split
from services import admin_service as admin
from services import conversation_service as cs
from services import lead_service as ls
from services.analytics_service import build_dashboard
from tests.test_conversation import ScriptedClient
from tests.test_nlp_service import SETTINGS, extraction

TODAY = "2026-10-21"


def tf(value, evidence=None, confidence=0.95, **kw):
    return {"value": value, "evidence": evidence or value, "confidence": confidence, **kw}


class AdminTestCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.path = str(Path(self._tmp.name) / "leads.db")
        self.conn = open_connection(self.path)
        self.client = ScriptedClient()

    def tearDown(self):
        self.conn.close()
        self._tmp.cleanup()

    def chat_lead(self, text, category="Website Development", level="medium", phone=None, **entities) -> int:
        """Create a lead the way the chat UI does: message -> analysis -> persist -> reply."""
        conversation_id = ls.start_conversation(self.conn, sender_phone=phone)
        ls.record_customer_message(self.conn, conversation_id, text)
        self.client.extractions.append(extraction(category, level, **entities))
        self.client.replies.append("Thanks! Could you tell me more?")
        result = cs.process_turn(self.conn, conversation_id, SETTINGS, client=self.client)
        self.assertTrue(result.ok, result.error)
        return result.outcome.lead_id

    def hot_lead(self, name="Priya", company="Bloom Bakery", phone="9876543210", source=None, industry="Bakery"):
        entities = dict(
            customer_name=tf(name, f"I'm {name}"), company_name=tf(company),
            industry=tf(industry), location=tf("Pune"), business_size=tf("4 outlets"),
            interested_service=tf("e-commerce website"), requirement_description=tf("e-commerce website"),
            requested_features={"values": ["ordering", "payments", "tracking"], "confidence": 0.95},
            budget={"amount_min": 150000, "amount_max": 150000, "currency": "INR", "raw_text": "1.5 lakh",
                    "confidence": 0.95},
            timeline=tf("2 weeks"), decision_authority={"value": "yes", "confidence": 0.95, "evidence": "I decide"},
            questions={"values": ["Can you send a quote?"], "confidence": 0.9},
        )
        if source:
            entities["lead_source"] = tf(source)
        text = (f"I'm {name} from {company}, a {industry} in Pune with 4 outlets. Need an e-commerce website "
                f"with ordering, payments and tracking. Budget 1.5 lakh, within 2 weeks. I decide. "
                f"Can you send a quote? Found you via {source or 'a friend'}.")
        return self.chat_lead(text, level="high", phone=phone, **entities)


class MigrationTests(AdminTestCase):
    def test_upgrade_from_v2_rebuilds_leads_without_losing_children(self):
        path = str(Path(self._tmp.name) / "v2.db")
        old = sqlite3.connect(path, autocommit=True)
        old.execute("PRAGMA foreign_keys = ON")
        old.execute("BEGIN")
        for version, script, step in MIGRATIONS[:2]:
            for statement in _split(script):
                old.execute(statement)
        old.execute("PRAGMA user_version = 2")
        old.execute("COMMIT")
        now = repo.utc_now()
        old.execute("INSERT INTO leads (customer_name, status, conversion_status, first_contact_date, "
                    "last_contact_date, created_at, updated_at) VALUES ('Old', 'closed', 'won', ?, ?, ?, ?)",
                    (now, now, now, now))
        old.execute("INSERT INTO conversations (lead_id, session_identifier, started_at, last_message_at, "
                    "created_at, updated_at) VALUES (1, 's', ?, ?, ?, ?)", (now, now, now, now))
        old.execute("INSERT INTO messages (conversation_id, sender_role, message_text, sent_at) "
                    "VALUES (1, 'customer', 'hello', ?)", (now,))
        old.execute("INSERT INTO lead_scores (lead_id, total_score, category, breakdown_json, explanation, "
                    "scored_at) VALUES (1, 50, 'WARM', '[]', 'x', ?)", (now,))
        old.close()

        conn = open_connection(path)
        self.assertEqual(conn.execute("PRAGMA user_version").fetchone()[0], SCHEMA_VERSION)
        self.assertEqual(conn.execute("PRAGMA foreign_keys").fetchone()[0], 1)
        lead = repo.get_lead(conn, 1)
        self.assertEqual((lead["status"], lead["closed_at"]), ("converted", now))
        self.assertEqual(len(repo.get_messages(conn, 1)), 1)
        self.assertEqual(len(repo.score_history(conn, 1)), 1)
        self.assertEqual(repo.get_conversation(conn, 1)["lead_id"], 1)
        self.assertEqual(repo.list_status_history(conn, 1)[0]["to_status"], "converted")
        self.assertEqual(conn.execute("PRAGMA foreign_key_check").fetchall(), [])
        with self.assertRaises(sqlite3.IntegrityError):  # FK enforcement is back on
            conn.execute("INSERT INTO lead_scores (lead_id, total_score, category, breakdown_json, explanation, "
                         "scored_at) VALUES (999, 1, 'COLD', '[]', 'x', ?)", (now,))
        conn.close()

    def test_new_statuses_allowed_and_invalid_rejected(self):
        lead_id = self.chat_lead("Need a website")
        admin.change_status(self.conn, lead_id, "converted")
        with self.assertRaises(sqlite3.IntegrityError):
            self.conn.execute("UPDATE leads SET status = 'closed' WHERE id = ?", (lead_id,))


class EmptyDatabaseTests(AdminTestCase):
    def test_dashboard_on_empty_database(self):
        d = build_dashboard(self.conn, TODAY)
        k = d.kpis
        self.assertEqual(k["total"], 0)
        self.assertEqual((k["hot"], k["warm"], k["cold"], k["converted"], k["lost"], k["pending_follow_ups"]),
                         (0, 0, 0, 0, 0, 0))
        self.assertIsNone(k["conversion_rate"])
        self.assertIsNone(k["avg_score"])
        self.assertEqual(d.by_date, [])
        self.assertEqual(d.by_service, [])
        self.assertEqual(d.conversion_trend, [])
        self.assertEqual([f["leads"] for f in d.funnel], [0] * 6)
        self.assertEqual(d.insights, [])
        self.assertEqual(repo.search_leads(self.conn, "anything"), [])


class LeadManagementTests(AdminTestCase):
    def test_status_lifecycle_and_history(self):
        lead_id = self.chat_lead("Need a website")
        for status in ("contacted", "qualified", "proposal", "negotiation"):
            self.assertTrue(admin.change_status(self.conn, lead_id, status))
        self.assertFalse(admin.change_status(self.conn, lead_id, "negotiation"))
        admin.mark_converted(self.conn, lead_id, "Signed contract")
        lead = repo.get_lead(self.conn, lead_id)
        self.assertEqual((lead["status"], lead["conversion_status"]), ("converted", "won"))
        self.assertIsNotNone(lead["closed_at"])
        history = [h["to_status"] for h in repo.list_status_history(self.conn, lead_id)]
        self.assertEqual(history, ["contacted", "qualified", "proposal", "negotiation", "converted"])
        self.assertEqual(repo.list_status_history(self.conn, lead_id)[-1]["note"], "Signed contract")

        admin.change_status(self.conn, lead_id, "negotiation")  # reopen
        lead = repo.get_lead(self.conn, lead_id)
        self.assertEqual((lead["conversion_status"], lead["closed_at"]), ("open", None))
        self.assertIn("reopened", [a["activity_type"] for a in repo.list_activities(self.conn, lead_id)])
        with self.assertRaises(ValueError):
            admin.change_status(self.conn, lead_id, "closed")

    def test_mark_lost_with_reason(self):
        lead_id = self.chat_lead("Need a website")
        admin.mark_lost(self.conn, lead_id, "Chose a competitor")
        lead = repo.get_lead(self.conn, lead_id)
        self.assertEqual((lead["status"], lead["conversion_status"]), ("lost", "lost"))
        self.assertTrue(any("Chose a competitor" in a["description"] for a in repo.list_activities(self.conn, lead_id)))

    def test_salesperson_notes_and_follow_ups(self):
        lead_id = self.chat_lead("Need a website")
        self.assertTrue(admin.assign_salesperson(self.conn, lead_id, "  Ravi "))
        self.assertFalse(admin.assign_salesperson(self.conn, lead_id, "Ravi"))
        self.assertEqual(repo.get_lead(self.conn, lead_id)["assigned_salesperson"], "Ravi")
        admin.add_note(self.conn, lead_id, "Prefers calls after 5pm")
        with self.assertRaises(ValueError):
            admin.add_note(self.conn, lead_id, "   ")

        later = admin.schedule_follow_up(self.conn, lead_id, "2026-11-05", "Send proposal")
        sooner = admin.schedule_follow_up(self.conn, lead_id, "2026-10-25", "Confirm budget")
        self.assertEqual(repo.get_lead(self.conn, lead_id)["next_follow_up_date"], "2026-10-25")
        admin.complete_follow_up(self.conn, sooner, "done", "Budget confirmed")
        self.assertEqual(repo.get_lead(self.conn, lead_id)["next_follow_up_date"], "2026-11-05")
        admin.complete_follow_up(self.conn, later, "missed")
        self.assertIsNone(repo.get_lead(self.conn, lead_id)["next_follow_up_date"])
        follow_ups = repo.list_follow_ups(self.conn, lead_id)
        self.assertEqual([(f["follow_up_count"], f["status"]) for f in follow_ups], [(2, "done"), (1, "missed")])
        self.assertEqual(follow_ups[0]["customer_response"], "Budget confirmed")
        self.assertIsNotNone(follow_ups[0]["completed_at"])

        self.assertTrue(admin.assign_salesperson(self.conn, lead_id, ""))
        self.assertIsNone(repo.get_lead(self.conn, lead_id)["assigned_salesperson"])
        types = [a["activity_type"] for a in repo.list_activities(self.conn, lead_id)]
        for expected in ("assigned", "unassigned", "note", "follow_up_scheduled", "follow_up_done", "follow_up_missed"):
            self.assertIn(expected, types)

    def test_admin_edits_keep_history_and_survive_new_messages(self):
        lead_id = self.hot_lead()
        conversation_id = repo.list_conversations(self.conn, lead_id)[0]["id"]
        messages_before = repo.get_messages(self.conn, conversation_id)
        scores_before = repo.score_history(self.conn, lead_id)
        profile_before = repo.get_lead(self.conn, lead_id)["profile_json"]

        admin.change_status(self.conn, lead_id, "qualified")
        admin.assign_salesperson(self.conn, lead_id, "Ravi")
        admin.schedule_follow_up(self.conn, lead_id, "2026-11-01", "Call")
        self.assertEqual(repo.get_messages(self.conn, conversation_id), messages_before)
        self.assertEqual(repo.score_history(self.conn, lead_id), scores_before)
        self.assertEqual(repo.get_lead(self.conn, lead_id)["profile_json"], profile_before)

        # A new customer message must not reset the admin-managed fields.
        ls.record_customer_message(self.conn, conversation_id, "Any update on the quote?")
        self.client.extractions.append(extraction("Website Development", "high"))
        self.client.replies.append("The team is preparing it.")
        cs.process_turn(self.conn, conversation_id, SETTINGS, client=self.client)
        lead = repo.get_lead(self.conn, lead_id)
        self.assertEqual((lead["status"], lead["assigned_salesperson"], lead["next_follow_up_date"]),
                         ("qualified", "Ravi", "2026-11-01"))
        self.assertEqual(len(repo.get_messages(self.conn, conversation_id)), len(messages_before) + 2)

    def test_edits_persist_after_restart(self):
        lead_id = self.chat_lead("Need a website")
        admin.change_status(self.conn, lead_id, "proposal")
        admin.assign_salesperson(self.conn, lead_id, "Ravi")
        admin.add_note(self.conn, lead_id, "Sent proposal v1")
        admin.schedule_follow_up(self.conn, lead_id, "2026-11-01")
        self.conn.close()
        self.conn = open_connection(self.path)  # simulated restart
        detail = admin.lead_detail(self.conn, lead_id)
        self.assertEqual(detail["lead"]["status"], "proposal")
        self.assertEqual(detail["lead"]["assigned_salesperson"], "Ravi")
        self.assertEqual(detail["lead"]["next_follow_up_date"], "2026-11-01")
        self.assertTrue(any(a["description"] == "Sent proposal v1" for a in detail["activities"]))
        self.assertEqual(len(detail["conversations"][0]["messages"]), 2)
        self.assertIsNotNone(detail["score"])


class SearchTests(AdminTestCase):
    def setUp(self):
        super().setUp()
        self.priya = self.hot_lead()
        self.cold = self.chat_lead("Hi, what are your prices for SEO?", category="Pricing Enquiry", level="low",
                                   questions={"values": ["what are your prices?"], "confidence": 0.9})

    def test_text_search(self):
        for term in ("priya", "Bloom", "98765", "e-commerce", "Website"):
            self.assertEqual([l["id"] for l in repo.search_leads(self.conn, term)], [self.priya], term)
        self.assertEqual(repo.search_leads(self.conn, "100%"), [])  # wildcards are literal
        self.assertEqual(repo.search_leads(self.conn, "_"), [])

    def test_filters(self):
        ids = lambda **f: {l["id"] for l in repo.search_leads(self.conn, filters=f)}
        self.assertEqual(ids(category=["HOT"]), {self.priya})
        self.assertEqual(ids(category=["COLD"]), {self.cold})
        self.assertEqual(ids(service_intent=["Pricing Enquiry"]), {self.cold})
        self.assertEqual(ids(industry=["Bakery"]), {self.priya})
        admin.change_status(self.conn, self.cold, "lost")
        self.assertEqual(ids(status=["lost"]), {self.cold})
        self.assertEqual(ids(status=["new"], category=["HOT"]), {self.priya})
        today = repo.search_leads(self.conn)[0]["first_contact_date"][:10]
        self.assertEqual(len(repo.search_leads(self.conn, created_from="2000-01-01", created_to="2999-12-31")), 2)
        self.assertEqual(repo.search_leads(self.conn, created_to="2000-01-01"), [])
        self.assertTrue(today)
        with self.assertRaises(ValueError):
            repo.search_leads(self.conn, filters={"profile_json": ["x"]})


class DashboardTests(AdminTestCase):
    def test_kpis_and_charts_reflect_database(self):
        hot = self.hot_lead(source="Instagram ad")
        hot2 = self.hot_lead(name="Arjun", company="Nexa Retail", phone="9123456780", source="Instagram ad",
                             industry="Retail")
        cold = self.chat_lead("What are your prices?", category="Pricing Enquiry", level="low")
        warmish = self.chat_lead("Need a Power BI dashboard for sales KPIs, budget 50k INR",
                                 category="Power BI Dashboard", level="medium",
                                 interested_service=tf("Power BI dashboard"),
                                 requirement_description=tf("Power BI dashboard for sales KPIs"),
                                 requested_features={"values": ["sales KPIs"], "confidence": 0.9},
                                 budget={"amount_min": 50000, "amount_max": 50000, "currency": "INR",
                                         "raw_text": "50k INR", "confidence": 0.95})
        admin.change_status(self.conn, hot, "contacted")
        admin.change_status(self.conn, hot, "qualified")
        admin.mark_converted(self.conn, hot)
        admin.mark_lost(self.conn, cold)
        admin.schedule_follow_up(self.conn, hot2, "2026-10-20", "Overdue call")   # before TODAY
        admin.schedule_follow_up(self.conn, warmish, "2026-10-30", "Upcoming call")

        d = build_dashboard(self.conn, TODAY)
        k = d.kpis
        db_scores = [r["score"] for r in self.conn.execute("SELECT score FROM leads")]
        db_categories = [r["category"] for r in self.conn.execute("SELECT category FROM leads")]
        self.assertEqual(k["total"], 4)
        self.assertEqual(k["new"], 2)  # hot2 and warmish were never moved
        self.assertEqual((k["hot"], k["warm"], k["cold"]),
                         tuple(db_categories.count(c) for c in ("HOT", "WARM", "COLD")))
        self.assertEqual((k["converted"], k["lost"]), (1, 1))
        self.assertEqual((k["pending_follow_ups"], k["overdue_follow_ups"]), (2, 1))
        self.assertAlmostEqual(k["conversion_rate"], 1 / 4)
        self.assertAlmostEqual(k["win_rate"], 1 / 2)
        self.assertEqual(k["avg_score"], round(sum(db_scores) / len(db_scores), 1))

        self.assertEqual(sum(r["leads"] for r in d.by_date), 4)
        services = {r["value"]: r["leads"] for r in d.by_service}
        self.assertEqual(services, {"Website Development": 2, "Pricing Enquiry": 1, "Power BI Dashboard": 1})
        industries = {r["value"]: r["leads"] for r in d.by_industry}
        self.assertEqual(industries, {"Bakery": 1, "Retail": 1, analytics.NOT_STATED: 2})
        funnel = {r["stage"]: r["leads"] for r in d.funnel}
        self.assertEqual(funnel, {"new": 4, "contacted": 1, "qualified": 1, "proposal": 1, "negotiation": 1,
                                  "converted": 1})
        sources = {r["source"]: (r["leads"], r["converted"]) for r in d.sources}
        self.assertEqual(sources["Instagram ad"], (2, 1))
        self.assertEqual(sources[analytics.NOT_STATED], (2, 0))
        self.assertEqual(d.follow_ups["overdue"], 1)
        self.assertEqual(d.follow_ups["upcoming"], 1)
        self.assertEqual(sum(r["converted"] for r in d.conversion_trend), 1)
        self.assertEqual(sum(r["lost"] for r in d.conversion_trend), 1)

        text = " ".join(d.insights)
        self.assertIn("1 follow-up(s) are overdue", text)
        self.assertIn("Most requested service: Website Development (2 of 4 leads", text)
        self.assertIn("too few to compare conversion", text)
        self.assertNotIn("because", text.lower())

    def test_funnel_counts_lost_leads_at_furthest_stage(self):
        lead_id = self.chat_lead("Need a website")
        admin.change_status(self.conn, lead_id, "contacted")
        admin.change_status(self.conn, lead_id, "proposal")
        admin.mark_lost(self.conn, lead_id)
        funnel = {r["stage"]: r["leads"] for r in analytics.funnel(self.conn)}
        self.assertEqual(funnel, {"new": 1, "contacted": 1, "qualified": 1, "proposal": 1, "negotiation": 0,
                                  "converted": 0})

    def test_follow_up_on_time_vs_late(self):
        lead_id = self.chat_lead("Need a website")
        on_time = admin.schedule_follow_up(self.conn, lead_id, "2999-01-01")
        late = admin.schedule_follow_up(self.conn, lead_id, "2026-10-20")
        admin.complete_follow_up(self.conn, on_time, "done")
        admin.complete_follow_up(self.conn, late, "done", now="2026-10-22T10:00:00+00:00")
        stats = analytics.follow_up_performance(self.conn, TODAY)
        self.assertEqual((stats["done_on_time"], stats["done_late"]), (1, 1))

    def test_conversion_comparison_needs_enough_data(self):
        from services.analytics_service import build_insights
        k = {"total": 12, "unscored": 0, "hot": 4, "warm": 4, "cold": 4, "converted": 4, "lost": 2}
        sources = [{"source": "Instagram ad", "leads": 6, "converted": 3}, {"source": "Referral", "leads": 5,
                                                                             "converted": 1}]
        follow = {"overdue": 0, "done_on_time": 0, "done_late": 0}
        gaps = {"total": 12, "no_budget": 0, "no_timeline": 0, "no_company": 0, "no_contact": 0}
        attention = {"hot_unassigned": 0, "hot_still_new": 0}
        text = " ".join(build_insights(k, [], sources, follow, gaps, attention))
        self.assertIn("Instagram ad has the highest conversion so far (3 of 6)", text)
        self.assertIn("not evidence of cause", text)
        small = [{"source": "Instagram ad", "leads": 4, "converted": 3}, {"source": "Referral", "leads": 5,
                                                                           "converted": 1}]
        self.assertNotIn("highest conversion", " ".join(build_insights(k, [], small, follow, gaps, attention)))


if __name__ == "__main__":
    unittest.main()
