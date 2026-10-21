"""Tests for SQLite persistence and the lead workflow.

Each test uses a fresh database file in a temporary directory and drives the
same path the app uses: store message -> analyze_conversation (fake Groq
client) -> persist_analysis.

Run:  python -m unittest discover tests
"""

import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from ai.nlp_service import analyze_conversation
from database import repository as repo
from database.connection import open_connection, transaction
from database.schema import SCHEMA_VERSION, migrate
from services import lead_service as ls
from tests.test_nlp_service import SETTINGS, FakeClient, extraction


def tf(value, evidence=None, confidence=0.95, **kw):
    return {"value": value, "evidence": evidence or value, "confidence": confidence, **kw}


class DbTestCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.path = str(Path(self._tmp.name) / "data" / "leads.db")
        self.conn = open_connection(self.path)
        self.profiles = {}  # conversation_id -> last LeadAnalysis, as the session keeps it

    def tearDown(self):
        self.conn.close()
        self._tmp.cleanup()

    def reconnect(self):
        self.conn.close()
        self.conn = open_connection(self.path)

    def start(self, phone=None, lead_id=None):
        conversation_id = ls.start_conversation(self.conn, sender_phone=phone, lead_id=lead_id)
        loaded = ls.load_conversation(self.conn, conversation_id)
        self.profiles[conversation_id] = loaded["profile"]
        return conversation_id

    def turn(self, conversation_id, text, category="Website Development", level="medium", **entities):
        """One customer message: save it, analyse the conversation, persist the lead."""
        ls.record_customer_message(self.conn, conversation_id, text)
        messages = [
            {"role": "user", "content": m["message_text"], "placeholder": False}
            for m in repo.get_messages(self.conn, conversation_id)
        ]
        analysis = analyze_conversation(
            messages, previous=self.profiles.get(conversation_id), settings=SETTINGS,
            client=FakeClient(extraction(category, level, **entities)),
        )
        outcome = ls.persist_analysis(self.conn, conversation_id, analysis)
        self.profiles[conversation_id] = outcome.profile
        return outcome

    def count(self, table):
        return self.conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]


class SchemaTests(DbTestCase):
    def test_tables_version_and_seeded_services(self):
        tables = {r["name"] for r in self.conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
        self.assertTrue({"leads", "conversations", "messages", "lead_scores", "follow_ups",
                         "activities", "services"} <= tables)
        self.assertEqual(self.conn.execute("PRAGMA user_version").fetchone()[0], SCHEMA_VERSION)
        self.assertEqual(len(repo.list_services(self.conn)), 9)
        self.assertTrue(Path(self.path).exists())

    def test_migrate_is_idempotent(self):
        self.assertEqual(migrate(self.conn), SCHEMA_VERSION)
        self.reconnect()
        self.assertEqual(len(repo.list_services(self.conn)), 9)

    def test_upgrade_from_version_1_keeps_data(self):
        from database.schema import MIGRATIONS, _split
        path = str(Path(self._tmp.name) / "v1.db")
        old = sqlite3.connect(path, autocommit=True)
        old.execute("BEGIN")
        for statement in _split(MIGRATIONS[0][1]):
            old.execute(statement)
        old.execute("PRAGMA user_version = 1")
        old.execute("COMMIT")
        now = repo.utc_now()
        old.execute("INSERT INTO conversations (session_identifier, started_at, last_message_at, created_at, "
                    "updated_at) VALUES ('s1', ?, ?, ?, ?)", (now, now, now, now))
        old.execute("INSERT INTO messages (conversation_id, sender_role, message_text, sent_at) "
                    "VALUES (1, 'customer', 'old message', ?)", (now,))
        old.close()

        upgraded = open_connection(path)
        self.assertEqual(upgraded.execute("PRAGMA user_version").fetchone()[0], SCHEMA_VERSION)
        message = repo.get_messages(upgraded, 1)[0]
        self.assertEqual((message["message_text"], message["asked_topics"]), ("old message", None))
        upgraded.close()

    def test_foreign_keys_enforced(self):
        with self.assertRaises(sqlite3.IntegrityError):
            repo.add_message(self.conn, 999, "customer", "orphan")

    def test_json_columns_must_be_valid(self):
        lead_id = repo.create_lead(self.conn, {}, contact_time=repo.utc_now())
        with self.assertRaises(sqlite3.IntegrityError):
            self.conn.execute("UPDATE leads SET requested_modules = 'not json' WHERE id = ?", (lead_id,))

    def test_phone_is_unique(self):
        repo.create_lead(self.conn, {"phone_normalized": "+919876543210"}, contact_time=repo.utc_now())
        with self.assertRaises(sqlite3.IntegrityError):
            repo.create_lead(self.conn, {"phone_normalized": "+919876543210"}, contact_time=repo.utc_now())

    def test_unknown_lead_column_rejected(self):
        with self.assertRaises(ValueError):
            repo.update_lead(self.conn, 1, {"score; DROP TABLE leads": 1})

    def test_transaction_rolls_back(self):
        with self.assertRaises(RuntimeError):
            with transaction(self.conn):
                repo.create_lead(self.conn, {}, contact_time=repo.utc_now())
                raise RuntimeError("boom")
        self.assertEqual(self.count("leads"), 0)

    def test_follow_up_and_activity_records(self):
        lead_id = repo.create_lead(self.conn, {}, contact_time=repo.utc_now())
        repo.add_follow_up(self.conn, lead_id, "2026-11-01", next_action="Send proposal")
        repo.log_activity(self.conn, lead_id, "note", "Called customer")
        repo.add_follow_up(self.conn, lead_id, "2026-11-08", next_action="Call again")
        self.assertEqual([f["follow_up_count"] for f in repo.list_follow_ups(self.conn, lead_id)], [1, 2])
        self.assertEqual(repo.list_activities(self.conn, lead_id)[0]["description"], "Called customer")


class CreateLeadTests(DbTestCase):
    def test_first_analysed_message_creates_lead(self):
        conv = self.start()
        outcome = self.turn(conv, "Hi, I'm Priya from Bloom Bakery. Need a website.",
                            customer_name=tf("Priya", "I'm Priya"), company_name=tf("Bloom Bakery"),
                            interested_service=tf("website", "Need a website"))
        self.assertEqual(outcome.action, "created")
        lead = repo.get_lead(self.conn, outcome.lead_id)
        self.assertEqual((lead["customer_name"], lead["company"]), ("Priya", "Bloom Bakery"))
        self.assertEqual(lead["service_intent"], "Website Development")
        self.assertEqual(lead["service_id"], repo.get_service_id(self.conn, "Website Development"))
        self.assertEqual(lead["score"], outcome.score.total)
        self.assertEqual(lead["status"], "new")
        self.assertEqual(repo.get_conversation(self.conn, conv)["lead_id"], outcome.lead_id)
        types = [a["activity_type"] for a in repo.list_activities(self.conn, outcome.lead_id)]
        self.assertIn("lead_created", types)

    def test_message_is_stored_before_analysis(self):
        conv = self.start()
        ls.record_customer_message(self.conn, conv, "hello")
        messages = repo.get_messages(self.conn, conv)
        self.assertEqual(len(messages), 1)
        self.assertEqual(messages[0]["processing_status"], "pending")
        self.assertEqual(self.count("leads"), 0)  # no lead until analysis succeeds

    def test_failed_analysis_marks_messages(self):
        conv = self.start()
        ls.record_customer_message(self.conn, conv, "hello")
        ls.record_analysis_failure(self.conn, conv, "Groq rate limit reached (429).")
        message = repo.get_messages(self.conn, conv)[0]
        self.assertEqual(message["processing_status"], "failed")
        self.assertIn("429", message["processing_error"])
        self.assertEqual(len(repo.list_conversations(self.conn, None)), 1)  # reopenable as unlinked

    def test_sql_is_parameterised(self):
        conv = self.start()
        text = "Robert'); DROP TABLE leads;--"
        ls.record_customer_message(self.conn, conv, text)
        self.assertEqual(repo.get_messages(self.conn, conv)[0]["message_text"], text)
        self.assertEqual(self.count("leads"), 0)


class UpdateLeadTests(DbTestCase):
    def test_new_information_updates_same_lead(self):
        conv = self.start()
        first = self.turn(conv, "I'm Priya, need a website", customer_name=tf("Priya", "I'm Priya"))
        second = self.turn(conv, "Budget is 50k INR, need it in 1 month",
                           budget={"amount_min": 50000, "amount_max": 50000, "currency": "INR",
                                   "raw_text": "50k INR", "confidence": 0.95},
                           timeline=tf("1 month", "in 1 month"))
        self.assertEqual(second.lead_id, first.lead_id)
        self.assertEqual(second.action, "updated")
        self.assertIn("Budget", second.changed_fields)
        self.assertIn("Timeline", second.changed_fields)
        self.assertEqual(self.count("leads"), 1)
        lead = repo.get_lead(self.conn, first.lead_id)
        self.assertEqual((lead["customer_name"], lead["budget_max"], lead["currency"]), ("Priya", 50000, "INR"))

    def test_known_values_not_erased_by_nulls(self):
        conv = self.start()
        first = self.turn(conv, "I'm Priya from Bloom Bakery in Pune",
                          customer_name=tf("Priya", "I'm Priya"), company_name=tf("Bloom Bakery"),
                          location=tf("Pune"))
        self.turn(conv, "Can you also do a logo?")  # extraction returns nothing for these fields
        lead = repo.get_lead(self.conn, first.lead_id)
        self.assertEqual((lead["customer_name"], lead["company"], lead["location"]),
                         ("Priya", "Bloom Bakery", "Pune"))

    def test_correction_updates_lead_and_keeps_history(self):
        conv = self.start()
        first = self.turn(conv, "My email is priya@old.com", email=tf("priya@old.com"))
        second = self.turn(conv, "Sorry, my email is actually priya@new.com",
                           email=tf("priya@new.com", is_correction=True))
        lead = repo.get_lead(self.conn, first.lead_id)
        self.assertEqual(lead["email"], "priya@new.com")
        self.assertIn("Email", second.changed_fields)
        self.assertEqual(len(repo.get_messages(self.conn, conv)), 2)
        activities = repo.list_activities(self.conn, first.lead_id)
        self.assertTrue(any(a["activity_type"] == "field_corrected" and "priya@new.com" in a["description"]
                            for a in activities))

    def test_conflict_is_logged_and_value_kept(self):
        conv = self.start()
        first = self.turn(conv, "We are in Pune", location=tf("Pune"))
        self.turn(conv, "Our office is in Mumbai", location=tf("Mumbai"))
        self.assertEqual(repo.get_lead(self.conn, first.lead_id)["location"], "Pune")
        self.assertTrue(any(a["activity_type"] == "conflict_detected"
                            for a in repo.list_activities(self.conn, first.lead_id)))

    def test_last_contact_and_updated_at_change(self):
        conv = self.start()
        outcome = self.turn(conv, "hello", customer_name=tf("Priya", "hello Priya"))
        self.conn.execute("UPDATE leads SET last_contact_date = '2000-01-01T00:00:00+00:00', "
                          "updated_at = '2000-01-01T00:00:00+00:00' WHERE id = ?", (outcome.lead_id,))
        ls.record_customer_message(self.conn, conv, "are you there?")
        lead = repo.get_lead(self.conn, outcome.lead_id)
        self.assertGreater(lead["last_contact_date"], "2000-01-01T00:00:00+00:00")
        self.assertGreater(lead["updated_at"], "2000-01-01T00:00:00+00:00")
        self.assertEqual(lead["first_contact_date"] <= lead["last_contact_date"], True)


class DuplicateAvoidanceTests(DbTestCase):
    def test_returning_sender_phone_reuses_lead(self):
        conv1 = self.start(phone="+91 98765 43210")
        first = self.turn(conv1, "I'm Priya, need a website", customer_name=tf("Priya", "I'm Priya"))
        conv2 = self.start(phone="09876543210")  # same number, different format
        self.assertEqual(repo.get_conversation(self.conn, conv2)["lead_id"], first.lead_id)
        self.assertIsNotNone(self.profiles[conv2])  # previous profile is loaded for the NLP context
        second = self.turn(conv2, "Hi again, budget is 1 lakh",
                           budget={"amount_min": 100000, "amount_max": 100000, "currency": "INR",
                                   "raw_text": "1 lakh", "confidence": 0.9})
        self.assertEqual(second.lead_id, first.lead_id)
        self.assertEqual(self.count("leads"), 1)
        self.assertEqual(len(repo.list_conversations(self.conn, first.lead_id)), 2)
        self.assertEqual(repo.get_lead(self.conn, first.lead_id)["customer_name"], "Priya")

    def test_typed_phone_merges_provisional_lead(self):
        conv1 = self.start()
        first = self.turn(conv1, "I'm Priya, call me on 9876543210",
                          customer_name=tf("Priya", "I'm Priya"), whatsapp_number=tf("9876543210"),
                          company_name=tf("Bloom Bakery", "Priya"))
        conv2 = self.start()  # anonymous chat: no identity yet
        provisional = self.turn(conv2, "Need a logo too", category="Graphic Design")
        self.assertEqual(provisional.action, "created")
        self.assertEqual(self.count("leads"), 2)

        merged = self.turn(conv2, "It's Priya again, my number is +91 98765 43210",
                           category="Graphic Design", whatsapp_number=tf("+91 98765 43210"))
        self.assertEqual(merged.action, "merged")
        self.assertEqual(merged.lead_id, first.lead_id)
        self.assertEqual(self.count("leads"), 1)
        self.assertIsNone(repo.get_lead(self.conn, provisional.lead_id))
        self.assertEqual(len(repo.list_conversations(self.conn, first.lead_id)), 2)
        self.assertEqual(len(repo.get_messages(self.conn, conv2)), 2)
        lead = repo.get_lead(self.conn, first.lead_id)
        self.assertEqual((lead["customer_name"], lead["company"]), ("Priya", "Bloom Bakery"))
        self.assertEqual(lead["phone_normalized"], "+919876543210")

    def test_email_matches_when_no_phone_conflict(self):
        conv1 = self.start()
        first = self.turn(conv1, "Email me at priya@bloom.in", email=tf("priya@bloom.in"))
        conv2 = self.start()
        second = self.turn(conv2, "Hi, PRIYA@bloom.in here", email=tf("PRIYA@bloom.in"))
        self.assertEqual(second.lead_id, first.lead_id)
        self.assertEqual(second.matched_by, "email")
        self.assertEqual(self.count("leads"), 1)

    def test_same_email_different_phone_is_not_merged(self):
        conv1 = self.start(phone="9876543210")
        first = self.turn(conv1, "Email priya@bloom.in", email=tf("priya@bloom.in"))
        conv2 = self.start(phone="9123456780")
        second = self.turn(conv2, "Email priya@bloom.in", email=tf("priya@bloom.in"))
        self.assertNotEqual(second.lead_id, first.lead_id)
        self.assertEqual(self.count("leads"), 2)

    def test_anonymous_customers_are_not_merged(self):
        a = self.turn(self.start(), "Need a website")
        b = self.turn(self.start(), "Need a website")
        self.assertNotEqual(a.lead_id, b.lead_id)
        self.assertEqual(self.count("leads"), 2)

    def test_reopened_lead_new_conversation_is_linked(self):
        first = self.turn(self.start(), "I'm Priya", customer_name=tf("Priya", "I'm Priya"))
        loaded = ls.load_lead(self.conn, first.lead_id)
        conv2 = ls.start_conversation(self.conn, lead_id=first.lead_id)
        self.profiles[conv2] = loaded["profile"]
        second = self.turn(conv2, "Back again about the website")
        self.assertEqual(second.lead_id, first.lead_id)
        self.assertEqual(self.count("leads"), 1)
        self.assertEqual(repo.get_lead(self.conn, first.lead_id)["customer_name"], "Priya")


class HistoryTests(DbTestCase):
    def test_conversation_history_survives_restart(self):
        conv = self.start()
        outcome = self.turn(conv, "First message", customer_name=tf("Priya", "First message Priya"))
        self.turn(conv, "Second message")
        self.reconnect()  # simulates an app restart
        loaded = ls.load_conversation(self.conn, conv)
        self.assertEqual([m["message_text"] for m in loaded["messages"]], ["First message", "Second message"])
        self.assertTrue(all(m["processing_status"] == "processed" for m in loaded["messages"]))
        self.assertEqual(loaded["lead"]["id"], outcome.lead_id)
        self.assertIsNotNone(loaded["profile"])
        self.assertEqual(loaded["score"].total, repo.get_lead(self.conn, outcome.lead_id)["score"])

    def test_score_history_only_when_score_changes(self):
        conv = self.start()
        first = self.turn(conv, "Need a website")
        self.assertTrue(first.score_saved)
        same = self.turn(conv, "Need a website")  # second message: engagement +1, so it changes
        unchanged_total = same.score.total
        self.assertTrue(same.score_saved)
        # Re-running the same analysis without new messages must not add a row.
        rows_before = len(repo.score_history(self.conn, first.lead_id))
        analysis = self.profiles[conv]
        again = ls.persist_analysis(self.conn, conv, analysis)
        self.assertFalse(again.score_saved)
        self.assertEqual(again.action, "unchanged")
        self.assertEqual(len(repo.score_history(self.conn, first.lead_id)), rows_before)
        self.assertEqual(again.score.total, unchanged_total)

    def test_score_history_records_increase(self):
        conv = self.start()
        first = self.turn(conv, "Need a website")
        second = self.turn(conv, "Budget 50k INR, timeline 2 weeks", level="high",
                           budget={"amount_min": 50000, "amount_max": 50000, "currency": "INR",
                                   "raw_text": "50k INR", "confidence": 0.95},
                           timeline=tf("2 weeks", "timeline 2 weeks"))
        history = repo.score_history(self.conn, first.lead_id)
        self.assertEqual(len(history), 2)
        self.assertLess(history[0]["total_score"], history[1]["total_score"])
        self.assertEqual(second.previous_score, history[0]["total_score"])
        for row in history:
            breakdown = json.loads(row["breakdown_json"])
            self.assertEqual(len(breakdown), 7)
            self.assertEqual(sum(c["earned"] for c in breakdown), row["total_score"])
            self.assertTrue(row["explanation"])
        self.assertTrue(any(a["activity_type"] == "score_changed"
                            for a in repo.list_activities(self.conn, first.lead_id)))


class NormalisationTests(unittest.TestCase):
    def test_phone(self):
        cases = {
            "+91 98765 43210": "+919876543210",
            "9876543210": "+919876543210",
            "09876543210": "+919876543210",
            "0091 98765-43210": "+919876543210",
            "+1 (415) 555-0100": "+14155550100",
            "12345": None,
            "": None,
            None: None,
        }
        for raw, expected in cases.items():
            self.assertEqual(ls.normalize_phone(raw), expected, raw)

    def test_email(self):
        self.assertEqual(ls.normalize_email("  Priya@Bloom.IN "), "priya@bloom.in")
        self.assertIsNone(ls.normalize_email("not-an-email"))

    def test_invalid_sender_phone_rejected(self):
        conn = open_connection(":memory:")
        with self.assertRaises(ValueError):
            ls.start_conversation(conn, sender_phone="123")
        conn.close()


if __name__ == "__main__":
    unittest.main()
