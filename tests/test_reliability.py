"""Reliability pass: end-to-end workflow and failure handling.

Run:  python -m unittest discover tests
"""

import json
import logging
import os
import sqlite3
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import groq
import httpx

import core.config as config
from ai.errors import ConfigurationError, LLMRequestError
from ai.groq_client import complete_json, create_client
from ai.nlp_service import analyze_conversation
from ai.responder import validate_reply
from core.business import BusinessProfile
from core.logging_setup import RedactSecretsFilter
from database import repository as repo
from database.connection import open_connection
from scoring import CRITERIA_MAX, classify
from services import conversation_service as cs
from services import lead_service as ls
from services.analytics_service import build_dashboard
from tests.test_conversation import ScriptedClient
from tests.test_nlp_service import SETTINGS, FakeClient, extraction, user


def tf(value, evidence=None, confidence=0.95, **kw):
    return {"value": value, "evidence": evidence or value, "confidence": confidence, **kw}


def status_error(cls, status, body=None):
    request = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    return cls("error", response=httpx.Response(status, request=request), body=body)


class DbCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.path = str(Path(self._tmp.name) / "leads.db")
        self.conn = open_connection(self.path)
        self.client = ScriptedClient()

    def tearDown(self):
        self.conn.close()
        self._tmp.cleanup()

    def count(self, table):
        return self.conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]

    def turn(self, conversation_id, text, reply, category="Website Development", level="medium", **entities):
        ls.record_customer_message(self.conn, conversation_id, text)
        self.client.extractions.append(extraction(category, level, **entities))
        self.client.replies.append(reply)
        return cs.process_turn(self.conn, conversation_id, SETTINGS, business=BusinessProfile(), client=self.client)


class EndToEndWorkflowTests(DbCase):
    """Workflow steps 1-10 from the reliability checklist, in order."""

    def test_full_lead_lifecycle(self):
        # 1. A new customer sends an enquiry.
        conversation_id = ls.start_conversation(self.conn, sender_phone="+91 98765 43210")
        first = self.turn(
            conversation_id,
            "Hi, I'm Priya. We need an e-commerce website for our bakery with online ordering.",
            "Thanks Priya! Do you have a rough budget and timeline in mind?",
            customer_name=tf("Priya", "I'm Priya"),
            interested_service=tf("e-commerce website"),
            requirement_description=tf("e-commerce website for our bakery"),
            requested_features={"values": ["online ordering"], "confidence": 0.9},
        )
        self.assertTrue(first.ok, first.error)
        # 2. Intent and entities were extracted (through the Groq client interface).
        self.assertEqual(first.outcome.profile.intent.category, "Website Development")
        self.assertEqual(first.outcome.profile.entities.customer_name.value, "Priya")
        self.assertEqual(self.client.json_calls[0]["response_format"], {"type": "json_object"})
        # 3. Score and category calculated deterministically.
        score1 = first.outcome.score
        self.assertEqual(sum(c.earned for c in score1.criteria), score1.total)
        self.assertEqual(score1.category, classify(score1.total))
        # 4. Lead and messages saved.
        lead_id = first.outcome.lead_id
        self.assertEqual(self.count("leads"), 1)
        self.assertEqual([m["sender_role"] for m in repo.get_messages(self.conn, conversation_id)],
                         ["customer", "assistant"])
        # 5. A useful follow-up question about missing information.
        self.assertEqual(first.reply.asked_topics, ["budget", "timeline"])

        # 6. The customer replies with more information.
        second = self.turn(conversation_id, "Budget is 1.5 lakh INR and we need it within 1 month.",
                           "Great, thank you! What's the name of your bakery?",
                           budget={"amount_min": 150000, "amount_max": 150000, "currency": "INR",
                                   "raw_text": "1.5 lakh INR", "confidence": 0.95},
                           timeline=tf("within 1 month"))
        self.assertTrue(second.ok, second.error)
        # 7. The same lead was updated, not duplicated.
        self.assertEqual(second.outcome.lead_id, lead_id)
        self.assertEqual(self.count("leads"), 1)
        lead = repo.get_lead(self.conn, lead_id)
        self.assertEqual((lead["customer_name"], lead["budget_max"], lead["timeline"]),
                         ("Priya", 150000, "within 1 month"))
        # 8. Score history recorded correctly.
        history = repo.score_history(self.conn, lead_id)
        self.assertEqual([h["total_score"] for h in history], [score1.total, second.outcome.score.total])
        self.assertGreater(history[1]["total_score"], history[0]["total_score"])
        self.assertEqual(lead["score"], history[-1]["total_score"])
        # 9. Dashboard reflects the latest data.
        dashboard = build_dashboard(self.conn, "2026-10-21")
        self.assertEqual(dashboard.kpis["total"], 1)
        self.assertEqual(dashboard.kpis["avg_score"], float(lead["score"]))
        self.assertEqual(dashboard.kpis[lead["category"].lower()], 1)
        self.assertEqual(dashboard.by_service[0]["value"], "Website Development")

        # 10. Reopening restores the context (history + profile) after a restart.
        self.conn.close()
        self.conn = open_connection(self.path)
        loaded = ls.load_conversation(self.conn, conversation_id)
        self.assertEqual(len(loaded["messages"]), 4)
        self.assertEqual(loaded["profile"].entities.budget.amount_max, 150000)
        third = self.turn(conversation_id, "It's Bloom Bakery", "Thanks!", company_name=tf("Bloom Bakery"))
        self.assertTrue(third.ok, third.error)
        extraction_prompt = self.client.json_calls[-1]["messages"][1]["content"]
        self.assertIn("1.5 lakh INR", extraction_prompt)  # previous entities passed as context
        self.assertIn("Business: Great, thank you! What's the name of your bakery?", extraction_prompt)
        self.assertEqual(len(self.client.reply_history()), 5)  # full history given to the reply model
        self.assertEqual(self.count("leads"), 1)


class ConfigTests(unittest.TestCase):
    def settings_with(self, **env):
        base = {"GROQ_API_KEY": "gsk_" + "a" * 40, "GROQ_MODEL": "some/model"}
        with mock.patch.object(config, "load_dotenv", lambda **kw: None), \
                mock.patch.dict(os.environ, {**base, **env}, clear=True):
            return config.get_settings()

    def test_valid_defaults(self):
        s = self.settings_with()
        self.assertEqual((s.groq_temperature, s.groq_reply_temperature, s.default_country_code), (0.0, 0.4, "91"))
        self.assertEqual(s.config_warnings, ())
        self.assertTrue(s.database_path.endswith(str(Path("data") / "leads.db")))

    def test_invalid_values_fall_back_with_warnings(self):
        s = self.settings_with(GROQ_TEMPERATURE="abc", GROQ_REPLY_TEMPERATURE="5", GROQ_REASONING_FORMAT="verbose",
                               DEFAULT_COUNTRY_CODE="+9a1")
        self.assertEqual((s.groq_temperature, s.groq_reply_temperature), (0.0, 0.4))
        self.assertIsNone(s.groq_reasoning_format)
        self.assertEqual(s.default_country_code, "91")
        self.assertEqual(len(s.config_warnings), 4)

    def test_malformed_key_warns(self):
        self.assertTrue(any("GROQ_API_KEY" in w for w in self.settings_with(GROQ_API_KEY="abc").config_warnings))

    def test_missing_key_or_model_blocks_ai(self):
        with self.assertRaisesRegex(ConfigurationError, "GROQ_API_KEY"):
            create_client(self.settings_with(GROQ_API_KEY=""))
        with self.assertRaisesRegex(ConfigurationError, "GROQ_MODEL"):
            create_client(self.settings_with(GROQ_MODEL=""))

    def test_settings_repr_hides_key(self):
        s = self.settings_with()
        self.assertNotIn(s.groq_api_key, repr(s))
        self.assertNotIn(s.groq_api_key, str(s))

    def test_sdk_retries_enabled(self):
        self.assertEqual(create_client(self.settings_with()).max_retries, 2)

    def test_unusable_database_path_raises_a_catchable_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            blocker = Path(tmp) / "not-a-folder"
            blocker.write_text("x")
            with self.assertRaises((sqlite3.Error, OSError)):
                open_connection(str(blocker / "leads.db"))


class GroqFailureTests(unittest.TestCase):
    def test_server_error(self):
        with self.assertRaisesRegex(LLMRequestError, "500"):
            analyze_conversation([user("hi")], settings=SETTINGS,
                                 client=FakeClient(status_error(groq.InternalServerError, 500)))

    def test_empty_choices(self):
        client = FakeClient()
        client.chat.completions.create = lambda **kw: SimpleNamespace(choices=[])
        with self.assertRaisesRegex(LLMRequestError, "empty response"):
            complete_json(client, SETTINGS, [])

    def test_code_fenced_json_is_accepted(self):
        result = analyze_conversation([user("need a site")], settings=SETTINGS,
                                      client=FakeClient("```json\n" + extraction() + "\n```"))
        self.assertEqual(result.intent.category, "Website Development")

    def test_malformed_outputs_are_retried_then_rejected(self):
        bad_outputs = [
            json.dumps({"intent": {"category": "Website Development", "confidence": "high"},
                        "purchase_intent": {"level": "high"}, "entities": {}}),             # wrong type
            json.dumps({"intent": {"category": "Website Development", "confidence": 1.5},
                        "purchase_intent": {"level": "high"}, "entities": {}}),             # out of range
            json.dumps({"purchase_intent": {"level": "high"}, "entities": {}}),             # missing intent
            '{"intent": ',                                                                   # truncated
            "[]",                                                                            # wrong shape
        ]
        for bad in bad_outputs:
            client = FakeClient(bad, extraction())
            result = analyze_conversation([user("hi")], settings=SETTINGS, client=client)
            self.assertEqual(result.intent.category, "Website Development", bad)
            self.assertEqual(len(client.calls), 2, bad)

    def test_extra_keys_ignored_and_bad_entity_values_cleaned(self):
        out = json.loads(extraction(company_name={"value": "N/A", "confidence": 0.9}))
        out["unexpected"] = {"anything": True}
        result = analyze_conversation([user("hi")], settings=SETTINGS, client=FakeClient(json.dumps(out)))
        self.assertIsNone(result.entities.company_name.value)


class LoggingTests(DbCase):
    def test_secrets_are_redacted(self):
        record = logging.LogRecord("leadbot", logging.INFO, __file__, 1,
                                   "key=%s", ("gsk_abcdefghijklmnopqrstuvwxyz",), None)
        RedactSecretsFilter().filter(record)
        self.assertNotIn("abcdefghijklmnop", record.getMessage())
        with mock.patch.dict(os.environ, {"GROQ_API_KEY": "my-secret-key-123"}):
            record = logging.LogRecord("leadbot", logging.INFO, __file__, 1, "auth my-secret-key-123", (), None)
            RedactSecretsFilter().filter(record)
            self.assertEqual(record.getMessage(), "auth ***")

    def test_logs_contain_no_customer_text_or_contacts(self):
        conversation_id = ls.start_conversation(self.conn)
        with self.assertLogs("leadbot", level="INFO") as logs:
            self.turn(conversation_id, "I'm Priya, email priya@bloom.in, call 9876543210",
                      "Thanks!", customer_name=tf("Priya", "I'm Priya"), email=tf("priya@bloom.in"),
                      whatsapp_number=tf("9876543210"))
        text = "\n".join(logs.output)
        self.assertIn("Lead #1 created", text)
        for private in ("priya@bloom.in", "9876543210", "I'm Priya"):
            self.assertNotIn(private, text)


class DatabaseSafetyTests(DbCase):
    def test_injection_attempts_are_inert(self):
        conversation_id = ls.start_conversation(self.conn)
        self.turn(conversation_id, "Need a website", "Sure!")
        self.assertEqual(repo.search_leads(self.conn, "' OR 1=1 --"), [])
        self.assertEqual(repo.search_leads(self.conn, "%"), [])
        for bad in ({"category; DROP TABLE leads": ["x"]}, {"profile_json": ["x"]}):
            with self.assertRaises(ValueError):
                repo.search_leads(self.conn, filters=bad)
        with self.assertRaises(ValueError):
            repo.distinct_lead_values(self.conn, "id; DROP TABLE leads")
        with self.assertRaises(ValueError):
            repo.create_lead(self.conn, {"name) VALUES (1); --": 1}, contact_time=repo.utc_now())
        with self.assertRaises(ValueError):
            repo.update_lead_admin(self.conn, 1, {"score": 100})  # not an admin-editable column
        self.assertEqual(self.count("leads"), 1)

    def test_failed_save_rolls_back_everything(self):
        conversation_id = ls.start_conversation(self.conn)
        ls.record_customer_message(self.conn, conversation_id, "Need a website")
        analysis = analyze_conversation([user("Need a website")], settings=SETTINGS, client=FakeClient(extraction()))
        with mock.patch.object(repo, "add_score", side_effect=sqlite3.OperationalError("disk full")):
            with self.assertRaises(sqlite3.OperationalError):
                ls.persist_analysis(self.conn, conversation_id, analysis)
        self.assertEqual(self.count("leads"), 0)
        self.assertIsNone(repo.get_conversation(self.conn, conversation_id)["lead_id"])
        self.assertEqual(repo.get_messages(self.conn, conversation_id)[0]["processing_status"], "pending")

    def test_save_failure_is_reported_not_raised(self):
        conversation_id = ls.start_conversation(self.conn)
        with mock.patch.object(repo, "add_score", side_effect=sqlite3.OperationalError("disk full")):
            result = self.turn(conversation_id, "Need a website", "Sure!")
        self.assertEqual(result.failed_stage, "save")
        self.assertIn("disk full", result.error)

    def test_unknown_conversation_is_reported_not_raised(self):
        result = cs.process_turn(self.conn, 999, SETTINGS, client=self.client)
        self.assertEqual(result.failed_stage, "analysis")

    def test_phone_owned_by_another_lead_never_violates_uniqueness(self):
        a = self.turn(ls.start_conversation(self.conn, sender_phone="9876543210"), "Hi", "Hello!")
        conv_b = ls.start_conversation(self.conn, sender_phone="9123456780")
        b = self.turn(conv_b, "Hi", "Hello!")
        # Customer B now types A's number: B keeps its own phone; no merge, no IntegrityError.
        result = self.turn(conv_b, "My other number is 9876543210", "Noted.", whatsapp_number=tf("9876543210"))
        self.assertTrue(result.ok, result.error)
        self.assertNotEqual(a.outcome.lead_id, b.outcome.lead_id)
        self.assertEqual(self.count("leads"), 2)
        self.assertEqual(repo.get_lead(self.conn, b.outcome.lead_id)["phone_normalized"], "+919123456780")


class NoFabricationTests(DbCase):
    def test_missing_fields_stay_null_in_database(self):
        conversation_id = ls.start_conversation(self.conn)
        result = self.turn(conversation_id, "Hi", "Hello! How can we help?", category="General Enquiry",
                           level="unknown", company_name={"value": "unknown", "confidence": 0.4},
                           email={"value": "not provided", "confidence": 0.3})
        lead = repo.get_lead(self.conn, result.outcome.lead_id)
        for column in ("customer_name", "company", "email", "phone", "budget_text", "timeline", "industry"):
            self.assertIsNone(lead[column], column)
        self.assertEqual(lead["decision_authority"], "unknown")
        self.assertIn("- nothing yet", self.client.reply_prompt())

    def test_scoring_weights_and_boundaries(self):
        self.assertEqual(sum(CRITERIA_MAX.values()), 100)
        self.assertEqual([classify(n) for n in (0, 49, 50, 79, 80, 100)],
                         ["COLD", "COLD", "WARM", "WARM", "HOT", "HOT"])


class CustomerSafetyTests(unittest.TestCase):
    def test_developer_information_is_rejected(self):
        for text in ('{"intent": "Website Development"}', "Based on the extracted details, you need a site.",
                     "Your confidence score is high.", "As shown in the developer panel...",
                     "Your lead score is 80/100.", "You are a HOT lead!"):
            self.assertTrue(validate_reply(text, "hi", BusinessProfile()), text)
        self.assertEqual(validate_reply("Thanks Priya! When would you like the site to go live?", "hi",
                                        BusinessProfile()), [])


if __name__ == "__main__":
    unittest.main()
