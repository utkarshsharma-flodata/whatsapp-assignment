"""Multi-turn tests for contextual reply generation.

A scripted fake Groq client routes JSON-mode calls (extraction) and text
calls (replies) to separate queues and records every prompt, so the tests
can check what the model was told. No API calls are made.

Run:  python -m unittest discover tests
"""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import groq
import httpx

from ai.responder import plan_questions, validate_reply
from ai.schemas import Entities, IntentResult, LeadAnalysis, PurchaseIntent
from core.business import BusinessProfile
from database import repository as repo
from database.connection import open_connection
from services import conversation_service as cs
from services import lead_service as ls
from tests.test_nlp_service import SETTINGS, extraction


def tf(value, evidence=None, confidence=0.95, **kw):
    return {"value": value, "evidence": evidence or value, "confidence": confidence, **kw}


class ScriptedClient:
    """Fake Groq client: JSON-mode calls pop `extractions`, text calls pop `replies`."""

    def __init__(self):
        self.extractions, self.replies = [], []
        self.json_calls, self.text_calls = [], []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        is_json = kwargs.get("response_format") == {"type": "json_object"}
        (self.json_calls if is_json else self.text_calls).append(kwargs)
        out = (self.extractions if is_json else self.replies).pop(0)
        if isinstance(out, Exception):
            raise out
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=out))])

    def reply_prompt(self, index=-1) -> str:
        return self.text_calls[index]["messages"][0]["content"]

    def reply_history(self, index=-1) -> list[dict]:
        return self.text_calls[index]["messages"][1:]


def ask_next(prompt: str) -> str:
    return prompt.split("## Ask next", 1)[1]


def known(prompt: str) -> str:
    return prompt.split("## Already known (provided by the customer)", 1)[1].split("##", 1)[0]


class ConversationTestCase(unittest.TestCase):
    business = BusinessProfile(name="Acme Tech")

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.conn = open_connection(str(Path(self._tmp.name) / "leads.db"))
        self.client = ScriptedClient()

    def tearDown(self):
        self.conn.close()
        self._tmp.cleanup()

    def turn(self, conversation_id, text, reply, category="Website Development", level="medium", **entities):
        ls.record_customer_message(self.conn, conversation_id, text)
        self.client.extractions.append(extraction(category, level, **entities))
        self.client.replies.extend(reply if isinstance(reply, list) else [reply])
        return cs.process_turn(self.conn, conversation_id, SETTINGS, business=self.business, client=self.client)

    def messages(self, conversation_id):
        return [(m["sender_role"], m["message_text"]) for m in repo.get_messages(self.conn, conversation_id)]

    def lead(self, result):
        return repo.get_lead(self.conn, result.outcome.lead_id)


WEBSITE_DETAILS = dict(
    interested_service=tf("e-commerce website", "need an e-commerce website"),
    requirement_description=tf("e-commerce website for a bakery", "need an e-commerce website"),
    requested_features={"values": ["online ordering", "payments"], "confidence": 0.9},
    budget={"amount_min": 150000, "amount_max": 150000, "currency": "INR", "raw_text": "1.5 lakh",
            "confidence": 0.95},
    timeline=tf("6 weeks", "within 6 weeks"),
)
WEBSITE_MESSAGE = ("Hi, I'm Priya. We need an e-commerce website with online ordering and payments. "
                   "Budget is 1.5 lakh, within 6 weeks.")


class MultiTurnTests(ConversationTestCase):
    def test_company_name_given_after_being_asked(self):
        conv = ls.start_conversation(self.conn)
        first = self.turn(conv, WEBSITE_MESSAGE, "Thanks Priya! What's the name of your business?",
                          customer_name=tf("Priya", "I'm Priya"), **WEBSITE_DETAILS)
        self.assertTrue(first.ok, first.error)
        self.assertEqual(first.reply.asked_topics, ["company", "contact"])
        self.assertIn("company or business name", ask_next(self.client.reply_prompt()))
        score_before = first.outcome.score.total

        second = self.turn(conv, "Bloom Bakery", "Great, thanks! Could you share an email for the proposal?",
                           company_name=tf("Bloom Bakery"))
        self.assertTrue(second.ok, second.error)
        # The extractor saw our question, so a bare answer can be understood.
        transcript = self.client.json_calls[-1]["messages"][1]["content"]
        self.assertIn("Business: Thanks Priya! What's the name of your business?", transcript)
        self.assertIn("Customer: Bloom Bakery", transcript)
        self.assertEqual(self.lead(second)["company"], "Bloom Bakery")
        self.assertGreater(second.outcome.score.total, score_before)
        prompt = self.client.reply_prompt()
        self.assertIn("Company: Bloom Bakery", known(prompt))
        self.assertNotIn("company", second.reply.asked_topics)
        self.assertNotIn("company or business name", ask_next(prompt))
        self.assertEqual([r for r, _ in self.messages(conv)], ["customer", "assistant", "customer", "assistant"])

    def test_budget_given_in_later_message(self):
        conv = ls.start_conversation(self.conn)
        first = self.turn(conv, "We need an e-commerce website with online ordering and payments",
                          "Sounds good! Do you have a rough budget in mind?",
                          interested_service=tf("e-commerce website", "need an e-commerce website"),
                          requirement_description=tf("e-commerce website", "need an e-commerce website"),
                          requested_features={"values": ["online ordering", "payments"], "confidence": 0.9})
        self.assertIn("budget", first.reply.asked_topics)
        self.assertEqual(first.outcome.score.criteria[1].earned, 0)

        second = self.turn(conv, "Budget is around 2 lakh INR", "Noted, 2 lakh works as a starting point. "
                           "When would you like it to go live?",
                           budget={"amount_min": 200000, "amount_max": 200000, "currency": "INR",
                                   "raw_text": "2 lakh INR", "confidence": 0.95})
        self.assertTrue(second.ok, second.error)
        self.assertEqual(self.lead(second)["budget_max"], 200000)
        self.assertEqual(second.outcome.score.criteria[1].earned, 20)
        self.assertTrue(second.outcome.score_saved)
        self.assertEqual(len(repo.score_history(self.conn, second.outcome.lead_id)), 2)
        self.assertNotIn("budget", second.reply.asked_topics)
        self.assertIn("timeline", second.reply.asked_topics)
        self.assertIn("Budget: INR 200,000", known(self.client.reply_prompt()))

    def test_customer_changes_timeline(self):
        conv = ls.start_conversation(self.conn)
        first = self.turn(conv, "Need a website in 3 months", "Got it, 3 months. What features do you need?",
                          interested_service=tf("website", "Need a website"), timeline=tf("3 months"))
        self.assertEqual(self.lead(first)["timeline"], "3 months")
        timeline_before = first.outcome.score.criteria[2].earned

        second = self.turn(conv, "Actually we need it in 1 month", "Understood, we'll plan for 1 month.",
                           timeline=tf("1 month", "Actually we need it in 1 month", is_correction=True))
        self.assertTrue(second.ok, second.error)
        self.assertEqual(self.lead(second)["timeline"], "1 month")
        self.assertIn("Timeline", second.outcome.changed_fields)
        self.assertGreater(second.outcome.score.criteria[2].earned, timeline_before)
        self.assertIn("Timeline: 1 month", known(self.client.reply_prompt()))
        self.assertTrue(any(a["activity_type"] == "field_corrected"
                            for a in repo.list_activities(self.conn, second.outcome.lead_id)))
        self.assertEqual(len([m for m in self.messages(conv) if m[0] == "customer"]), 2)

    def test_pricing_enquiry_without_configured_prices(self):
        conv = ls.start_conversation(self.conn)
        result = self.turn(
            conv, "How much does a website cost?",
            ["A website is usually ₹50,000.",  # invented price: rejected
             "It depends on what you need. Could you tell me a bit about the website you have in mind?"],
            category="Pricing Enquiry", level="low",
            questions={"values": ["How much does a website cost?"], "confidence": 0.95},
        )
        self.assertTrue(result.ok, result.error)
        self.assertTrue(result.reply.regenerated)
        self.assertNotIn("₹", result.reply.text)
        self.assertEqual(result.reply.asked_topics, ["requirement"])  # one question: customer asked one
        prompt = self.client.reply_prompt(0)
        self.assertIn("Configured prices: NONE", prompt)
        self.assertIn("rejected", self.client.text_calls[1]["messages"][-1]["content"])
        saved = self.messages(conv)
        self.assertEqual(saved[-1], ("assistant", result.reply.text))
        self.assertNotIn("₹50,000", " ".join(t for _, t in saved))

    def test_pricing_with_configured_prices(self):
        self.business = BusinessProfile(name="Acme Tech", pricing={"Website Development": "Starts at ₹40,000"})
        conv = ls.start_conversation(self.conn)
        result = self.turn(conv, "What's your website price?",
                           "Our websites start at ₹40,000. What would you like the site to do?",
                           category="Pricing Enquiry", level="low")
        self.assertTrue(result.ok, result.error)
        self.assertFalse(result.reply.regenerated)
        self.assertIn("Website Development: Starts at ₹40,000", self.client.reply_prompt())

    def test_returning_customer_reopened_conversation(self):
        conv1 = ls.start_conversation(self.conn, sender_phone="9876543210")
        first = self.turn(conv1, WEBSITE_MESSAGE + " We're Bloom Bakery.", "Thanks Priya!",
                          customer_name=tf("Priya", "I'm Priya"), company_name=tf("Bloom Bakery"),
                          **WEBSITE_DETAILS)
        lead_id = first.outcome.lead_id

        # Reopen the same conversation later: history comes from the database.
        reopened = self.turn(conv1, "Any update?", "The team is reviewing your requirements, Priya.")
        self.assertTrue(reopened.ok, reopened.error)
        history = self.client.reply_history()
        self.assertEqual([m["role"] for m in history], ["user", "assistant", "user"])
        self.assertEqual(history[1]["content"], "Thanks Priya!")

        # Start a new conversation from the lead browser: same lead, known profile.
        conv2 = ls.start_conversation(self.conn, lead_id=lead_id)
        result = self.turn(conv2, "Hi, it's me again. Can we also add a blog?",
                           "Welcome back, Priya! Yes, we can include a blog.",
                           requested_features={"values": ["blog"], "confidence": 0.9})
        self.assertTrue(result.ok, result.error)
        self.assertEqual(result.outcome.lead_id, lead_id)
        self.assertEqual(self.count("leads"), 1)
        prompt = self.client.reply_prompt()
        for fact in ("Name: Priya", "Company: Bloom Bakery", "Budget: INR 150,000", "Timeline: 6 weeks"):
            self.assertIn(fact, known(prompt))
        self.assertTrue(set(result.reply.asked_topics).isdisjoint({"name", "company", "budget", "timeline"}))
        lead = repo.get_lead(self.conn, lead_id)
        self.assertIn("blog", lead["requested_modules"])
        self.assertEqual(lead["company"], "Bloom Bakery")
        # The previous profile was given to the extractor as context.
        self.assertIn("Bloom Bakery", self.client.json_calls[-1]["messages"][1]["content"])

    def test_ignored_question_is_not_asked_a_third_time(self):
        conv = ls.start_conversation(self.conn, sender_phone="9876543210")
        details = dict(WEBSITE_DETAILS, company_name=tf("Bloom Bakery", "We're Bloom Bakery"))
        first = self.turn(conv, WEBSITE_MESSAGE.replace("I'm Priya. ", "") + " We're Bloom Bakery.",
                          "Thanks! May I have your name?", **details)
        self.assertIn("name", first.reply.asked_topics)
        second = self.turn(conv, "Can you also do a logo?", "Yes, we can. May I have your name?")
        self.assertIn("name", second.reply.asked_topics)
        third = self.turn(conv, "And SEO?", "Yes, SEO is something we offer too.")
        self.assertNotIn("name", third.reply.asked_topics)
        self.assertNotIn("their name", ask_next(self.client.reply_prompt()))
        stored = [m["asked_topics"] for m in repo.get_messages(self.conn, conv) if m["sender_role"] == "assistant"]
        self.assertEqual(len(stored), 3)
        self.assertIn("name", stored[0])

    def test_status_says_returning_only_when_matched(self):
        conv1 = ls.start_conversation(self.conn)
        self.turn(conv1, "Hi, call me on 9876543210", "Thanks!", whatsapp_number=tf("9876543210"))
        later = self.turn(conv1, "Need a website", "Great!")
        self.assertNotIn("Returning customer", later.outcome.summary())
        conv2 = ls.start_conversation(self.conn)
        returning = self.turn(conv2, "It's me again, 9876543210", "Welcome back!",
                              whatsapp_number=tf("9876543210"))
        self.assertIn("Returning customer matched by phone", returning.outcome.summary())

    def count(self, table):
        return self.conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]


class FailureTests(ConversationTestCase):
    def test_analysis_failure_makes_no_reply(self):
        conv = ls.start_conversation(self.conn)
        ls.record_customer_message(self.conn, conv, "hello")
        request = httpx.Request("POST", "https://api.groq.com")
        self.client.extractions.append(groq.RateLimitError("rate", response=httpx.Response(429, request=request),
                                                           body=None))
        result = cs.process_turn(self.conn, conv, SETTINGS, business=self.business, client=self.client)
        self.assertFalse(result.ok)
        self.assertEqual(result.failed_stage, "analysis")
        self.assertIn("rate limit", result.error)
        self.assertEqual(self.client.text_calls, [])
        self.assertEqual(self.messages(conv), [("customer", "hello")])
        self.assertEqual(repo.get_messages(self.conn, conv)[0]["processing_status"], "failed")

    def test_reply_failure_keeps_lead_and_saves_no_fake_reply(self):
        conv = ls.start_conversation(self.conn)
        request = httpx.Request("POST", "https://api.groq.com")
        error = groq.APIConnectionError(request=request)
        result = self.turn(conv, "I'm Priya, need a website", error, customer_name=tf("Priya", "I'm Priya"))
        self.assertFalse(result.ok)
        self.assertEqual(result.failed_stage, "reply")
        self.assertIsNotNone(result.outcome)  # lead was saved before the reply step
        self.assertEqual(self.lead(result)["customer_name"], "Priya")
        self.assertEqual(self.messages(conv), [("customer", "I'm Priya, need a website")])

        self.client.replies.append("Hi Priya! What kind of website do you need?")
        retry = cs.retry_reply(self.conn, conv, SETTINGS, business=self.business, client=self.client)
        self.assertTrue(retry.ok, retry.error)
        self.assertEqual(self.messages(conv)[-1], ("assistant", "Hi Priya! What kind of website do you need?"))
        self.assertEqual(len(self.client.json_calls), 1)  # retry does not re-run extraction

    def test_reply_rejected_twice_is_an_error(self):
        conv = ls.start_conversation(self.conn)
        result = self.turn(conv, "Price?", ["It's $500.", "Only $450 for you!"], category="Pricing Enquiry")
        self.assertFalse(result.ok)
        self.assertEqual(result.failed_stage, "reply")
        self.assertIn("rejected", result.error)
        self.assertEqual([r for r, _ in self.messages(conv)], ["customer"])

    def test_internal_scoring_never_reaches_customer(self):
        conv = ls.start_conversation(self.conn)
        result = self.turn(conv, "Need a website", ["You're a WARM lead with a score of 40/100!",
                                                    "Happy to help! What should the website do?"])
        self.assertTrue(result.ok, result.error)
        self.assertTrue(result.reply.regenerated)
        prompt = self.client.reply_prompt(0)
        self.assertNotIn(str(result.outcome.score.total) + "/100", prompt)
        for internal in ("WARM", "COLD", "HOT", "purchase_intent", "score:"):
            self.assertNotIn(internal, prompt.split("## Business information")[1])

    def test_missing_api_key_is_reported(self):
        conv = ls.start_conversation(self.conn)
        ls.record_customer_message(self.conn, conv, "hi")
        settings = type(SETTINGS)(**{**SETTINGS.__dict__, "groq_api_key": None})
        result = cs.process_turn(self.conn, conv, settings, business=self.business)
        self.assertEqual(result.failed_stage, "analysis")
        self.assertIn("GROQ_API_KEY", result.error)


def profile(category="Website Development", conflicts=(), **entities) -> LeadAnalysis:
    return LeadAnalysis(
        intent=IntentResult(category=category, confidence=0.9),
        purchase_intent=PurchaseIntent(level="medium", confidence=0.8),
        entities=Entities.model_validate(entities),
        conflicts=list(conflicts),
        model="test",
        analysed_messages=1,
    )


class QuestionPlanningTests(unittest.TestCase):
    def test_first_message_without_profile(self):
        self.assertEqual(plan_questions(None, "hi"), ["requirement"])

    def test_at_most_two_and_never_known(self):
        p = profile(interested_service=tf("website"), budget={"raw_text": "50k", "amount_min": 50000,
                                                              "amount_max": 50000, "confidence": 0.9})
        topics = plan_questions(p, "We need a website")
        self.assertLessEqual(len(topics), 2)
        self.assertEqual(topics, ["features", "timeline"])

    def test_one_question_when_customer_asked_something(self):
        self.assertEqual(len(plan_questions(profile(), "Do you build apps?")), 1)

    def test_support_asks_for_issue_details_first(self):
        self.assertEqual(plan_questions(profile("Support"), "Site is down")[0], "issue_details")

    def test_conflict_is_confirmed_first(self):
        from ai.schemas import FieldConflict
        p = profile(conflicts=[FieldConflict(field="Location", previous="Pune", new="Mumbai")])
        self.assertEqual(plan_questions(p, "Our office is in Mumbai"), ["confirm_conflict"])

    def test_contact_known_from_whatsapp_number(self):
        p = profile(interested_service=tf("website"), requested_features={"values": ["blog"], "confidence": 0.9},
                    budget={"raw_text": "50k", "amount_min": 50000, "amount_max": 50000, "confidence": 0.9},
                    timeline=tf("1 month"), company_name=tf("Acme"), customer_name=tf("Priya"))
        self.assertEqual(plan_questions(p, "ok", contact_known=True), ["decision"])
        self.assertEqual(plan_questions(p, "ok", contact_known=False), ["contact", "decision"])

    def test_backs_off_after_two_unanswered_asks(self):
        p = profile(interested_service=tf("website"))
        self.assertEqual(plan_questions(p, "ok", recently_asked=[["features"], ["features", "budget"]]),
                         ["budget", "timeline"])
        # Asked only once recently: still asked.
        self.assertEqual(plan_questions(p, "ok", recently_asked=[["budget"], ["features"]])[0], "features")

    def test_nothing_to_ask_when_everything_known(self):
        p = profile(interested_service=tf("website"), requested_features={"values": ["blog"], "confidence": 0.9},
                    budget={"raw_text": "50k", "amount_min": 50000, "amount_max": 50000, "confidence": 0.9},
                    timeline=tf("1 month"), company_name=tf("Acme"), customer_name=tf("Priya"),
                    email=tf("p@acme.in"), decision_authority={"value": "yes", "confidence": 0.9})
        self.assertEqual(plan_questions(p, "ok"), [])


class ReplyValidationTests(unittest.TestCase):
    def test_rules(self):
        empty = BusinessProfile()
        self.assertTrue(validate_reply("A website costs ₹50,000.", "how much?", empty))
        self.assertTrue(validate_reply("Plans from $5", "call 98765 43215", empty))
        self.assertTrue(validate_reply("Packages cost 2 lakh", "budget 50k", empty))
        self.assertEqual(validate_reply("Your 1.5 lakh budget is noted", "budget 1.5 lakh", empty), [])
        self.assertEqual(validate_reply("Noted ₹50,000", "budget 50k", empty), [])
        self.assertEqual(validate_reply("We can start in 2-3 weeks.", "hi", empty), [])
        priced = BusinessProfile(pricing={"Website Development": "Starts at ₹40,000"})
        self.assertEqual(validate_reply("Websites start at ₹40,000.", "price?", priced), [])
        self.assertTrue(validate_reply("You are a hot lead", "hi", empty))
        self.assertTrue(validate_reply("", "hi", empty))


if __name__ == "__main__":
    unittest.main()
