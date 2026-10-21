"""Offline tests for the NLP pipeline using a fake Groq client.

These verify validation, retries, merging, conflict detection and error
handling. They do not test extraction quality; for that run
scripts/test_extraction_live.py with a real GROQ_API_KEY.

Run:  python -m unittest discover tests
"""

import json
import unittest
from types import SimpleNamespace

import groq
import httpx

from ai.errors import ConfigurationError, ExtractionValidationError, LLMRequestError
from ai.nlp_service import analyze_conversation
from core.config import Settings

SETTINGS = Settings(
    app_title="Test",
    groq_api_key="test-key",
    groq_model="test-model",
    groq_temperature=0.0,
    groq_reasoning_format=None,
    database_path=":memory:",
)


def _response(content: str):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


class FakeClient:
    """Returns queued responses (str) or raises queued exceptions."""

    def __init__(self, *outputs):
        self.outputs = list(outputs)
        self.calls = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        out = self.outputs.pop(0)
        if isinstance(out, Exception):
            raise out
        return _response(out)


def extraction(category="Website Development", level="medium", **entities) -> str:
    return json.dumps(
        {
            "intent": {"category": category, "confidence": 0.9, "rationale": "test"},
            "purchase_intent": {"level": level, "confidence": 0.7},
            "entities": entities,
        }
    )


def user(text):
    return {"role": "user", "content": text, "placeholder": False}


def _status_error(cls, status, body=None):
    request = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    return cls("error", response=httpx.Response(status, request=request), body=body)


class AnalyzeConversationTests(unittest.TestCase):
    def test_valid_extraction_and_missing_information(self):
        client = FakeClient(
            extraction(
                customer_name={"value": "Rahul", "confidence": 0.95, "evidence": "I'm Rahul"},
                budget={"amount_min": 50000, "amount_max": 50000, "currency": "INR",
                        "raw_text": "50k", "confidence": 0.9},
            )
        )
        result = analyze_conversation([user("I'm Rahul, need a website, budget 50k")], settings=SETTINGS, client=client)

        self.assertEqual(result.intent.category, "Website Development")
        self.assertEqual(result.purchase_intent.level, "medium")
        self.assertEqual(result.entities.customer_name.value, "Rahul")
        self.assertEqual(result.entities.budget.display(), "INR 50,000")
        self.assertNotIn("Customer name", result.missing_information)
        self.assertIn("Timeline", result.missing_information)
        self.assertEqual(client.calls[0]["response_format"], {"type": "json_object"})
        self.assertEqual(client.calls[0]["model"], "test-model")

    def test_null_like_strings_become_none(self):
        client = FakeClient(extraction(company_name={"value": "unknown", "confidence": 0.5}))
        result = analyze_conversation([user("hello")], settings=SETTINGS, client=client)
        self.assertIsNone(result.entities.company_name.value)
        self.assertEqual(result.entities.company_name.confidence, 0.0)

    def test_invalid_output_is_retried_then_accepted(self):
        bad = extraction(category="Web Stuff")  # not an allowed category
        client = FakeClient(bad, extraction())
        result = analyze_conversation([user("need a site")], settings=SETTINGS, client=client)
        self.assertEqual(result.intent.category, "Website Development")
        self.assertEqual(len(client.calls), 2)
        self.assertIn("failed schema validation", client.calls[1]["messages"][-1]["content"])

    def test_invalid_output_twice_raises(self):
        client = FakeClient("not json", "{}")
        with self.assertRaises(ExtractionValidationError):
            analyze_conversation([user("hi")], settings=SETTINGS, client=client)

    def test_groq_json_validate_failed_is_retried(self):
        err = _status_error(groq.BadRequestError, 400, {"error": {"code": "json_validate_failed", "message": "bad"}})
        client = FakeClient(err, extraction())
        result = analyze_conversation([user("hi")], settings=SETTINGS, client=client)
        self.assertEqual(result.intent.category, "Website Development")

    def test_hallucinated_contact_details_are_discarded(self):
        client = FakeClient(
            extraction(
                email={"value": "rahul@example.com", "confidence": 0.9},
                whatsapp_number={"value": "+91 98765 43210", "confidence": 0.9},
            )
        )
        result = analyze_conversation([user("Hi, I need a website")], settings=SETTINGS, client=client)
        self.assertIsNone(result.entities.email.value)
        self.assertIsNone(result.entities.whatsapp_number.value)
        self.assertEqual(len(result.warnings), 2)

    def test_contact_details_present_in_text_are_kept(self):
        client = FakeClient(extraction(whatsapp_number={"value": "+91 98765 43210", "confidence": 0.9}))
        result = analyze_conversation([user("call me on 9876543210")], settings=SETTINGS, client=client)
        self.assertEqual(result.entities.whatsapp_number.value, "+91 98765 43210")

    def test_previous_values_preserved_when_missing_from_new_output(self):
        first = analyze_conversation(
            [user("I'm Rahul from Acme")], settings=SETTINGS,
            client=FakeClient(extraction(customer_name={"value": "Rahul", "confidence": 0.9},
                                         company_name={"value": "Acme", "confidence": 0.9})),
        )
        second = analyze_conversation(
            [user("I'm Rahul from Acme"), user("Need it in 2 months")], previous=first, settings=SETTINGS,
            client=FakeClient(extraction(timeline={"value": "2 months", "confidence": 0.9})),
        )
        self.assertEqual(second.entities.customer_name.value, "Rahul")
        self.assertEqual(second.entities.company_name.value, "Acme")
        self.assertEqual(second.entities.timeline.value, "2 months")
        self.assertEqual(second.conflicts, [])

    def test_conflict_is_flagged_not_overwritten(self):
        first = analyze_conversation(
            [user("We are in Pune")], settings=SETTINGS,
            client=FakeClient(extraction(location={"value": "Pune", "confidence": 0.9})),
        )
        second = analyze_conversation(
            [user("We are in Pune"), user("Our office is in Mumbai")], previous=first, settings=SETTINGS,
            client=FakeClient(extraction(location={"value": "Mumbai", "confidence": 0.9})),
        )
        self.assertEqual(second.entities.location.value, "Pune")
        self.assertEqual(len(second.conflicts), 1)
        self.assertEqual((second.conflicts[0].previous, second.conflicts[0].new), ("Pune", "Mumbai"))

    def test_explicit_correction_overwrites_and_resolves_conflict(self):
        first = analyze_conversation(
            [user("budget 50k")], settings=SETTINGS,
            client=FakeClient(extraction(budget={"amount_min": 50000, "amount_max": 50000,
                                                 "currency": "INR", "raw_text": "50k", "confidence": 0.9})),
        )
        second = analyze_conversation(
            [user("budget 50k"), user("sorry, budget is actually 80k")], previous=first, settings=SETTINGS,
            client=FakeClient(extraction(budget={"amount_min": 80000, "amount_max": 80000, "currency": "INR",
                                                 "raw_text": "80k", "confidence": 0.9, "is_correction": True})),
        )
        self.assertEqual(second.entities.budget.amount_min, 80000)
        self.assertEqual(second.conflicts, [])
        self.assertEqual(len(second.corrections), 1)

    def test_list_fields_accumulate_without_duplicates(self):
        first = analyze_conversation(
            [user("need login")], settings=SETTINGS,
            client=FakeClient(extraction(requested_features={"values": ["Login"], "confidence": 0.8})),
        )
        second = analyze_conversation(
            [user("need login"), user("and payments")], previous=first, settings=SETTINGS,
            client=FakeClient(extraction(requested_features={"values": ["login", "Payments"], "confidence": 0.8})),
        )
        self.assertEqual(second.entities.requested_features.values, ["Login", "Payments"])

    def test_placeholder_messages_are_excluded_from_transcript(self):
        client = FakeClient(extraction())
        messages = [user("hello"), {"role": "assistant", "content": "PLACEHOLDER", "placeholder": True}]
        analyze_conversation(messages, settings=SETTINGS, client=client)
        self.assertNotIn("PLACEHOLDER", client.calls[0]["messages"][1]["content"])


class ErrorHandlingTests(unittest.TestCase):
    def test_missing_api_key(self):
        settings = Settings(**{**SETTINGS.__dict__, "groq_api_key": None})
        with self.assertRaisesRegex(ConfigurationError, "GROQ_API_KEY"):
            analyze_conversation([user("hi")], settings=settings)

    def test_missing_model(self):
        settings = Settings(**{**SETTINGS.__dict__, "groq_model": None})
        with self.assertRaisesRegex(ConfigurationError, "GROQ_MODEL"):
            analyze_conversation([user("hi")], settings=settings)

    def test_auth_error(self):
        client = FakeClient(_status_error(groq.AuthenticationError, 401))
        with self.assertRaisesRegex(LLMRequestError, "API key"):
            analyze_conversation([user("hi")], settings=SETTINGS, client=client)

    def test_unknown_model_is_reported_not_substituted(self):
        client = FakeClient(_status_error(groq.NotFoundError, 404))
        with self.assertRaisesRegex(LLMRequestError, "test-model"):
            analyze_conversation([user("hi")], settings=SETTINGS, client=client)
        self.assertEqual(len(client.calls), 1)

    def test_rate_limit(self):
        client = FakeClient(_status_error(groq.RateLimitError, 429))
        with self.assertRaisesRegex(LLMRequestError, "rate limit"):
            analyze_conversation([user("hi")], settings=SETTINGS, client=client)


if __name__ == "__main__":
    unittest.main()
