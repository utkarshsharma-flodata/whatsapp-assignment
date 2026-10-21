"""Unit tests for the deterministic lead scoring engine.

Run:  python -m unittest discover tests
"""

import subprocess
import sys
import unittest

from pydantic import ValidationError

from ai.nlp_service import analyze_conversation
from ai.schemas import Entities, IntentResult, PurchaseIntent
from scoring import CRITERIA_MAX, LeadScore, classify, score_lead
from scoring.engine import timeline_days
from tests.test_nlp_service import SETTINGS, FakeClient, extraction, user


def tf(value, evidence=None, confidence=0.95, **kw):
    return {"value": value, "evidence": evidence or value, "confidence": confidence, **kw}


def intent(category="Website Development", confidence=0.9):
    return IntentResult(category=category, confidence=confidence)


def purchase(level="unknown", confidence=0.8):
    return PurchaseIntent(level=level, confidence=confidence)


def by_key(score: LeadScore) -> dict:
    return {c.key: c.earned for c in score.criteria}


HIGH_INTENT_MESSAGES = [
    "Hi, I'm Priya, owner of Bloom Bakery, a bakery chain in Bengaluru with 4 outlets.",
    "We need an e-commerce website for online cake orders with online ordering, payment gateway "
    "and delivery tracking. Budget is ₹1.5 lakh and we need it within 2 weeks. "
    "Can you send a quote? Email priya@bloombakery.in. I make the final decision.",
]
HIGH_INTENT_ENTITIES = Entities.model_validate(
    {
        "customer_name": tf("Priya", "I'm Priya"),
        "email": tf("priya@bloombakery.in"),
        "company_name": tf("Bloom Bakery"),
        "industry": tf("Bakery", "a bakery chain"),
        "location": tf("Bengaluru"),
        "business_size": tf("4 outlets"),
        "interested_service": tf("e-commerce website"),
        "requirement_description": tf("E-commerce website for online cake orders",
                                      "e-commerce website for online cake orders"),
        "requested_features": {"values": ["Online ordering", "Payment gateway", "Delivery tracking"],
                               "confidence": 0.95},
        "budget": {"amount_min": 150000, "amount_max": 150000, "currency": "INR",
                   "raw_text": "₹1.5 lakh", "confidence": 0.95},
        "timeline": tf("within 2 weeks"),
        "decision_authority": {"value": "yes", "confidence": 0.95, "evidence": "I make the final decision"},
        "questions": {"values": ["Can you send a quote?"], "confidence": 0.9},
        "engagement_indicators": {"values": ["Requested a quote", "Shared email"], "confidence": 0.9},
    }
)


class HighIntentTests(unittest.TestCase):
    def test_high_intent_customer_is_hot_with_full_points(self):
        score = score_lead(HIGH_INTENT_ENTITIES, intent(), purchase("high", 0.9), HIGH_INTENT_MESSAGES)
        self.assertEqual(score.category, "HOT")
        points = by_key(score)
        self.assertEqual(points["budget"], 20)
        self.assertEqual(points["timeline"], 15)
        self.assertEqual(points["purchase_intent"], 20)
        self.assertEqual(points["requirement_clarity"], 20)
        self.assertEqual(points["decision_authority"], 5)
        self.assertEqual(points["business_information"], 10)
        self.assertEqual(points["engagement"], 2 + 2 + 1 + 2 + 2)  # 2 messages
        self.assertEqual(score.total, 99)
        self.assertNotIn("Budget (amount and currency)", score.missing_information)

    def test_every_criterion_has_a_reason_and_evidence_level(self):
        score = score_lead(HIGH_INTENT_ENTITIES, intent(), purchase("high", 0.9), HIGH_INTENT_MESSAGES)
        self.assertEqual([c.key for c in score.criteria], list(CRITERIA_MAX))
        for c in score.criteria:
            self.assertTrue(c.reason)
            self.assertIn(c.evidence, ("explicit", "inferred", "missing"))

    def test_scoring_is_deterministic(self):
        a = score_lead(HIGH_INTENT_ENTITIES, intent(), purchase("high", 0.9), HIGH_INTENT_MESSAGES)
        b = score_lead(HIGH_INTENT_ENTITIES, intent(), purchase("high", 0.9), HIGH_INTENT_MESSAGES)
        self.assertEqual(a, b)


class LowInformationTests(unittest.TestCase):
    def test_price_checking_customer_is_cold(self):
        messages = ["Hi, what are your packages and price range? Just want a rough idea for now."]
        entities = Entities.model_validate(
            {"questions": {"values": ["What are your packages and price range?"], "confidence": 0.9}}
        )
        score = score_lead(entities, intent("Pricing Enquiry"), purchase("low", 0.8), messages)
        self.assertEqual(score.category, "COLD")
        points = by_key(score)
        self.assertEqual(points["requirement_clarity"], 0)  # Pricing Enquiry is not a service
        self.assertEqual(points["budget"], 0)
        self.assertEqual(points["purchase_intent"], 4)
        self.assertEqual(points["engagement"], 1 + 2)  # one message + asked a question
        self.assertEqual(score.total, 7)
        self.assertIn("Budget (amount and currency)", score.missing_information)
        self.assertIn("Expected timeline", score.missing_information)

    def test_very_little_information(self):
        score = score_lead(Entities(), intent("General Enquiry", 0.6), purchase("unknown", 0.0), ["Hi"])
        self.assertEqual(score.total, 1)  # only engagement: one message
        self.assertEqual(score.category, "COLD")
        self.assertEqual(sum(1 for c in score.criteria if c.evidence == "missing"), 6)

    def test_no_messages_scores_zero(self):
        score = score_lead(Entities(), intent("General Enquiry", 0.5), purchase(), [])
        self.assertEqual(score.total, 0)
        self.assertEqual(score.category, "COLD")


class GuessedInformationTests(unittest.TestCase):
    """Values not supported by the customer's own words never earn full points."""

    messages = ["We need a website for our shop."]

    def test_budget_not_in_messages_is_inferred(self):
        entities = Entities.model_validate(
            {"budget": {"amount_min": 50000, "amount_max": 50000, "currency": "INR",
                        "raw_text": "50k", "confidence": 0.95}}
        )
        criterion = next(c for c in score_lead(entities, intent(), purchase(), self.messages).criteria
                         if c.key == "budget")
        self.assertEqual((criterion.earned, criterion.evidence), (8, "inferred"))

    def test_low_confidence_budget_is_inferred(self):
        messages = ["Budget maybe 50k"]
        entities = Entities.model_validate(
            {"budget": {"amount_min": 50000, "amount_max": 50000, "currency": "INR",
                        "raw_text": "50k", "confidence": 0.5}}
        )
        self.assertEqual(by_key(score_lead(entities, intent(), purchase(), messages))["budget"], 8)

    def test_budget_without_currency(self):
        messages = ["Our budget is 50000"]
        entities = Entities.model_validate(
            {"budget": {"amount_min": 50000, "amount_max": 50000, "raw_text": "50000", "confidence": 0.9}}
        )
        self.assertEqual(by_key(score_lead(entities, intent(), purchase(), messages))["budget"], 16)

    def test_timeline_not_in_messages_is_inferred(self):
        entities = Entities.model_validate({"timeline": tf("1 month")})
        self.assertEqual(by_key(score_lead(entities, intent(), purchase(), self.messages))["timeline"], 5)

    def test_decision_authority_without_evidence_is_partial(self):
        entities = Entities.model_validate(
            {"decision_authority": {"value": "yes", "confidence": 0.9, "evidence": "I am the owner"}}
        )
        self.assertEqual(by_key(score_lead(entities, intent(), purchase(), self.messages))["decision_authority"], 3)

    def test_uncorroborated_high_intent_is_capped(self):
        score = score_lead(Entities(), intent(), purchase("high", 0.95), self.messages)
        self.assertEqual(by_key(score)["purchase_intent"], 14)

    def test_specific_intent_without_stated_service_gets_partial_points(self):
        score = score_lead(Entities(), intent("Website Development", 0.9), purchase(), self.messages)
        self.assertEqual(by_key(score)["requirement_clarity"], 3)


class ProgressiveConversationTests(unittest.TestCase):
    """Score is recalculated as the customer adds information in later messages."""

    def test_score_increases_when_customer_adds_details(self):
        first_messages = [user("Hi, I need a website for my clinic")]
        first = analyze_conversation(
            first_messages, settings=SETTINGS,
            client=FakeClient(extraction(
                "Website Development", "medium",
                interested_service=tf("website", "I need a website"),
                industry=tf("Clinic", "my clinic"),
            )),
        )
        first_score = score_lead(first.entities, first.intent, first.purchase_intent,
                                 [m["content"] for m in first_messages])

        second_messages = first_messages + [user(
            "I'm Dr. Mehta from Smile Dental, Pune. Budget is $2000 and we need it in 1 month. "
            "I'm the owner. Email: mehta@smiledental.com"
        )]
        second = analyze_conversation(
            second_messages, previous=first, settings=SETTINGS,
            client=FakeClient(extraction(
                "Website Development", "high",
                customer_name=tf("Dr. Mehta"),
                company_name=tf("Smile Dental"),
                location=tf("Pune"),
                email=tf("mehta@smiledental.com"),
                budget={"amount_min": 2000, "amount_max": 2000, "currency": "USD",
                        "raw_text": "$2000", "confidence": 0.95},
                timeline=tf("1 month", "in 1 month"),
                decision_authority={"value": "yes", "confidence": 0.9, "evidence": "I'm the owner"},
            )),
        )
        second_score = score_lead(second.entities, second.intent, second.purchase_intent,
                                  [m["content"] for m in second_messages])

        self.assertEqual(first_score.category, "COLD")
        self.assertGreater(second_score.total, first_score.total)
        before, after = by_key(first_score), by_key(second_score)
        self.assertEqual((before["budget"], after["budget"]), (0, 20))
        self.assertEqual((before["timeline"], after["timeline"]), (0, 15))
        self.assertEqual(after["decision_authority"], 5)
        # Information from the first message is preserved by the merge.
        self.assertEqual(second.entities.industry.value, "Clinic")
        self.assertNotIn("Budget (amount and currency)", second_score.missing_information)
        self.assertIn("Budget (amount and currency)", first_score.missing_information)


class BoundaryAndConstraintTests(unittest.TestCase):
    def test_category_boundaries(self):
        cases = {0: "COLD", 49: "COLD", 50: "WARM", 79: "WARM", 80: "HOT", 100: "HOT"}
        for total, expected in cases.items():
            self.assertEqual(classify(total), expected, total)

    def test_weights_total_100(self):
        self.assertEqual(CRITERIA_MAX, {
            "requirement_clarity": 20, "budget": 20, "timeline": 15, "purchase_intent": 20,
            "business_information": 10, "engagement": 10, "decision_authority": 5,
        })
        self.assertEqual(sum(CRITERIA_MAX.values()), 100)

    def test_criteria_never_exceed_maximum(self):
        messages = [f"Message {i}? My email is a@b.co" for i in range(20)]
        entities = HIGH_INTENT_ENTITIES.model_copy(deep=True)
        entities.requested_features.values = [f"Feature {i}" for i in range(15)]
        entities.engagement_indicators.values = [f"Signal {i}" for i in range(10)]
        entities.questions.values = [f"Q{i}" for i in range(10)]
        score = score_lead(entities, intent(), purchase("high", 1.0), messages + HIGH_INTENT_MESSAGES)
        for c in score.criteria:
            self.assertLessEqual(c.earned, c.maximum, c.key)
        self.assertEqual(by_key(score)["engagement"], 10)
        self.assertEqual(by_key(score)["requirement_clarity"], 20)
        self.assertEqual(score.total, 100)
        self.assertEqual(score.category, "HOT")

    def _criteria(self, **overrides):
        earned = {k: 0 for k in CRITERIA_MAX} | overrides
        return [{"key": k, "label": k, "earned": v, "maximum": CRITERIA_MAX[k], "reason": "x",
                 "evidence": "missing"} for k, v in earned.items()]

    def test_valid_score_model(self):
        LeadScore(total=50, category="WARM", criteria=self._criteria(budget=20, timeline=15, purchase_intent=15))

    def test_rejects_sum_mismatch(self):
        with self.assertRaisesRegex(ValidationError, "Sum of criterion scores"):
            LeadScore(total=30, category="COLD", criteria=self._criteria(budget=20))

    def test_rejects_total_out_of_range(self):
        with self.assertRaises(ValidationError):
            LeadScore(total=101, category="HOT", criteria=self._criteria())

    def test_rejects_wrong_category(self):
        with self.assertRaisesRegex(ValidationError, "does not match"):
            LeadScore(total=20, category="WARM", criteria=self._criteria(budget=20))

    def test_rejects_earned_above_maximum(self):
        with self.assertRaisesRegex(ValidationError, "exceeds maximum"):
            LeadScore(total=6, category="COLD", criteria=self._criteria(decision_authority=6))

    def test_rejects_missing_criterion(self):
        criteria = self._criteria()[:-1]
        with self.assertRaisesRegex(ValidationError, "Criteria must be exactly"):
            LeadScore(total=0, category="COLD", criteria=criteria)

    def test_rejects_non_integer_total(self):
        with self.assertRaises(ValidationError):
            LeadScore(total=50.5, category="WARM", criteria=self._criteria())


class TimelineParsingTests(unittest.TestCase):
    def test_durations(self):
        cases = {
            "ASAP": 0, "urgent": 0, "this week": 0, "within 2 weeks": 14, "6 weeks": 42,
            "1 month": 30, "a month": 30, "2-3 months": 90, "next month": 45,
            "one year": 365, "1.5 months": 45, "after Diwali": None, "Q3": None,
        }
        for text, expected in cases.items():
            self.assertEqual(timeline_days(text), expected, text)

    def test_timeline_bands(self):
        def points(value):
            entities = Entities.model_validate({"timeline": tf(value)})
            return by_key(score_lead(entities, intent(), purchase(), [f"Timeline: {value}"]))["timeline"]

        self.assertEqual(points("ASAP"), 15)
        self.assertEqual(points("within 30 days"), 15)
        self.assertEqual(points("2 months"), 12)
        self.assertEqual(points("6 months"), 8)
        self.assertEqual(points("after Diwali"), 8)
        self.assertEqual(points("no rush"), 4)


class IndependenceTests(unittest.TestCase):
    def test_scoring_does_not_import_streamlit_or_groq(self):
        code = "import sys, scoring; print(sorted(m for m in ('streamlit', 'groq') if m in sys.modules))"
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
        self.assertEqual(out.stdout.strip(), "[]")


if __name__ == "__main__":
    unittest.main()
