"""Live extraction and scoring check against the Groq model configured in .env.

Runs example conversations (website, Power BI, digital marketing, pricing,
general enquiry, support) through the real NLP service and prints the
results. Uses your Groq quota: one or two requests per message.

Run from the project root:  python -m scripts.test_extraction_live
"""

import sys

from ai.errors import NLPError
from ai.nlp_service import analyze_conversation
from scoring import score_lead

EXAMPLES = [
    {
        "name": "Website development",
        "expected": "Website Development",
        "messages": [
            "Hi, I'm Priya from Bloom Bakery in Bengaluru.",
            "We need an e-commerce website with online ordering, payment gateway and delivery tracking. "
            "Budget is around ₹1.5 lakh and we want to launch within 6 weeks. I'm the owner. "
            "Email: priya@bloombakery.in",
        ],
    },
    {
        "name": "Power BI dashboard",
        "expected": "Power BI Dashboard",
        "messages": [
            "Hello, I head finance at Shree Logistics, about 200 employees.",
            "We want a Power BI dashboard pulling data from SQL Server and Excel for fleet costs and on-time delivery KPIs. "
            "My director will approve the budget. How long would something like this take?",
        ],
    },
    {
        "name": "Digital marketing",
        "expected": "Digital Marketing",
        "messages": [
            "Saw your Instagram ad. We run a dental clinic chain in Pune with 3 branches.",
            "Looking for SEO and Google Ads management to get more patient bookings. "
            "Our budget is $800 per month. Is there a minimum contract period?",
        ],
    },
    {
        "name": "Pricing enquiry",
        "expected": "Pricing Enquiry",
        "messages": ["Hi, what are your packages and price range? Just want a rough idea for now."],
    },
    {
        "name": "General enquiry",
        "expected": "General Enquiry",
        "messages": ["Hello! Where is your office located and what are your working hours?"],
    },
    {
        "name": "Support",
        "expected": "Support",
        "messages": [
            "Hi, this is Arjun from Nexa Retail. The website you built for us last year is showing a 500 error "
            "on the checkout page since this morning. Please help urgently, my number is +91 98450 12345."
        ],
    },
    {
        "name": "Conflict detection (multi-turn)",
        "expected": "Website Development",
        "messages": [
            "Hi, I'm Karan. We need a company website, budget 50k INR.",
            "Our office is in Delhi.",
            "Our office is in Gurgaon by the way, and timeline is 1 month.",
        ],
    },
]


def run_example(example: dict) -> bool:
    print(f"\n=== {example['name']} ===")
    messages, result = [], None
    for text in example["messages"]:
        messages.append({"role": "user", "content": text, "placeholder": False})
        result = analyze_conversation(messages, previous=result)

    ok = result.intent.category == example["expected"]
    print(f"Intent: {result.intent.category} ({result.intent.confidence:.0%})"
          f"  expected {example['expected']}  {'PASS' if ok else 'MISMATCH'}")
    print(f"Purchase intent: {result.purchase_intent.level} ({result.purchase_intent.confidence:.0%})")
    for field, value in result.entities.model_dump(exclude_defaults=True).items():
        print(f"  {field}: {value}")
    print(f"Missing: {', '.join(result.missing_information) or 'none'}")
    for c in result.conflicts:
        print(f"Conflict: {c.field}: '{c.previous}' vs '{c.new}'")
    for w in result.warnings:
        print(f"Warning: {w}")
    score = score_lead(result.entities, result.intent, result.purchase_intent, example["messages"])
    print(f"Score: {score.total}/100 {score.category}")
    for c in score.criteria:
        print(f"  {c.label}: {c.earned}/{c.maximum} [{c.evidence}] {c.reason}")
    return ok


def main() -> int:
    passed = 0
    for example in EXAMPLES:
        try:
            passed += run_example(example)
        except NLPError as exc:
            print(f"ERROR: {exc}")
            if "GROQ_API_KEY" in str(exc) or "GROQ_MODEL" in str(exc) or "API key" in str(exc):
                return 1
    print(f"\nIntent matched expected category in {passed}/{len(EXAMPLES)} examples.")
    return 0 if passed == len(EXAMPLES) else 1


if __name__ == "__main__":
    sys.exit(main())
