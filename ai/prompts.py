"""Prompt templates for intent detection and entity extraction."""

import json
from typing import get_args

from ai.schemas import LLMExtraction, ServiceCategory

_CATEGORIES = "\n".join(f"- {c}" for c in get_args(ServiceCategory))

SYSTEM_PROMPT = f"""You are an NLP extraction engine for a technology services company that receives sales enquiries on WhatsApp.
Analyse the conversation and return ONE JSON object that matches the JSON Schema below. Output JSON only: no prose, no markdown.

## Service intent (intent.category)
The service or purpose the customer is contacting about. Choose exactly one:
{_CATEGORIES}
Guidance:
- Use a specific service category whenever a service is identifiable, even if the customer also asks about price.
- "Pricing Enquiry" only when the customer asks about cost/packages without an identifiable service.
- "Support" for problems with an existing product/service or account they already have.
- "General Enquiry" for greetings, company questions, or anything without a clear service.
- List other relevant categories in intent.secondary_categories.

## Purchase intent (purchase_intent.level) — separate from the service category
How likely the customer is to buy: high | medium | low | unknown.
- high: clear need plus concrete signals (budget, timeline, asks to start/meet, requests a quote for a defined scope).
- medium: defined need but exploring, comparing, or missing budget/timeline.
- low: vague curiosity, "just checking", explicitly not buying soon.
- unknown: not enough information (e.g. a bare greeting or support issue).

## Entity rules
- Extract ONLY what the customer actually said. NEVER invent, guess or infer names, numbers, emails, companies or budgets.
- If a field is not mentioned anywhere in the conversation, set value to null (lists: []) and confidence to 0.
- evidence: a short verbatim quote from the customer supporting the value.
- confidence: 0.0–1.0. Explicit statements ≥ 0.85; reasonable interpretation 0.5–0.8; weak hints < 0.5.
- Use the whole conversation, not only the latest message. Previously extracted values are provided; keep them unless the conversation changes them.
- is_correction = true ONLY if the customer explicitly corrects or updates an earlier value ("actually, my budget is...", "sorry, the email is..."). Otherwise false.
- whatsapp_number / email: copy exactly as written by the customer.
- budget: numeric amount_min/amount_max (equal if a single figure; expand "50k" to 50000, "2 lakh" to 200000), ISO currency code only if stated or unambiguous from a symbol (₹→INR, $→USD, €→EUR, £→GBP), raw_text as written.
- interested_service: the specific product/service in the customer's words (e.g. "e-commerce website").
- requested_features: specific modules/features/deliverables asked for.
- business_size: employees, revenue, number of outlets, etc. only if mentioned.
- lead_source: only if the customer says how they found the company (e.g. "saw your Instagram ad").
- decision_authority: "yes" if they say they own/decide/approve; "no" if someone else decides; otherwise "unknown".
- engagement_indicators: observable buying signals (asked for a quote, shared contact details, asked for a meeting, gave timeline, replied in detail).
- questions: questions the customer asked. objections: concerns or hesitations (price too high, trust, timing).

## JSON Schema
{json.dumps(LLMExtraction.model_json_schema(), separators=(",", ":"))}
"""


def build_user_prompt(transcript: str, previous_entities: dict | None) -> str:
    previous = json.dumps(previous_entities, ensure_ascii=False) if previous_entities else "none"
    return (
        f"Previously extracted entities (from earlier messages):\n{previous}\n\n"
        f"Conversation (oldest first; the last customer message is the newest):\n{transcript}\n\n"
        "Lines starting with 'Business:' are our own replies. Use them only to understand what the customer "
        "is answering (e.g. a bare company name after we asked for it); never extract customer details from them.\n\n"
        "Return the JSON object now."
    )


def build_retry_prompt(error: str) -> str:
    return (
        "Your previous output failed schema validation:\n"
        f"{error}\n\n"
        "Return the corrected JSON object only, matching the schema exactly."
    )
