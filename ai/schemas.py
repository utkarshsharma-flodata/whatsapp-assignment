"""Pydantic schemas for intent detection and entity extraction.

``LLMExtraction`` is the exact JSON shape the model must return.
``LeadAnalysis`` is what the application stores after validation and merging.
"""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

ServiceCategory = Literal[
    "Website Development",
    "Software Development",
    "Power BI Dashboard",
    "Data Analytics",
    "Business Automation",
    "Digital Marketing",
    "Graphic Design",
    "Cybersecurity",
    "SaaS Development",
    "Pricing Enquiry",
    "General Enquiry",
    "Support",
]
PurchaseIntentLevel = Literal["high", "medium", "low", "unknown"]
DecisionAuthority = Literal["yes", "no", "unknown"]
Confidence = Annotated[float, Field(ge=0.0, le=1.0)]

# Strings models commonly emit instead of a real null.
_NULL_STRINGS = {"", "null", "none", "n/a", "na", "unknown", "not provided", "not mentioned", "-"}


def _none_if_blank(value):
    if isinstance(value, str) and value.strip().lower() in _NULL_STRINGS:
        return None
    return value.strip() if isinstance(value, str) else value


class _Base(BaseModel):
    model_config = ConfigDict(extra="ignore")


class TextField(_Base):
    value: str | None = None
    confidence: Confidence = 0.0
    evidence: str | None = Field(None, description="Verbatim customer quote supporting the value")
    is_correction: bool = Field(False, description="True if the customer explicitly corrected an earlier value")

    _clean = field_validator("value", "evidence", mode="before")(_none_if_blank)

    @field_validator("confidence", mode="after")
    @classmethod
    def _zero_when_empty(cls, v, info):
        return v if info.data.get("value") is not None else 0.0


class ListField(_Base):
    values: list[str] = Field(default_factory=list)
    confidence: Confidence = 0.0

    @field_validator("values", mode="before")
    @classmethod
    def _clean_list(cls, v):
        if v is None:
            return []
        if isinstance(v, str):
            v = [v]
        return [s for s in (_none_if_blank(x) for x in v) if s]


class BudgetField(_Base):
    amount_min: float | None = None
    amount_max: float | None = None
    currency: str | None = Field(None, description="ISO 4217 code, e.g. INR, USD")
    raw_text: str | None = Field(None, description="Budget exactly as the customer stated it")
    confidence: Confidence = 0.0
    is_correction: bool = False

    _clean = field_validator("currency", "raw_text", mode="before")(_none_if_blank)

    @field_validator("currency", mode="after")
    @classmethod
    def _upper(cls, v):
        return v.upper() if v else v

    @property
    def is_known(self) -> bool:
        return self.raw_text is not None or self.amount_min is not None or self.amount_max is not None

    def display(self) -> str | None:
        if not self.is_known:
            return None
        if self.amount_min is not None or self.amount_max is not None:
            lo, hi = self.amount_min, self.amount_max
            amount = f"{lo:,.0f}–{hi:,.0f}" if lo is not None and hi is not None and lo != hi else f"{(lo if lo is not None else hi):,.0f}"
            return f"{self.currency + ' ' if self.currency else ''}{amount}"
        return self.raw_text


class DecisionAuthorityField(_Base):
    value: DecisionAuthority = "unknown"
    confidence: Confidence = 0.0
    evidence: str | None = None

    @field_validator("value", mode="before")
    @classmethod
    def _normalise(cls, v):
        v = _none_if_blank(v)
        return v.lower() if isinstance(v, str) else "unknown"


class IntentResult(_Base):
    category: ServiceCategory
    confidence: Confidence
    secondary_categories: list[ServiceCategory] = Field(default_factory=list)
    rationale: str | None = None


class PurchaseIntent(_Base):
    level: PurchaseIntentLevel = "unknown"
    confidence: Confidence = 0.0
    rationale: str | None = None

    @field_validator("level", mode="before")
    @classmethod
    def _normalise(cls, v):
        return v.lower() if isinstance(v, str) else "unknown"


class Entities(_Base):
    customer_name: TextField = Field(default_factory=TextField)
    whatsapp_number: TextField = Field(default_factory=TextField)
    company_name: TextField = Field(default_factory=TextField)
    email: TextField = Field(default_factory=TextField)
    location: TextField = Field(default_factory=TextField)
    industry: TextField = Field(default_factory=TextField)
    business_size: TextField = Field(default_factory=TextField)
    requirement_description: TextField = Field(default_factory=TextField)
    requested_features: ListField = Field(default_factory=ListField)
    interested_service: TextField = Field(default_factory=TextField)
    budget: BudgetField = Field(default_factory=BudgetField)
    timeline: TextField = Field(default_factory=TextField)
    lead_source: TextField = Field(default_factory=TextField)
    decision_authority: DecisionAuthorityField = Field(default_factory=DecisionAuthorityField)
    engagement_indicators: ListField = Field(default_factory=ListField)
    questions: ListField = Field(default_factory=ListField)
    objections: ListField = Field(default_factory=ListField)


class LLMExtraction(_Base):
    """Exact structure the LLM must return."""

    intent: IntentResult
    purchase_intent: PurchaseIntent
    entities: Entities


class FieldConflict(_Base):
    field: str
    previous: str
    new: str


class LeadAnalysis(_Base):
    """Validated, merged result stored in the session and shown in the UI."""

    intent: IntentResult
    purchase_intent: PurchaseIntent
    entities: Entities
    missing_information: list[str] = Field(default_factory=list)
    conflicts: list[FieldConflict] = Field(default_factory=list)
    corrections: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    model: str
    analysed_messages: int
