"""Validated result types for the lead scoring engine."""

from typing import Literal

from pydantic import BaseModel, Field, model_validator

Category = Literal["HOT", "WARM", "COLD"]
EvidenceLevel = Literal["explicit", "inferred", "missing"]

HOT_THRESHOLD = 80
WARM_THRESHOLD = 50

# Criterion key -> maximum points. Must total 100.
CRITERIA_MAX = {
    "requirement_clarity": 20,
    "budget": 20,
    "timeline": 15,
    "purchase_intent": 20,
    "business_information": 10,
    "engagement": 10,
    "decision_authority": 5,
}
CRITERIA_LABELS = {
    "requirement_clarity": "Requirement clarity",
    "budget": "Budget",
    "timeline": "Timeline",
    "purchase_intent": "Purchase intent",
    "business_information": "Business information",
    "engagement": "Engagement",
    "decision_authority": "Decision-making authority",
}
assert sum(CRITERIA_MAX.values()) == 100


def classify(total: int) -> Category:
    if total >= HOT_THRESHOLD:
        return "HOT"
    if total >= WARM_THRESHOLD:
        return "WARM"
    return "COLD"


class CriterionScore(BaseModel):
    key: str
    label: str
    earned: int = Field(ge=0)
    maximum: int = Field(gt=0)
    reason: str
    evidence: EvidenceLevel

    @model_validator(mode="after")
    def _within_maximum(self):
        if self.key not in CRITERIA_MAX:
            raise ValueError(f"Unknown criterion '{self.key}'")
        if self.maximum != CRITERIA_MAX[self.key]:
            raise ValueError(f"{self.key}: maximum must be {CRITERIA_MAX[self.key]}")
        if self.earned > self.maximum:
            raise ValueError(f"{self.key}: earned {self.earned} exceeds maximum {self.maximum}")
        return self


class LeadScore(BaseModel):
    total: int = Field(ge=0, le=100)
    category: Category
    criteria: list[CriterionScore]
    missing_information: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _consistent(self):
        keys = [c.key for c in self.criteria]
        if sorted(keys) != sorted(CRITERIA_MAX):
            raise ValueError(f"Criteria must be exactly {list(CRITERIA_MAX)}, got {keys}")
        if sum(c.earned for c in self.criteria) != self.total:
            raise ValueError("Sum of criterion scores does not equal the total")
        if classify(self.total) != self.category:
            raise ValueError(f"Category {self.category} does not match total {self.total}")
        return self
