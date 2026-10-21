"""Business details the assistant may share with customers.

Loaded from an optional JSON file (BUSINESS_PROFILE_PATH, default
``business_profile.json``). Anything not configured is unknown to the
assistant, which then says the team will confirm instead of inventing it.
"""

import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, ValidationError


class BusinessProfile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str | None = None
    description: str | None = None
    office_location: str | None = None
    working_hours: str | None = None
    contact_email: str | None = None
    support_contact: str | None = None
    # Service name -> price text exactly as it may be quoted, e.g. "Starts at ₹40,000".
    pricing: dict[str, str] = Field(default_factory=dict)


def load_business_profile(path: str | None) -> BusinessProfile:
    """Return the configured profile, or an empty one if the file does not exist."""
    if not path or not Path(path).exists():
        return BusinessProfile()
    try:
        return BusinessProfile.model_validate(json.loads(Path(path).read_text(encoding="utf-8")))
    except (json.JSONDecodeError, ValidationError) as exc:
        raise ValueError(f"Invalid business profile in {path}: {exc}") from exc
