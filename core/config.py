"""Application settings loaded from environment variables (.env).

Invalid values never crash the app: each one falls back to a safe default
and is reported in ``Settings.config_warnings``, which the UI displays.
"""

import os
import re
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REASONING_FORMATS = ("hidden", "raw", "parsed")


@dataclass(frozen=True)
class Settings:
    app_title: str
    groq_api_key: str | None
    # No default: the exact model identifier must come from .env.
    groq_model: str | None
    groq_temperature: float
    # "hidden" strips the reasoning trace of reasoning models (e.g. Qwen3);
    # leave unset for models that do not support the parameter.
    groq_reasoning_format: str | None
    # Relative paths are resolved against the project root.
    database_path: str
    # Prepended to 10-digit local numbers when normalising phone numbers.
    default_country_code: str = "91"
    # Temperature for customer replies (extraction always uses groq_temperature).
    groq_reply_temperature: float = 0.4
    # Optional JSON file with business details and configured prices.
    business_profile_path: str = ""
    # Human-readable problems found while reading the environment.
    config_warnings: tuple[str, ...] = field(default=(), compare=False)

    @property
    def ai_configured(self) -> bool:
        return bool(self.groq_api_key and self.groq_model)

    def __repr__(self) -> str:  # never print the API key
        key = "set" if self.groq_api_key else "not set"
        return f"Settings(model={self.groq_model!r}, api_key={key}, database={self.database_path!r})"


def _float(name: str, default: float, low: float, high: float, warnings: list[str]) -> float:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError:
        warnings.append(f"{name}='{raw}' is not a number; using {default}.")
        return default
    if not low <= value <= high:
        warnings.append(f"{name}={raw} is outside {low}-{high}; using {default}.")
        return default
    return value


def get_settings() -> Settings:
    """Read settings fresh from .env, so edits apply without restarting the app."""
    load_dotenv(override=True)
    warnings: list[str] = []

    reasoning = os.getenv("GROQ_REASONING_FORMAT", "").strip().lower() or None
    if reasoning and reasoning not in REASONING_FORMATS:
        warnings.append(f"GROQ_REASONING_FORMAT='{reasoning}' is not one of {', '.join(REASONING_FORMATS)}; "
                        "leaving it unset.")
        reasoning = None

    country = os.getenv("DEFAULT_COUNTRY_CODE", "91").strip().lstrip("+") or "91"
    if not re.fullmatch(r"\d{1,3}", country):
        warnings.append(f"DEFAULT_COUNTRY_CODE='{country}' must be 1-3 digits (e.g. 91); using 91.")
        country = "91"

    api_key = os.getenv("GROQ_API_KEY", "").strip() or None
    if api_key and (any(c.isspace() for c in api_key) or len(api_key) < 20):
        warnings.append("GROQ_API_KEY looks malformed (too short or contains spaces). Check .env.")

    return Settings(
        app_title=os.getenv("APP_TITLE", "").strip() or "Lead Assistant",
        groq_api_key=api_key,
        groq_model=os.getenv("GROQ_MODEL", "").strip() or None,
        groq_temperature=_float("GROQ_TEMPERATURE", 0.0, 0.0, 2.0, warnings),
        groq_reasoning_format=reasoning,
        database_path=_resolve_path(os.getenv("DATABASE_PATH", "").strip() or "data/leads.db"),
        default_country_code=country,
        groq_reply_temperature=_float("GROQ_REPLY_TEMPERATURE", 0.4, 0.0, 2.0, warnings),
        business_profile_path=_resolve_path(os.getenv("BUSINESS_PROFILE_PATH", "").strip() or "business_profile.json"),
        config_warnings=tuple(warnings),
    )


def _resolve_path(path: str) -> str:
    if path == ":memory:":
        return path
    p = Path(path)
    return str(p if p.is_absolute() else PROJECT_ROOT / p)
