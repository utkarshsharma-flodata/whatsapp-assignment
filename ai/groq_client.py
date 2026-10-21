"""Thin wrapper around the Groq SDK that converts failures into NLPErrors."""

import re
import time

import groq

from ai.errors import ConfigurationError, ExtractionValidationError, LLMRequestError, NLPError
from core.config import Settings
from core.logging_setup import get_logger

log = get_logger("groq")

_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL)
_CODE_FENCE = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.DOTALL | re.IGNORECASE)


def create_client(settings: Settings) -> groq.Groq:
    if not settings.groq_api_key:
        raise ConfigurationError("GROQ_API_KEY is not set. Add it to your .env file.")
    if not settings.groq_model:
        raise ConfigurationError(
            "GROQ_MODEL is not set. Add the exact Groq model identifier to your .env file."
        )
    return groq.Groq(api_key=settings.groq_api_key, timeout=60.0, max_retries=2)


def complete_json(client: groq.Groq, settings: Settings, messages: list[dict]) -> str:
    """Request a JSON-object completion and return the raw JSON text."""
    text = _complete(client, settings, {
        "messages": messages,
        "response_format": {"type": "json_object"},
        "temperature": settings.groq_temperature,
        "max_completion_tokens": 4096,
    }, kind="json")
    # Some models wrap JSON in a markdown code fence despite JSON mode.
    fenced = _CODE_FENCE.match(text)
    return fenced.group(1) if fenced else text


def complete_text(client: groq.Groq, settings: Settings, messages: list[dict]) -> str:
    """Request a plain-text completion (used for customer replies)."""
    return _complete(client, settings, {
        "messages": messages,
        "temperature": settings.groq_reply_temperature,
        "max_completion_tokens": 2048,
    }, kind="text")


def _complete(client: groq.Groq, settings: Settings, params: dict, kind: str) -> str:
    params = {"model": settings.groq_model, **params}
    if settings.groq_reasoning_format:
        params["reasoning_format"] = settings.groq_reasoning_format
    started = time.perf_counter()
    try:
        response = _request(client, params, settings)
    except NLPError as exc:
        log.warning("Groq %s request failed after %.1fs: %s (%s)", kind, time.perf_counter() - started,
                    type(exc).__name__, exc)
        raise
    usage = getattr(response, "usage", None)
    log.info("Groq %s request ok in %.1fs (model=%s, tokens=%s)", kind, time.perf_counter() - started,
             settings.groq_model, getattr(usage, "total_tokens", "n/a"))
    if not response.choices:
        raise LLMRequestError("Groq returned an empty response (no choices).")
    content = response.choices[0].message.content or ""
    return _THINK_BLOCK.sub("", content).strip()


def _request(client: groq.Groq, params: dict, settings: Settings):
    """Send the request, mapping SDK exceptions to NLPErrors with actionable messages."""
    try:
        response = client.chat.completions.create(**params)
    except groq.AuthenticationError as exc:
        raise LLMRequestError("Groq rejected the API key (401). Check GROQ_API_KEY in .env.") from exc
    except groq.PermissionDeniedError as exc:
        raise LLMRequestError(
            f"Access to model '{settings.groq_model}' was denied (403). Check your Groq account permissions."
        ) from exc
    except groq.NotFoundError as exc:
        raise LLMRequestError(
            f"Model '{settings.groq_model}' was not found on Groq (404). Check GROQ_MODEL in .env."
        ) from exc
    except groq.RateLimitError as exc:
        raise LLMRequestError("Groq rate limit reached (429). Wait a moment and retry.") from exc
    except groq.APITimeoutError as exc:
        raise LLMRequestError("The Groq request timed out. Check your connection and retry.") from exc
    except groq.APIConnectionError as exc:
        raise LLMRequestError("Could not connect to the Groq API. Check your internet connection.") from exc
    except groq.BadRequestError as exc:
        detail = _error_detail(exc)
        if "json_validate_failed" in detail:
            # Groq refused because the model produced invalid JSON; the caller retries.
            raise ExtractionValidationError(f"Model produced invalid JSON: {detail}") from exc
        hint = ""
        if "reasoning_format" in detail:
            hint = " This model may not support reasoning_format: clear GROQ_REASONING_FORMAT in .env."
        raise LLMRequestError(f"Groq rejected the request (400): {detail}.{hint}") from exc
    except groq.APIStatusError as exc:
        raise LLMRequestError(f"Groq API error ({exc.status_code}): {_error_detail(exc)}") from exc
    except groq.APIError as exc:
        raise LLMRequestError(f"Groq API error: {exc}") from exc
    return response


def _error_detail(exc: groq.APIStatusError) -> str:
    body = exc.body
    if isinstance(body, dict):
        err = body.get("error", body)
        if isinstance(err, dict):
            parts = [str(err.get(k)) for k in ("code", "message") if err.get(k)]
            if parts:
                return " — ".join(parts)
    return str(exc)
