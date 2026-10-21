"""Errors raised by the NLP layer. Each message is safe to show in the UI."""


class NLPError(Exception):
    """Base class for all NLP failures."""


class ConfigurationError(NLPError):
    """GROQ_API_KEY or GROQ_MODEL is missing or invalid."""


class LLMRequestError(NLPError):
    """The Groq API request failed (auth, model, rate limit, network...)."""


class ExtractionValidationError(NLPError):
    """The model's output could not be validated against the schema."""


class ResponseValidationError(NLPError):
    """The generated customer reply broke a safety rule (e.g. invented a price)."""
