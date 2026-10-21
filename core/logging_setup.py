"""Logging configuration.

Logs go to ``logs/app.log`` (rotating, 1 MB x 3) and warnings also to the
console. A filter masks API keys in every record, so secrets cannot leak
even if one ends up in an exception message. Log lines record ids, counts,
timings and error types, never customer message text, emails or phone numbers.
"""

import logging
import os
import re
from logging.handlers import RotatingFileHandler

from core.config import PROJECT_ROOT

LOGGER_NAME = "leadbot"
_SECRET = re.compile(r"gsk_[A-Za-z0-9]{8,}")


class RedactSecretsFilter(logging.Filter):
    """Mask Groq keys (``gsk_...``) and the configured key in log messages and arguments."""

    def filter(self, record: logging.LogRecord) -> bool:
        key = os.getenv("GROQ_API_KEY", "").strip()
        message = record.getMessage()
        redacted = _SECRET.sub("gsk_***", message)
        if key and len(key) >= 8:
            redacted = redacted.replace(key, "***")
        if redacted != message:
            record.msg, record.args = redacted, ()
        return True


def configure_logging() -> logging.Logger:
    """Configure the app logger once (safe to call on every Streamlit rerun)."""
    logger = logging.getLogger(LOGGER_NAME)
    if getattr(logger, "_configured", False):
        return logger
    level = getattr(logging, os.getenv("LOG_LEVEL", "INFO").strip().upper(), logging.INFO)
    logger.setLevel(level)
    logger.propagate = False
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")

    try:
        log_dir = PROJECT_ROOT / "logs"
        log_dir.mkdir(exist_ok=True)
        file_handler = RotatingFileHandler(log_dir / "app.log", maxBytes=1_000_000, backupCount=3, encoding="utf-8")
        file_handler.setFormatter(fmt)
        file_handler.addFilter(RedactSecretsFilter())
        logger.addHandler(file_handler)
    except OSError:
        pass  # read-only location: console logging only

    console = logging.StreamHandler()
    console.setLevel(logging.WARNING)
    console.setFormatter(fmt)
    console.addFilter(RedactSecretsFilter())
    logger.addHandler(console)
    logger._configured = True
    return logger


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(f"{LOGGER_NAME}.{name}")
