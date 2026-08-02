"""Structured logging compatibility for minimal local/test environments."""

from __future__ import annotations

import logging
import re


_SECRET_KEYS = re.compile(r"(?i)(api[_-]?key|token|authorization|password|base64|image[_-]?data)")


def redact_sensitive(value, *, max_length: int = 500):
    """Return a bounded log-safe representation without secrets or image data."""
    if isinstance(value, dict):
        return {
            key: "[REDACTED]" if _SECRET_KEYS.search(str(key)) else redact_sensitive(item, max_length=max_length)
            for key, item in value.items()
        }
    text = str(value)
    text = re.sub(r"(?i)(sk-[A-Za-z0-9_-]{8,})", "[REDACTED_KEY]", text)
    return text[:max_length]


class _FallbackLogger:
    def __init__(self, name: str):
        self._logger = logging.getLogger(name)

    def _write(self, level: int, event: str, **context) -> None:
        message = event if not context else f"{event} {context}"
        self._logger.log(level, message)

    def debug(self, event: str, **context) -> None:
        self._write(logging.DEBUG, event, **context)

    def info(self, event: str, **context) -> None:
        self._write(logging.INFO, event, **context)

    def warning(self, event: str, **context) -> None:
        self._write(logging.WARNING, event, **context)

    def error(self, event: str, **context) -> None:
        self._write(logging.ERROR, event, **context)


try:
    import structlog as structlog
except ImportError:
    class _StructlogFallback:
        @staticmethod
        def get_logger(name: str | None = None) -> _FallbackLogger:
            return _FallbackLogger(name or "tuxun")

    structlog = _StructlogFallback()
