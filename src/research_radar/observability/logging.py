"""Secret-conscious structured JSON logging with run-level context."""

from __future__ import annotations

import contextvars
import json
import logging
import re
from datetime import UTC, datetime
from typing import Any

# Digest run context — set by the digest engine, read by the formatter.
current_run_id: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "current_run_id", default=None
)

_BEARER_PATTERN = re.compile(r"(?i)(bearer\s+)[A-Za-z0-9._~+/=-]+")
_TELEGRAM_BOT_URL_PATTERN = re.compile(r"(?i)(https://api\.telegram\.org/bot)[^/\s]+")
_SECRET_ASSIGNMENT_PATTERN = re.compile(
    r"(?i)((?:api[_-]?key|token|authorization|secret)\s*[=:]\s*)[^\s,;]+"
)
_RESERVED = {
    "args",
    "asctime",
    "created",
    "exc_info",
    "exc_text",
    "filename",
    "funcName",
    "levelname",
    "levelno",
    "lineno",
    "module",
    "msecs",
    "message",
    "msg",
    "name",
    "pathname",
    "process",
    "processName",
    "relativeCreated",
    "stack_info",
    "thread",
    "threadName",
}


def redact_text(value: str) -> str:
    """Redact common credential shapes without logging their values."""

    value = _TELEGRAM_BOT_URL_PATTERN.sub(r"\1[REDACTED]", value)
    value = _BEARER_PATTERN.sub(r"\1[REDACTED]", value)
    return _SECRET_ASSIGNMENT_PATTERN.sub(r"\1[REDACTED]", value)


def _safe_value(key: str, value: Any) -> Any:
    normalized_key = key.lower()
    if any(marker in normalized_key for marker in ("key", "token", "secret", "authorization")):
        return "[REDACTED]"
    if isinstance(value, dict):
        return {str(k): _safe_value(str(k), v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe_value("item", item) for item in value]
    if isinstance(value, str):
        return redact_text(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return redact_text(str(value))


class JsonFormatter(logging.Formatter):
    """Render one JSON object per log record."""

    def format(self, record: logging.LogRecord) -> str:
        run_id = current_run_id.get()
        payload: dict[str, Any] = {
            "timestamp": datetime.now(UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "event": _safe_value("event", getattr(record, "event", record.getMessage())),
            "message": redact_text(record.getMessage()),
        }
        if run_id is not None:
            payload["run_id"] = run_id
        for key, value in record.__dict__.items():
            if key not in _RESERVED and key not in payload:
                payload[key] = _safe_value(key, value)
        if record.exc_info:
            payload["exception"] = redact_text(self.formatException(record.exc_info))
        return json.dumps(payload, ensure_ascii=False, default=str)


def configure_logging(level: str = "INFO") -> None:
    """Configure the root logger exactly once for command-line execution."""

    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level)
    # Routine request URLs add noise, disk I/O, and may contain credentials for
    # APIs that encode auth in the path (notably Telegram).
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("google_genai").setLevel(logging.WARNING)
