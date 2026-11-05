"""Structured, dependency-free logging.

A single configuration point for the library, backend and training scripts. JSON output is
opt-in (``monitoring.json_logs``) because it is what production log shippers expect, while the
default human-readable form keeps local development usable.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from contextvars import ContextVar
from typing import Any

request_id_var: ContextVar[str] = ContextVar("aether_request_id", default="-")

_CONFIGURED = False

_RESERVED = {
    "name",
    "msg",
    "args",
    "levelname",
    "levelno",
    "pathname",
    "filename",
    "module",
    "exc_info",
    "exc_text",
    "stack_info",
    "lineno",
    "funcName",
    "created",
    "msecs",
    "relativeCreated",
    "thread",
    "threadName",
    "processName",
    "process",
    "taskName",
    "message",
    "asctime",
}


class JsonFormatter(logging.Formatter):
    """Render a log record as a single-line JSON object."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created))
            + f".{int(record.msecs):03d}Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "request_id": request_id_var.get(),
        }
        for key, value in record.__dict__.items():
            if key not in _RESERVED and not key.startswith("_"):
                payload[key] = value
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


class TextFormatter(logging.Formatter):
    """Render a compact human-readable line including the active request identifier."""

    def format(self, record: logging.LogRecord) -> str:
        request_id = request_id_var.get()
        record.__dict__["request_id"] = request_id
        extras = {
            key: value
            for key, value in record.__dict__.items()
            if key not in _RESERVED and not key.startswith("_") and key != "request_id"
        }
        base = (
            f"{self.formatTime(record, '%H:%M:%S')} {record.levelname:<7} "
            f"{record.name} [{request_id}] {record.getMessage()}"
        )
        if extras:
            base += " " + " ".join(f"{key}={value}" for key, value in sorted(extras.items()))
        if record.exc_info:
            base += "\n" + self.formatException(record.exc_info)
        return base


def configure_logging(level: str = "INFO", json_logs: bool = False) -> None:
    """Configure the root logger once. Repeated calls are cheap and idempotent."""
    global _CONFIGURED
    root = logging.getLogger()
    if _CONFIGURED:
        for handler in root.handlers:
            if isinstance(handler, logging.StreamHandler):
                handler.setFormatter(JsonFormatter() if json_logs else TextFormatter())
        root.setLevel(level)
        return
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter() if json_logs else TextFormatter())
    root.handlers = [handler]
    root.setLevel(level)
    for noisy in ("uvicorn.access", "matplotlib", "PIL"):
        logging.getLogger(noisy).setLevel(max(logging.INFO, getattr(logging, level)))
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a namespaced logger."""
    return logging.getLogger(name)
