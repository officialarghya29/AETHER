"""Authentication and rate limiting.

Protected operations (review submission, feedback, model management, experiment writes) require
a bearer token when ``api.auth.enabled`` is true. The token is read from an environment variable
at request time; it is never logged and never written to disk by the service.

Rate limiting is a per-IP sliding window sized from configuration. It is intentionally in-process:
a single-node deployment does not need Redis, and a multi-node deployment should use an ingress
limiter instead of pretending this one is global.
"""

from __future__ import annotations

import hmac
import os
import threading
import time
from collections import deque

from fastapi import Request

from aether.errors import AuthError, RateLimitError
from aether.utils.logging import get_logger

logger = get_logger("aether.api.security")


def require_auth(request: Request) -> str:
    """Return the actor identity, enforcing bearer-token auth when enabled."""
    settings = request.app.state.config.api.auth
    if not settings.enabled:
        return "anonymous"
    expected = os.environ.get(settings.token_env, "")
    if not expected:
        raise AuthError(
            "API authentication is enabled but no token is configured",
            details={"env": settings.token_env},
        )
    header = request.headers.get("authorization", "")
    if not header.lower().startswith("bearer "):
        raise AuthError("missing bearer token")
    provided = header.split(" ", 1)[1].strip()
    if not hmac.compare_digest(provided, expected):
        raise AuthError("invalid bearer token")
    return request.headers.get("x-actor", "api-client")[:128]


class SlidingWindowLimiter:
    """Per-key sliding-window request limiter."""

    def __init__(self, limit_per_minute: int) -> None:
        self.limit = limit_per_minute
        self._hits: dict[str, deque[float]] = {}
        self._lock = threading.Lock()

    def check(self, key: str) -> None:
        if self.limit <= 0:
            return
        now = time.monotonic()
        window_start = now - 60.0
        with self._lock:
            bucket = self._hits.setdefault(key, deque())
            while bucket and bucket[0] < window_start:
                bucket.popleft()
            if len(bucket) >= self.limit:
                raise RateLimitError(
                    "rate limit exceeded", details={"limit_per_minute": self.limit}
                )
            bucket.append(now)


def client_key(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"
