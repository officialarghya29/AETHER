"""HTTP error mapping and request-scoped context middleware."""

from __future__ import annotations

import time
import uuid

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from aether.errors import AetherError
from aether.monitoring.metrics import METRICS
from aether.utils.logging import get_logger, request_id_var

logger = get_logger("aether.api")


class RequestContextMiddleware(BaseHTTPMiddleware):
    """Attach a request id, log completion, and record latency."""

    async def dispatch(self, request: Request, call_next):
        request_id = request.headers.get("x-request-id") or uuid.uuid4().hex[:16]
        token = request_id_var.set(request_id)
        started = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            elapsed_ms = (time.perf_counter() - started) * 1000
            METRICS.increment("http_requests_total")
            METRICS.increment("http_requests_failed")
            logger.exception(
                "request failed",
                extra={"path": request.url.path, "method": request.method, "ms": round(elapsed_ms, 2)},
            )
            request_id_var.reset(token)
            raise
        elapsed_ms = (time.perf_counter() - started) * 1000
        response.headers["x-request-id"] = request_id
        METRICS.increment("http_requests_total")
        METRICS.observe("http_request_ms", elapsed_ms)
        if response.status_code >= 500:
            METRICS.increment("http_requests_failed")
        logger.info(
            "request",
            extra={
                "path": request.url.path,
                "method": request.method,
                "status": response.status_code,
                "ms": round(elapsed_ms, 2),
            },
        )
        request_id_var.reset(token)
        return response


def install_exception_handlers(app: FastAPI) -> None:
    """Map domain errors onto explicit HTTP responses without leaking internals."""

    @app.exception_handler(AetherError)
    async def _aether_error(request: Request, exc: AetherError) -> JSONResponse:
        logger.warning("domain error", extra={"code": exc.code, "error": exc.message})
        return JSONResponse(
            status_code=exc.http_status,
            content={**exc.to_dict(), "request_id": request_id_var.get()},
        )

    @app.exception_handler(RequestValidationError)
    async def _validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
        return JSONResponse(
            status_code=422,
            content={
                "error": {
                    "code": "validation_error",
                    "message": "request validation failed",
                    "details": {"errors": exc.errors()[:10]},
                },
                "request_id": request_id_var.get(),
            },
        )

    @app.exception_handler(Exception)
    async def _unhandled(request: Request, exc: Exception) -> JSONResponse:
        # Log with traceback, return a sanitised body: no paths, no stack, no secrets.
        logger.exception("unhandled exception", extra={"path": request.url.path})
        return JSONResponse(
            status_code=500,
            content={
                "error": {
                    "code": "internal_error",
                    "message": "an internal error occurred",
                    "details": {},
                },
                "request_id": request_id_var.get(),
            },
        )
