"""Typed exception hierarchy.

Every failure surfaced to a caller is one of these; the API maps them to explicit HTTP
status codes and never leaks stack traces or filesystem paths to clients.
"""

from __future__ import annotations


class AetherError(Exception):
    """Base class for all AETHER-ML errors."""

    code = "aether_error"
    http_status = 500

    def __init__(self, message: str, *, details: dict | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def to_dict(self) -> dict:
        return {"error": {"code": self.code, "message": self.message, "details": self.details}}


class ConfigurationError(AetherError):
    code = "configuration_error"
    http_status = 500


class InvalidInputError(AetherError):
    code = "invalid_input"
    http_status = 422


class ArtifactNotFoundError(AetherError):
    code = "artifact_not_found"
    http_status = 503


class ModelNotAvailableError(ArtifactNotFoundError):
    code = "model_not_available"


class CalibrationNotFoundError(ArtifactNotFoundError):
    code = "calibration_not_found"


class ReferenceNotFoundError(ArtifactNotFoundError):
    code = "reference_not_found"


class BudgetExceededError(AetherError):
    code = "budget_exceeded"
    http_status = 429


class NotFoundError(AetherError):
    code = "not_found"
    http_status = 404


class ConflictError(AetherError):
    code = "conflict"
    http_status = 409


class AuthError(AetherError):
    code = "unauthorized"
    http_status = 401


class RateLimitError(AetherError):
    code = "rate_limited"
    http_status = 429


class InferenceError(AetherError):
    code = "inference_error"
    http_status = 500


class ValidationFailedError(AetherError):
    code = "validation_failed"
    http_status = 400
