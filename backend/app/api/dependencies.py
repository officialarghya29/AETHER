"""FastAPI dependencies shared by the routers."""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends, Request
from sqlalchemy.orm import Session

from aether.config import AetherConfig
from aether.pipeline import ReliabilityPipeline
from backend.app.database.session import get_db

SessionDep = Annotated[Session, Depends(get_db)]


def get_config(request: Request) -> AetherConfig:
    return request.app.state.config


def get_pipeline(request: Request) -> ReliabilityPipeline:
    return request.app.state.pipeline


ConfigDep = Annotated[AetherConfig, Depends(get_config)]
PipelineDep = Annotated[ReliabilityPipeline, Depends(get_pipeline)]


def enforce_rate_limit(request: Request) -> None:
    """Apply the configured per-client rate limit."""
    limiter = getattr(request.app.state, "limiter", None)
    if limiter is None:
        return
    from backend.app.core.security import client_key

    limiter.check(client_key(request))


RateLimitDep = Annotated[None, Depends(enforce_rate_limit)]

__all__ = [
    "ConfigDep",
    "PipelineDep",
    "RateLimitDep",
    "SessionDep",
    "get_config",
    "get_db",
    "get_pipeline",
]
