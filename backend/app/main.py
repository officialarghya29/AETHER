"""AETHER-ML FastAPI application factory."""

from __future__ import annotations

import time
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from aether.config import AetherConfig, load_config
from aether.monitoring.metrics import METRICS
from aether.pipeline import ReliabilityPipeline
from aether.runtime import configure_runtime, resolve_device
from aether.utils.logging import configure_logging, get_logger
from backend.app.api.routes import (
    experiments,
    monitoring,
    predictions,
    reports,
    reviews,
    system,
)
from backend.app.core.exceptions import RequestContextMiddleware, install_exception_handlers
from backend.app.core.security import SlidingWindowLimiter
from backend.app.database.session import create_all, init_engine

logger = get_logger("aether.api")
STARTED_AT = time.time()


def build_pipeline(config: AetherConfig) -> ReliabilityPipeline:
    """Construct the pipeline once per process; artifacts load lazily."""
    configure_runtime(config.runtime)
    device = resolve_device(config.runtime.device)
    pipeline = ReliabilityPipeline(config, device=device)
    try:
        pipeline.warm(("lightweight",))
    except Exception as exc:
        # Degraded start is intentional: /health reports the missing model instead of the
        # process refusing to boot, so operators can see *why* the service is unhealthy.
        logger.error("lightweight model unavailable at startup", extra={"error": str(exc)})
    return pipeline


@asynccontextmanager
async def lifespan(app: FastAPI):
    config: AetherConfig = app.state.config
    init_engine(config.storage.database_url)
    if config.application.environment != "production":
        create_all()
    pipeline = build_pipeline(config)
    app.state.pipeline = pipeline
    logger.info(
        "application started",
        extra={
            "environment": config.application.environment,
            "device": str(pipeline.device),
            "policy": config.routing.policy,
        },
    )
    METRICS.set_gauge("service_up", 1.0)
    try:
        yield
    finally:
        METRICS.set_gauge("service_up", 0.0)
        pipeline.registry.close()
        logger.info("application stopped")


def create_app(config: AetherConfig | None = None) -> FastAPI:
    """Create the application. Accepting a config makes the app testable in isolation."""
    cfg = config or load_config()
    configure_logging(cfg.monitoring.log_level, cfg.monitoring.json_logs)
    app = FastAPI(
        title=f"{cfg.application.name} API",
        version=cfg.application.version,
        description=(
            "Reliability-first inference: calibrated uncertainty, distribution-shift detection, "
            "cost-aware routing, evidence-backed diagnosis and human feedback."
        ),
        lifespan=lifespan,
    )
    app.state.config = cfg
    app.state.limiter = SlidingWindowLimiter(cfg.api.rate_limit_per_minute)

    app.add_middleware(RequestContextMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cfg.api.cors_origins,
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["Authorization", "Content-Type", "X-Request-Id", "X-Actor"],
        expose_headers=["X-Request-Id"],
    )
    install_exception_handlers(app)

    prefix = cfg.application.api_prefix
    app.include_router(system.router)
    app.include_router(system.router, prefix=prefix)
    app.include_router(predictions.router, prefix=prefix)
    app.include_router(monitoring.router, prefix=prefix)
    app.include_router(reviews.router, prefix=prefix)
    app.include_router(experiments.router, prefix=prefix)
    app.include_router(reports.router, prefix=prefix)

    @app.get("/", include_in_schema=False)
    def root() -> dict:
        return {
            "name": cfg.application.name,
            "version": cfg.application.version,
            "docs": "/docs",
            "health": "/health",
            "api": prefix,
        }

    return app


app = create_app()


def run() -> None:
    """Entry point for ``python -m backend.app.main``."""
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000, log_config=None)  # noqa: S104


if __name__ == "__main__":
    torch.set_num_threads(8)
    run()
