"""Experiment registry backed by the database."""

from __future__ import annotations

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from aether.errors import NotFoundError
from backend.app.database.models import ExperimentModel


class ExperimentService:
    """Read/write access to recorded experiments."""

    def __init__(self, session: Session) -> None:
        self.session = session

    def list(self, limit: int = 50, offset: int = 0) -> tuple[list[ExperimentModel], int]:
        rows = list(
            self.session.scalars(
                select(ExperimentModel)
                .order_by(ExperimentModel.created_at.desc())
                .limit(limit)
                .offset(offset)
            )
        )
        total = int(self.session.scalar(select(func.count()).select_from(ExperimentModel)) or 0)
        return rows, total

    def get(self, experiment_id: str) -> ExperimentModel:
        row = self.session.scalar(
            select(ExperimentModel).where(ExperimentModel.experiment_id == experiment_id)
        )
        if row is None:
            raise NotFoundError(f"experiment {experiment_id} not found")
        return row

    def upsert(self, payload, notes: str | None = None) -> ExperimentModel:
        existing = self.session.scalar(
            select(ExperimentModel).where(ExperimentModel.experiment_id == payload.experiment_id)
        )
        if existing is not None:
            existing.name = payload.name
            existing.config = payload.config
            existing.metrics = payload.metrics
            existing.dataset_id = payload.dataset_id
            existing.seed = payload.seed
            existing.notes = notes
            return existing
        row = ExperimentModel(
            experiment_id=payload.experiment_id,
            name=payload.name,
            status="completed",
            config=payload.config,
            metrics=payload.metrics,
            dataset_id=payload.dataset_id,
            seed=payload.seed,
            notes=notes,
        )
        self.session.add(row)
        self.session.flush()
        return row
