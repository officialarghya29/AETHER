"""Read-only access to generated experiment reports.

Only an allow-list of report names is served, so this cannot be used to read arbitrary files.
The dashboard renders exactly the artifacts written by the experiment scripts — it never
invents numbers.
"""

from __future__ import annotations

from fastapi import APIRouter

from aether.errors import NotFoundError
from aether.utils.io import read_json
from backend.app.api.dependencies import ConfigDep

router = APIRouter(tags=["reports"])

ALLOWED_REPORTS = {
    "benchmarks": "benchmarks.json",
    "calibration": "calibration.json",
    "reference": "reference.json",
    "training_lightweight": "training_lightweight.json",
    "training_advanced": "training_advanced.json",
}


@router.get("/reports")
def list_reports(config: ConfigDep) -> dict:
    """List the reports that currently exist on disk."""
    directory = config.resolve(config.storage.artifacts_root) / "reports"
    available = {
        name: (directory / filename).exists() for name, filename in ALLOWED_REPORTS.items()
    }
    return {"available": available, "directory": f"{config.storage.artifacts_root}/reports"}


@router.get("/reports/{name}")
def get_report(name: str, config: ConfigDep) -> dict:
    """Return a report by allow-listed name."""
    if name not in ALLOWED_REPORTS:
        raise NotFoundError(
            f"report '{name}' is not available",
            details={"available": sorted(ALLOWED_REPORTS)},
        )
    path = config.resolve(config.storage.artifacts_root) / "reports" / ALLOWED_REPORTS[name]
    payload = read_json(path, default=None)
    if payload is None:
        raise NotFoundError(
            f"report '{name}' has not been generated yet",
            details={"expected": f"artifacts/reports/{ALLOWED_REPORTS[name]}"},
        )
    return payload
