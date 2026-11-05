"""Temperature scaling.

``p = softmax(logits / T)`` with a single scalar ``T`` fitted by minimising negative
log-likelihood on a **calibration split that is disjoint from test data**. Fitting uses
log-space bounded scalar minimisation, which is cheap and deterministic.

A calibrated probability is more *honest* about its uncertainty; it is not a guarantee of
correctness, and the artifact records exactly which split and dataset fingerprint produced it.
"""

from __future__ import annotations

import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar

from aether.uncertainty.entropy import softmax
from aether.uncertainty.metrics import (
    expected_calibration_error,
    negative_log_likelihood,
)
from aether.utils.io import atomic_write_json, read_json
from aether.utils.logging import get_logger

logger = get_logger("aether.uncertainty.calibration")


@dataclass(frozen=True)
class TemperatureCalibration:
    """A fitted temperature plus the evidence needed to audit the fit."""

    model_id: str
    temperature: float
    split: str
    samples: int
    nll_before: float
    nll_after: float
    ece_before: float
    ece_after: float
    fitted_at: str
    dataset_fingerprint: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def fit_temperature(
    logits: np.ndarray,
    labels: np.ndarray,
    *,
    model_id: str,
    split: str = "calibration",
    dataset_fingerprint: str = "",
    bounds: tuple[float, float] = (0.05, 20.0),
    n_bins: int = 15,
) -> TemperatureCalibration:
    """Fit ``T`` by minimising calibration-split NLL; ``T=1`` is the baseline."""
    logits = np.asarray(logits, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    if logits.ndim != 2:
        raise ValueError("logits must be a 2-D (N, C) array")
    if logits.shape[0] != labels.shape[0]:
        raise ValueError("logits and labels must have the same number of samples")
    if split == "test":
        raise ValueError("refusing to fit temperature on the test split (leakage)")

    def objective(log_temperature: float) -> float:
        temperature = float(np.exp(log_temperature))
        return negative_log_likelihood(softmax(logits / temperature), labels)

    result = minimize_scalar(
        objective,
        bounds=(float(np.log(bounds[0])), float(np.log(bounds[1]))),
        method="bounded",
        options={"xatol": 1e-4},
    )
    temperature = float(np.exp(result.x)) if result.success else 1.0

    baseline = softmax(logits)
    scaled = softmax(logits / temperature)
    calibration = TemperatureCalibration(
        model_id=model_id,
        temperature=round(temperature, 6),
        split=split,
        samples=int(labels.shape[0]),
        nll_before=negative_log_likelihood(baseline, labels),
        nll_after=negative_log_likelihood(scaled, labels),
        ece_before=expected_calibration_error(baseline, labels, n_bins),
        ece_after=expected_calibration_error(scaled, labels, n_bins),
        fitted_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        dataset_fingerprint=dataset_fingerprint,
    )
    logger.info(
        "temperature fitted",
        extra={
            "model_id": model_id,
            "temperature": calibration.temperature,
            "nll_before": round(calibration.nll_before, 4),
            "nll_after": round(calibration.nll_after, 4),
            "ece_before": round(calibration.ece_before, 4),
            "ece_after": round(calibration.ece_after, 4),
        },
    )
    return calibration


def apply_temperature(logits: np.ndarray, temperature: float) -> np.ndarray:
    """Return calibrated probabilities."""
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    return softmax(np.asarray(logits, dtype=np.float64) / float(temperature))


class CalibrationStore:
    """JSON-backed store of fitted temperatures keyed by model id.

    The artifact is read once and cached against the file's modification time and size. Serving
    previously re-read and re-parsed the JSON on every prediction, which put a filesystem round
    trip on the inference hot path for a value that changes at most once per calibration run.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.Lock()
        self._cache: dict[str, dict] = {}
        self._stamp: tuple[int, int] | None = None

    def load(self) -> dict[str, dict]:
        try:
            stat = self.path.stat()
        except OSError:
            stat = None
        stamp = (stat.st_mtime_ns, stat.st_size) if stat is not None else None
        with self._lock:
            if stamp is not None and stamp == self._stamp:
                return self._cache
            payload = read_json(self.path, default={}) or {}
            calibrations = payload.get("calibrations", payload)
            if not isinstance(calibrations, dict):
                calibrations = {}
            self._cache = calibrations
            self._stamp = stamp
            return calibrations

    def get(self, model_id: str) -> TemperatureCalibration | None:
        record = self.load().get(model_id)
        if not record:
            return None
        return TemperatureCalibration(**record)

    def temperature(self, model_id: str, default: float = 1.0) -> float:
        calibration = self.get(model_id)
        return calibration.temperature if calibration else default

    def save(self, calibration: TemperatureCalibration) -> None:
        existing = self.load()
        existing[calibration.model_id] = calibration.to_dict()
        atomic_write_json(
            self.path,
            {
                "schema_version": 1,
                "calibrations": existing,
                "note": "Temperatures are fitted on a dedicated calibration split, never on test data.",
            },
        )
        logger.info("calibration artifact updated", extra={"path": str(self.path)})
