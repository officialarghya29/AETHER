"""Validated, centralised configuration.

A single YAML file is the source of truth. Values are validated by Pydantic at startup, so an
invalid combination fails loudly before a model is loaded. Environment variables using the
``AETHER__<SECTION>__<FIELD>`` convention override individual values without editing files::

    AETHER__RUNTIME__SEED=7
    AETHER__STORAGE__DATABASE_URL=postgresql+psycopg://user:pass@host/db
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class StrictModel(BaseModel):
    """Base model that rejects unknown keys: silently ignored config is a latent bug."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class ApplicationConfig(StrictModel):
    name: str = "AETHER-ML"
    environment: Literal["development", "staging", "production"] = "development"
    api_prefix: str = "/api/v1"
    version: str = "1.0.0"

    @field_validator("api_prefix")
    @classmethod
    def _prefix(cls, value: str) -> str:
        return value if value.startswith("/") else f"/{value}"


class RuntimeConfig(StrictModel):
    device: Literal["auto", "cpu", "cuda"] = "auto"
    seed: int = 42
    max_batch_size: int = Field(64, ge=1, le=1024)
    num_threads: int = Field(8, ge=1, le=128)
    deterministic: bool = True


class SplitSizes(StrictModel):
    train: int = Field(32000, gt=0)
    validation: int = Field(4000, ge=0)
    calibration: int = Field(4000, ge=0)
    routing_dev: int = Field(5000, ge=0)
    holdout: int = Field(5000, ge=0)

    @property
    def total(self) -> int:
        return self.train + self.validation + self.calibration + self.routing_dev + self.holdout


class DataConfig(StrictModel):
    root: str = "data"
    dataset: Literal["cifar10"] = "cifar10"
    image_size: int = Field(64, ge=32, le=256)
    num_workers: int = Field(8, ge=0, le=64)
    eval_batch_size: int = Field(256, ge=1)
    splits: SplitSizes = SplitSizes()
    split_seed: int = 42
    reference_window_size: int = Field(2000, ge=100)

    @model_validator(mode="after")
    def _capacity(self) -> DataConfig:
        if self.splits.total > 50000:
            raise ValueError(
                f"split sizes total {self.splits.total} exceeds the 50,000 CIFAR-10 training images"
            )
        if self.splits.total < 2000:
            raise ValueError("split sizes total must be at least 2,000 images to train a model")
        return self


class ModelSpec(StrictModel):
    architecture: str
    role: Literal["lightweight", "advanced", "optional"] = "lightweight"
    checkpoint: str = ""
    version: str = "1.0.0"
    num_classes: int = Field(10, ge=2)
    pretrained: bool = True
    input_size: int | None = None


class ModelsConfig(StrictModel):
    lightweight: ModelSpec
    advanced: ModelSpec
    optional: ModelSpec | None = None

    @model_validator(mode="after")
    def _roles(self) -> ModelsConfig:
        if self.lightweight.num_classes != self.advanced.num_classes:
            raise ValueError("lightweight and advanced models must share a class space")
        return self


class MCDropoutConfig(StrictModel):
    enabled: bool = False
    samples: int = Field(8, ge=2, le=64)


class UncertaintyConfig(StrictModel):
    temperature_artifact: str = "artifacts/calibration/temperatures.json"
    calibration_bins: int = Field(15, ge=2, le=100)
    epsilon: float = Field(1e-12, gt=0)
    mc_dropout: MCDropoutConfig = MCDropoutConfig()


class SeverityThresholds(StrictModel):
    low: float = Field(1.5, gt=0)
    moderate: float = Field(3.0, gt=0)
    severe: float = Field(6.0, gt=0)

    @model_validator(mode="after")
    def _ordered(self) -> SeverityThresholds:
        if not (self.low < self.moderate < self.severe):
            raise ValueError("drift severity thresholds must be strictly increasing")
        return self


class DriftConfig(StrictModel):
    reference_artifact: str = "artifacts/reference/reference_stats.npz"
    evaluation_window_size: int = Field(200, ge=10)
    embedding_components: int = Field(32, ge=2, le=256)
    psi_bins: int = Field(10, ge=2, le=50)
    ks_alpha: float = Field(0.01, gt=0, lt=1)
    mmd_alpha: float = Field(0.01, gt=0, lt=1)
    mmd_max_samples: int = Field(400, ge=50)
    mmd_permutations: int = Field(64, ge=10, le=1000)
    severity: SeverityThresholds = SeverityThresholds()
    min_window_size: int = Field(30, ge=2)


class CostsConfig(StrictModel):
    error: float = Field(10.0, gt=0)
    escalate: float = Field(1.0, ge=0)
    review: float = Field(4.0, ge=0)
    abstain: float = Field(2.0, ge=0)
    correct: float = Field(1.0, gt=0)


class LearnedPolicyConfig(StrictModel):
    artifact: str = "artifacts/calibration/learned_policy.json"
    min_expected_benefit: float = Field(0.05, ge=0, le=1)


class RoutingConfig(StrictModel):
    policy: Literal["rule_based", "cost_aware", "learned"] = "cost_aware"
    risk_model_artifact: str = "artifacts/calibration/risk_model.json"
    accept_risk: float = Field(0.10, gt=0, lt=1)
    escalate_risk: float = Field(0.45, gt=0, lt=1)
    high_risk: float = Field(0.60, gt=0, lt=1)
    latency_budget_ms: float = Field(250.0, gt=0)
    advanced_latency_estimate_ms: float = Field(40.0, gt=0)
    max_escalations: int = Field(1, ge=0, le=4)
    enable_escalation: bool = True
    enable_abstention: bool = True
    review_available: bool = True
    reviewer_error_rate: float = Field(0.02, gt=0, lt=1)
    # Test-time augmentation on the escalated tier: average the logits of the image and its mirror.
    # Measured on routing data; off by default because it doubles the advanced-model spend.
    tta: bool = False
    # When TTA is on, pay the mirror pass only for requests whose window the drift monitor has
    # flagged. Clean traffic keeps single-pass cost while shifted traffic gets the accuracy gain.
    tta_on_drift: bool = True
    costs: CostsConfig = CostsConfig()
    learned_policy: LearnedPolicyConfig = LearnedPolicyConfig()

    @model_validator(mode="after")
    def _ordered(self) -> RoutingConfig:
        if not (self.accept_risk <= self.escalate_risk <= self.high_risk):
            raise ValueError("routing risks must satisfy accept <= escalate <= high")
        return self


class MonitoringConfig(StrictModel):
    enabled: bool = True
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    json_logs: bool = False
    experiment_tracking: Literal["local", "mlflow"] = "local"
    mlflow_tracking_uri: str = "file:./mlruns"
    mlflow_experiment: str = "aether-ml"


class StorageConfig(StrictModel):
    database_url: str = "sqlite:///./artifacts/db/aether.db"
    artifacts_root: str = "artifacts"
    upload_dir: str = "artifacts/uploads"
    upload_max_bytes: int = Field(5_242_880, gt=0)
    upload_max_pixels: int = Field(4_096_000, gt=0)
    atomic_store: bool = True


class AuthConfig(StrictModel):
    enabled: bool = False
    token_env: str = "AETHER_API_TOKEN"


class APIConfig(StrictModel):
    cors_origins: list[str] = Field(default_factory=lambda: ["http://localhost:5173"])
    request_timeout_ms: int = Field(30_000, gt=0)
    rate_limit_per_minute: int = Field(120, ge=0)
    auth: AuthConfig = AuthConfig()


class AetherConfig(StrictModel):
    application: ApplicationConfig = ApplicationConfig()
    runtime: RuntimeConfig = RuntimeConfig()
    data: DataConfig = DataConfig()
    models: ModelsConfig
    uncertainty: UncertaintyConfig = UncertaintyConfig()
    drift: DriftConfig = DriftConfig()
    routing: RoutingConfig = RoutingConfig()
    monitoring: MonitoringConfig = MonitoringConfig()
    storage: StorageConfig = StorageConfig()
    api: APIConfig = APIConfig()

    # --- path helpers -------------------------------------------------------
    def resolve(self, relative: str | Path) -> Path:
        """Resolve a configured path against the project root (absolute paths pass through)."""
        path = Path(relative)
        return path if path.is_absolute() else (PROJECT_ROOT / path)

    @property
    def data_root(self) -> Path:
        return self.resolve(self.data.root)

    @property
    def artifacts_root(self) -> Path:
        return self.resolve(self.storage.artifacts_root)

    @property
    def upload_dir(self) -> Path:
        return self.resolve(self.storage.upload_dir)


def _apply_env_overrides(payload: dict[str, Any]) -> dict[str, Any]:
    prefix = "AETHER__"
    for key, raw in os.environ.items():
        if not key.startswith(prefix):
            continue
        parts = [part.lower() for part in key[len(prefix) :].split("__") if part]
        if len(parts) < 2:
            continue
        node: dict[str, Any] = payload
        for part in parts[:-1]:
            node = node.setdefault(part, {})  # type: ignore[assignment]
            if not isinstance(node, dict):
                raise ValueError(f"environment override {key} collides with a scalar config value")
        node[parts[-1]] = yaml.safe_load(raw)
    return payload


def load_config(path: str | Path | None = None) -> AetherConfig:
    """Load and validate configuration from YAML, then apply environment overrides."""
    candidate = path or os.environ.get("AETHER_CONFIG") or PROJECT_ROOT / "configs" / "aether.yaml"
    config_path = Path(candidate)
    if not config_path.exists():
        raise FileNotFoundError(f"configuration file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"configuration root must be a mapping, got {type(payload).__name__}")
    return AetherConfig.model_validate(_apply_env_overrides(payload))
