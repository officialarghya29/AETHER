"""AETHER-ML: Adaptive Evidence-based Trust, Evaluation & Recovery for ML systems.

The package is deliberately model-agnostic. ``aether.pipeline.ReliabilityPipeline`` is the
single orchestration entry point used by both the HTTP backend and the CLI/experiments, so
behaviour observed in benchmarks is the behaviour served in production.
"""

from aether.config import AetherConfig, load_config

__version__ = "1.0.0"
__all__ = ["AetherConfig", "load_config", "__version__"]
