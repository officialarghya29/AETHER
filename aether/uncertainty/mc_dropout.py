"""Optional Monte Carlo dropout uncertainty.

Dropout stays enabled at inference and ``T`` stochastic passes are drawn. Two quantities are
returned: the mean predictive distribution and the mutual information between predictions and
parameters (a proxy for epistemic uncertainty).

    total uncertainty      = H(mean_t p_t)
    aleatoric uncertainty  = mean_t H(p_t)
    epistemic (mutual info) = total - aleatoric

It is disabled by default: it costs ``T`` forward passes, and the dropout rate learned during
training is not a calibrated Bayesian posterior.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass

import numpy as np
import torch
from torch import nn

from aether.uncertainty.entropy import predictive_entropy
from aether.utils.logging import get_logger

logger = get_logger("aether.uncertainty.mc_dropout")


@dataclass(frozen=True)
class MCDropoutResult:
    """Monte Carlo dropout summary for a single input."""

    samples: int
    mean_probabilities: list[float]
    predictive_entropy: float
    expected_entropy: float
    mutual_information: float

    def to_dict(self) -> dict:
        return asdict(self)


@contextmanager
def dropout_enabled(model: nn.Module) -> Iterator[None]:
    """Put dropout modules into training mode, restoring their previous state afterwards."""
    toggled: list[tuple[nn.Module, bool]] = []
    for module in model.modules():
        if isinstance(module, nn.modules.dropout._DropoutNd) and module.p > 0:
            toggled.append((module, module.training))
            module.train()
    if not toggled:
        logger.warning("MC dropout requested but the model has no active dropout layers")
    try:
        yield
    finally:
        for module, was_training in toggled:
            module.train(was_training)


def mc_dropout_predict(
    model: nn.Module,
    batch: torch.Tensor,
    device: torch.device,
    samples: int = 8,
) -> list[MCDropoutResult]:
    """Run ``samples`` stochastic forward passes and summarise uncertainty per input."""
    if samples < 2:
        raise ValueError("MC dropout requires at least two samples")
    collected: list[np.ndarray] = []
    with dropout_enabled(model), torch.inference_mode():
        for _ in range(samples):
            logits = model(batch.to(device)).float()
            probabilities = torch.softmax(logits, dim=-1).cpu().numpy()
            collected.append(probabilities.astype(np.float64))
    stacked = np.stack(collected, axis=0)  # (T, N, C)
    mean_probabilities = stacked.mean(axis=0)
    total_entropy = predictive_entropy(mean_probabilities)
    expected_entropy = np.array(
        [predictive_entropy(stacked[:, index, :]).mean() for index in range(stacked.shape[1])]
    )
    mutual_information = np.clip(total_entropy - expected_entropy, 0.0, None)
    return [
        MCDropoutResult(
            samples=samples,
            mean_probabilities=mean_probabilities[index].tolist(),
            predictive_entropy=float(total_entropy[index]),
            expected_entropy=float(expected_entropy[index]),
            mutual_information=float(mutual_information[index]),
        )
        for index in range(mean_probabilities.shape[0])
    ]
