"""Feature-space drift detection.

Raw 576/2048-dimensional embeddings make two-sample tests noisy in small windows. The detector
therefore fits a PCA basis on the *reference* embeddings and compares the leading components,
which concentrates most of the variance into a handful of well-conditioned one-dimensional
tests, plus one multivariate MMD test on the projected space.

The PCA basis is part of the reference artifact: drift is measured against a frozen yardstick,
never recomputed from incoming traffic.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from aether.drift.statistical import (
    TestResult,
    kolmogorov_smirnov,
    population_stability_index,
    rbf_mmd,
)


@dataclass(frozen=True)
class PCABasis:
    """Frozen projection fitted on reference embeddings."""

    mean: np.ndarray
    components: np.ndarray  # (n_components, n_features)
    explained_variance_ratio: np.ndarray

    @property
    def n_components(self) -> int:
        return int(self.components.shape[0])

    def transform(self, embeddings: np.ndarray) -> np.ndarray:
        matrix = np.asarray(embeddings, dtype=np.float64)
        if matrix.ndim != 2 or matrix.shape[1] != self.mean.shape[0]:
            observed = matrix.shape[1] if matrix.ndim == 2 else matrix.shape
            raise ValueError(
                f"embedding dimension {observed} does not match the reference basis "
                f"({self.mean.shape[0]}); a reference is model-specific"
            )
        return (matrix - self.mean) @ self.components.T


def fit_pca(embeddings: np.ndarray, n_components: int = 32) -> PCABasis:
    """Fit a PCA basis using SVD on the reference embeddings."""
    matrix = np.asarray(embeddings, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError("embeddings must be a 2-D (N, D) array")
    if matrix.shape[0] < 2:
        raise ValueError("need at least two reference embeddings to fit PCA")
    n_components = int(min(n_components, matrix.shape[0] - 1, matrix.shape[1]))
    mean = matrix.mean(axis=0)
    centered = matrix - mean
    # economy SVD keeps memory bounded for 2048-dimensional embeddings
    _, singular_values, vt = np.linalg.svd(centered, full_matrices=False)
    variance = singular_values**2
    total = variance.sum() if variance.sum() > 0 else 1.0
    components = vt[:n_components]
    return PCABasis(
        mean=mean,
        components=components,
        explained_variance_ratio=(variance[:n_components] / total).astype(np.float64),
    )


def embedding_drift(
    reference_projected: np.ndarray,
    current_embeddings: np.ndarray,
    basis: PCABasis,
    *,
    psi_bins: int = 10,
    max_samples: int = 400,
    permutations: int = 64,
    seed: int = 0,
) -> dict[str, object]:
    """Compare current embeddings to the reference projection.

    Returns per-component KS tests, one PSI per leading component (first 8), one MMD on the
    projected space, and a summary share of components flagged as shifted before correction.
    """
    current_projected = basis.transform(current_embeddings)
    reference_projected = np.asarray(reference_projected, dtype=np.float64)
    if reference_projected.ndim != 2:
        raise ValueError("reference projection must be 2-D")

    ks_tests: list[TestResult] = []
    psi_values: list[dict[str, object]] = []
    significant = 0
    for index in range(reference_projected.shape[1]):
        test = kolmogorov_smirnov(reference_projected[:, index], current_projected[:, index])
        ks_tests.append(test)
        if test.available and test.p_value is not None and test.p_value < 0.01:
            significant += 1
        if index < min(8, reference_projected.shape[1]):
            psi = population_stability_index(
                reference_projected[:, index], current_projected[:, index], bins=psi_bins
            )
            psi_values.append(
                {
                    "component": index,
                    "psi": psi.statistic,
                    "status": psi.status,
                    "explained_variance_ratio": float(basis.explained_variance_ratio[index]),
                }
            )

    mmd = rbf_mmd(
        reference_projected,
        current_projected,
        max_samples=max_samples,
        permutations=permutations,
        seed=seed,
    )
    return {
        "statistic": "embedding_drift",
        "ks_tests": [test.to_dict() for test in ks_tests],
        "psi_components": psi_values,
        "mmd": mmd.to_dict(),
        "significant_components": significant,
        "components_tested": int(reference_projected.shape[1]),
        "projected_dim": int(current_projected.shape[1]),
        "explained_variance": float(basis.explained_variance_ratio.sum()),
    }
