"""Two-sample statistics used for drift detection.

Every function documents its assumptions and returns an explicit status when the data cannot
support a test (constant feature, empty window, too few samples). Silent zeros are never
returned for undefined statistics: upstream code must decide what to do.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy import stats


@dataclass(frozen=True)
class TestResult:
    """Outcome of one statistical comparison."""

    name: str
    statistic: float | None
    p_value: float | None
    status: str  # ok | insufficient_data | constant_feature | skipped
    note: str = ""

    @property
    def available(self) -> bool:
        return self.status == "ok" and self.p_value is not None

    def to_dict(self) -> dict:
        return asdict(self)


def population_stability_index(
    reference: np.ndarray,
    current: np.ndarray,
    bins: int = 10,
    epsilon: float = 1e-6,
) -> TestResult:
    """PSI between a reference and a current 1-D sample.

    Bins are reference quantiles, so the reference distribution defines the yardstick. A
    constant reference cannot define bins and is reported as ``constant_feature`` rather than
    producing a misleading score.
    """
    reference = np.asarray(reference, dtype=np.float64).ravel()
    current = np.asarray(current, dtype=np.float64).ravel()
    reference = reference[np.isfinite(reference)]
    current = current[np.isfinite(current)]
    if reference.size < 2 or current.size < 2:
        return TestResult("psi", None, None, "insufficient_data", "need >=2 finite samples")
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    if edges.size < 3:
        return TestResult("psi", None, None, "constant_feature", "reference has <3 distinct values")
    edges[0], edges[-1] = -np.inf, np.inf
    ref_counts, _ = np.histogram(reference, bins=edges)
    cur_counts, _ = np.histogram(current, bins=edges)
    ref_fraction = np.clip(ref_counts / max(1, ref_counts.sum()), epsilon, None)
    cur_fraction = np.clip(cur_counts / max(1, cur_counts.sum()), epsilon, None)
    psi = float(np.sum((cur_fraction - ref_fraction) * np.log(cur_fraction / ref_fraction)))
    # PSI has no closed-form null distribution; significance is judged by configuration.
    return TestResult("psi", psi, None, "ok", f"{edges.size - 1} reference-quantile bins")


def kolmogorov_smirnov(reference: np.ndarray, current: np.ndarray) -> TestResult:
    """Two-sample KS test on a 1-D numerical feature."""
    reference = np.asarray(reference, dtype=np.float64).ravel()
    current = np.asarray(current, dtype=np.float64).ravel()
    reference = reference[np.isfinite(reference)]
    current = current[np.isfinite(current)]
    if reference.size < 5 or current.size < 5:
        return TestResult("ks", None, None, "insufficient_data", "need >=5 finite samples")
    if np.ptp(reference) == 0 and np.ptp(current) == 0:
        return TestResult("ks", None, None, "constant_feature", "both samples constant")
    result = stats.ks_2samp(reference, current, alternative="two-sided", method="auto")
    return TestResult("ks", float(result.statistic), float(result.pvalue), "ok")


def chi_square_class_frequency(
    reference_counts: np.ndarray, current_counts: np.ndarray
) -> TestResult:
    """Chi-square goodness-of-fit of the current predicted-class histogram to the reference."""
    reference = np.asarray(reference_counts, dtype=np.float64).ravel()
    current = np.asarray(current_counts, dtype=np.float64).ravel()
    if reference.shape != current.shape:
        raise ValueError("reference and current histograms must have the same shape")
    if reference.sum() < 5 or current.sum() < 5:
        return TestResult("chi2_class_freq", None, None, "insufficient_data", "need >=5 counts")
    expected = reference / reference.sum() * current.sum()
    keep = expected >= 1.0
    if keep.sum() < 2:
        return TestResult(
            "chi2_class_freq", None, None, "insufficient_data", "too few populated classes"
        )
    # Rescale expected mass onto the kept classes to preserve the total count.
    expected = expected[keep] * (current.sum() / expected[keep].sum())
    observed = current[keep]
    result = stats.chisquare(f_obs=observed, f_exp=expected)
    return TestResult(
        "chi2_class_freq",
        float(result.statistic),
        float(result.pvalue),
        "ok",
        f"{int(keep.sum())} classes evaluated",
    )


def rbf_mmd(
    reference: np.ndarray,
    current: np.ndarray,
    *,
    gamma: float | None = None,
    max_samples: int = 400,
    permutations: int = 100,
    seed: int = 0,
) -> TestResult:
    """Maximum Mean Discrepancy with an RBF kernel and a permutation p-value.

    ``gamma`` defaults to the median-heuristic computed on the pooled sample. Subsampling keeps
    the cost at ``O((n+m)^2 d)`` and is applied deterministically from ``seed``.
    """
    reference = np.asarray(reference, dtype=np.float64)
    current = np.asarray(current, dtype=np.float64)
    if reference.ndim == 1:
        reference = reference[:, None]
    if current.ndim == 1:
        current = current[:, None]
    if not (np.isfinite(reference).all() and np.isfinite(current).all()):
        return TestResult("mmd", None, None, "insufficient_data", "non-finite values present")
    if reference.shape[0] < 10 or current.shape[0] < 10:
        return TestResult("mmd", None, None, "insufficient_data", "need >=10 rows per sample")
    rng = np.random.default_rng(seed)
    if reference.shape[0] > max_samples:
        reference = reference[rng.choice(reference.shape[0], max_samples, replace=False)]
    if current.shape[0] > max_samples:
        current = current[rng.choice(current.shape[0], max_samples, replace=False)]

    pooled = np.vstack([reference, current])
    if gamma is None:
        pairwise = np.sum((pooled[:, None, :] - pooled[None, :, :]) ** 2, axis=-1)
        median = np.median(pairwise[pairwise > 0]) if np.any(pairwise > 0) else 1.0
        gamma = 1.0 / max(median, 1e-12)

    def statistic(first: np.ndarray, second: np.ndarray) -> float:
        k_xx = _rbf(first, first, gamma)
        k_yy = _rbf(second, second, gamma)
        k_xy = _rbf(first, second, gamma)
        return float(k_xx.mean() + k_yy.mean() - 2 * k_xy.mean())

    observed = statistic(reference, current)
    n_ref = reference.shape[0]
    combined = pooled
    exceed = 0
    for _ in range(permutations):
        order = rng.permutation(combined.shape[0])
        permuted = combined[order]
        value = statistic(permuted[:n_ref], permuted[n_ref:])
        if value >= observed:
            exceed += 1
    p_value = (exceed + 1) / (permutations + 1)
    return TestResult(
        "mmd",
        observed,
        float(p_value),
        "ok",
        f"rbf gamma={gamma:.6g}, {permutations} permutations",
    )


def _rbf(first: np.ndarray, second: np.ndarray, gamma: float) -> np.ndarray:
    squared = (
        np.sum(first**2, axis=1)[:, None]
        + np.sum(second**2, axis=1)[None, :]
        - 2 * first @ second.T
    )
    return np.exp(-gamma * np.clip(squared, 0, None))


def bonferroni(p_values: list[float]) -> list[float]:
    """Bonferroni-adjust p-values, clipped at 1, preserving order."""
    count = len(p_values)
    if count == 0:
        return []
    return [min(1.0, value * count) for value in p_values]
