# Algorithms — AETHER-ML

Every quantity below is implemented in `aether/` and covered by unit tests. Formulas are stated so
the implementation can be checked against them, not to imply novelty.

## 1. Uncertainty

**Predictive entropy** (nats):

```
H(p) = -Σ_i p_i log(p_i + ε),   ε = 1e-12
```

`normalized_entropy = H(p) / log C` makes scores comparable across class counts.

**Maximum softmax probability:** `MSP(x) = max_i p_i` — a confidence score, *not* a probability of
correctness.

**Prediction margin:** `p_top1 - p_top2`.

**Temperature scaling:** `p_i = softmax(z_i / T)`, with `T` fitted by minimising NLL on the
calibration split using L-BFGS-B over `log T`, bounded so `T ∈ [0.05, 20]`. Fitting on the test
split is refused by the API, not merely discouraged. Temperature scaling optimises NLL; ECE is
reported before and after, and is not guaranteed to improve on every dataset.

**Expected calibration error** with equal-width bins:

```
ECE = Σ_m (|B_m| / N) · |acc(B_m) - conf(B_m)|
```

Empty bins are reported with `accuracy = null` rather than a fabricated zero.

**Brier score** (multi-class, one-hot):

```
Brier = (1/N) Σ_n Σ_c (p_nc - y_nc)²
```

**Risk–coverage / AURC.** Samples are ranked by an uncertainty score; for each coverage level the
selective risk is the error rate among the most confident samples, and AURC is the area under that
curve (lower is better).

**MC dropout.** `mc_dropout.py` keeps dropout active for `samples` passes and returns mean
probabilities plus mutual-information-style dispersion. Disabled by default because it multiplies
latency.

## 2. Drift

Drift is computed over a **window** of observations and compared with a frozen reference built from
the holdout split.

**Population Stability Index** over reference decile bins:

```
PSI = Σ_b (p_b - q_b) · ln(p_b / q_b)
```

with empty bins clipped to a small floor. A constant reference feature returns
`status: constant_feature` instead of a divide-by-zero.

**Two-sample Kolmogorov–Smirnov** per numerical feature; the p-value is the asymptotic two-sided
value from `scipy.stats.ks_2samp`. Fewer than the minimum samples returns
`status: insufficient_data`.

**Maximum Mean Discrepancy** with an RBF kernel on standardised PCA components of embeddings:

```
MMD² = E[k(x,x')] + E[k(y,y')] - 2E[k(x,y)]
```

The p-value comes from a permutation null (configurable permutations, fixed seed). The unbiased
estimator is used, and negative values are clipped to zero.

**Prediction-behaviour tests:** KS on entropy and maximum probability, chi-square on class
frequencies.

**Composite severity.** Each test produces a p-value; Bonferroni correction multiplies by the
number of tests and clips at 1. The composite score is the maximum `-log10(p_adj)` across signals,
mapped to bands:

| Band | Score |
| --- | --- |
| none | < low |
| low | ≥ 1.5 |
| moderate | ≥ 3.0 |
| severe | ≥ 6.0 |

Thresholds are configuration, not universal constants. A window below `min_window_size` reports
`insufficient_window`; a missing reference reports `reference_unavailable`. **Neither is a
detection, and neither implies accuracy loss.** Unlabelled traffic can show that inputs moved; it
cannot show that accuracy fell.

## 3. Error risk

Three logistic models are fitted on the **calibration** split, augmented with corrupted calibration
samples so drift features have observed variation. The routing-development split is never used for
fitting; it is where the policies are evaluated:

```
P(error | x) = σ(w·φ(x) + b)
φ(x) = [entropy, MSP, margin, shifted_flag, disagreement]
```

| head | evidence | label | used when |
| --- | --- | --- | --- |
| `lightweight` | lightweight uncertainty | lightweight model wrong | first call |
| `advanced_precall` | lightweight uncertainty | advanced model wrong | pricing escalation |
| `advanced` | advanced uncertainty + disagreement | advanced model wrong | after the second call |

`disagreement` is excluded from the first two heads because it does not exist before a second model
call; including it would leak the escalation's own outcome. `drift_score` is excluded from every
head: development data has only two drift levels, so its coefficient is unidentified.

The pre-call head is what makes escalation a priced decision. Using the post-escalation head (or
the advanced model's own outputs) to estimate advanced risk before paying for the call would leak
information the caller does not have; the simulation and the service both use the pre-call head.
Such a prediction is marked as an estimate — only a completed call sets `advanced_evaluated`, and
from then on ACCEPT and HUMAN_REVIEW are priced with the measured advanced risk.

## 4. Routing utility

Actions are scored with configured costs (error 10, escalate 1, review 4, abstain 2, correct 1):

```
U_accept  = (1 - r_l) · c_correct - r_l · c_error
U_escalate = (1 - r_a) · c_correct - r_a · c_error - c_escalate
U_review  = (1 - r_l·ρ) · c_correct - (r_l·ρ) · c_error - c_review
U_abstain = -c_abstain
```

where `r_l` is lightweight risk, `r_a` advanced risk (estimated pre-call, measured after the call),
and `ρ` the reviewer error rate. In the offline replay, an abstention is charged the abstention cost
only — it is uncovered, not wrong — while a review carries the review cost plus the expected
residual error. That matches the utility model above, so the reported cost ranks policies by the
same objective the router optimises. Feasibility is applied first: escalation is removed when the request is over
`max_escalations`, the model is unavailable, or the latency budget cannot absorb the measured
advanced latency. The best feasible utility wins; ties break ACCEPT → MONITOR → ESCALATE →
HUMAN_REVIEW → ABSTAIN.

The **learned** policy adds a minimum-benefit gate:

```
benefit = r_l - r_a ;  escalate only if benefit ≥ min_expected_benefit
```

which removes second calls that are projected to change almost nothing.

## 5. Diagnosis rules

| Code | Trigger | Severity |
| --- | --- | --- |
| `ADVANCED_MODEL_FAILURE` | escalation raised or timed out | critical |
| `EMBEDDING_SHIFT` | drift severity moderate/severe | high |
| `HIGH_ENTROPY` | entropy ≥ threshold | medium |
| `LOW_CONFIDENCE` | MSP ≤ threshold | medium |
| `AMBIGUOUS_TOP2` | margin ≤ threshold | low |
| `IMAGE_BLUR` | Laplacian variance ≤ threshold | medium |
| `LOW_CONTRAST` / `OVEREXPOSED` | contrast or brightness outside band | low |

Each finding carries the measured value in `evidence`, an explanation that avoids causal language,
and a recommended action. Findings are sorted by severity.

## 6. Corruptions

Ten families are implemented with explicit severity tables: gaussian noise, shot noise, impulse
noise, defocus blur, gaussian blur, motion blur, zoom blur, brightness, contrast, pixelate and JPEG
compression. All are deterministic given a seed and severity, and preserve shape and dtype. Fog,
frost, snow, spatter and elastic transform are **not** implemented; results must not be described as
full CIFAR-10-C coverage.
