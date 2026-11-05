# Product Requirements Document — AETHER-ML

## 1. Product vision

Machine-learning services fail quietly. A classifier keeps answering after the input distribution
has moved, keeps returning confident softmax scores that no longer mean what they did at training
time, and keeps spending compute on the same model regardless of whether a second, stronger model
would have answered correctly. AETHER-ML is the reliability layer that makes those failures visible
and actionable at inference time.

The product is a model-agnostic framework with one reference implementation: image classification
on CIFAR-10 and controlled corruptions of it.

## 2. Problem statement

Operators of ML systems cannot currently answer four questions from the traffic they already serve:

1. **How uncertain is this specific prediction, on a calibrated scale?**
2. **Has the input distribution moved away from what the model was validated on?**
3. **Is more compute — a stronger model, or a human — worth paying for on this request?**
4. **Why did the system answer, escalate, defer or refuse, and can that be audited later?**

Softmax confidence alone answers none of them. AETHER-ML separates the concepts that are usually
conflated: confidence, predictive uncertainty, out-of-distribution score, distribution drift,
estimated error risk, and actual correctness.

## 3. Target users

| User | Needs |
| --- | --- |
| ML engineer | Calibrated metrics, corruption robustness, a defensible routing baseline |
| Platform engineer | One service that decides accept/escalate/review/abstain with a stated budget |
| Reviewer / annotator | A queue of uncertain cases with model evidence attached |
| Research engineer | Reproducible split discipline, ablations, exportable evidence |
| SRE | Latency, escalation and drift signals without label access |

## 4. Product features

- **Model adapter layer** — MobileNetV3-Small and ResNet50 behind one contract, with logits,
  probabilities, embeddings, latency and version metadata.
- **Uncertainty intelligence** — entropy, MSP, margin, temperature scaling fitted on a dedicated
  calibration split, ECE/MCE, Brier, NLL, reliability and risk-coverage curves, MC dropout.
- **Distribution-shift detection** — PSI, KS and permutation MMD over input statistics and
  embeddings, plus prediction-behaviour tests, with a Bonferroni-corrected composite score and
  severity bands.
- **Adaptive routing** — a transparent rule policy, an expected-utility policy and a learned
  minimum-benefit policy sharing one evidence structure and one set of budgets.
- **Failure diagnosis** — diagnostic codes with severity, evidence, explanation and recommended
  action; image-quality indicators for acquisition problems.
- **Human feedback and recovery** — review queue, verified labels, candidate dataset with manifest
  and digest, offline promotion gate, model registry and rollback.
- **Operations surface** — FastAPI service, PostgreSQL schema and migrations, React dashboard,
  structured events, metrics and experiment reports.

## 5. Non-goals for v1.0.0

- Training state-of-the-art models; accuracy is a means, not the contribution.
- NLP or tabular adapters (the interfaces allow them; no implementation ships).
- Automatic retraining or deployment without human approval.
- Using an LLM as the routing mechanism.
- Claiming accuracy degradation from unlabelled traffic.

## 6. Acceptance criteria

A release is acceptable only when all of the following hold and are evidenced in
`docs/IMPLEMENTATION_STATUS.md`:

1. Both models load from versioned checkpoints and produce consistent class mappings.
2. Splits are disjoint and reproducible from a recorded seed and fingerprint.
3. Temperature scaling is fitted on the calibration split, never on test data.
4. Drift detectors report no-shift, mild, moderate and severe conditions with documented
   thresholds, and never report accuracy drift without labels.
5. Every routing action is reachable and explained; budgets are enforced; decisions are
   deterministic for identical evidence.
6. Diagnostics cite their evidence.
7. Feedback cannot enter trusted training data without a verified label and reviewer identity.
8. Promotion requires a passing offline gate and an explicit approver; rollback works.
9. API endpoints return real data with consistent error shapes and request identifiers.
10. The dashboard renders live API data, including honest empty and error states.
11. Tests pass; benchmarks reproduce from documented commands.
12. Limitations are documented rather than hidden.

## 7. Product limitations

- Corruption coverage is 10 of 15 CIFAR-10-C families.
- Routing evidence is single-image; batch context is not used.
- The learned policy models risk with logistic regression over a small evidence vector.
- Review quality is modelled, not measured: `reviewer_error_rate` is a configured assumption.
- CPU-only training constrains model size, epochs and image resolution.
