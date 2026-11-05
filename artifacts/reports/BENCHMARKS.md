# AETHER-ML benchmark report

Generated: 2026-11-04T20:43:03Z  
Device: `cpu` (cpu), threads: 8  
PyTorch 2.13.0+cpu, NumPy 2.5.2, image size 64, seed 42  
Test set: 10000 CIFAR-10 images (fingerprint `f9b1e2130bab680d`)

## Clean test performance

| model | accuracy | macro F1 | ECE | Brier | NLL | AURC | ms/sample |
| --- | --- | --- | --- | --- | --- | --- | --- |
| lightweight (1,528,106 params) | 0.7874 | 0.7879 | 0.0218 | 0.3018 | 0.6319 | 0.0677 | 0.18 |
| advanced (23,528,522 params) | 0.9442 | 0.9440 | 0.0131 | 0.0893 | 0.2029 | 0.0119 | 2.10 |

Selective prediction (risk at coverage):

- **lightweight**: AURC 0.0677; 0.50->0.0446, 0.70->0.0970, 0.80->0.1315, 0.90->0.1661, 1.00->0.2126
- **advanced**: AURC 0.0119; 0.50->0.0078, 0.70->0.0094, 0.80->0.0115, 0.90->0.0224, 1.00->0.0558

## Robustness

| model | clean acc | mean corrupted acc | worst corrupted acc | mean acc drop | mean corrupted ECE |
| --- | --- | --- | --- | --- | --- |
| lightweight | 0.7916 | 0.5318 | 0.2108 | 0.2598 | 0.1622 |
| advanced | 0.9468 | 0.6961 | 0.1904 | 0.2507 | 0.1312 |

Worst condition: `gaussian_noise` severity 5.

## Drift detection

- False-positive rate on clean development windows: `0.0`
- Mean detection rate at severity 1: `0.793`
- Mean detection rate at severity 3: `1.000`
- Mean detection rate at severity 5: `1.000`

## Routing policies

Stream: 5,000 clean routing-dev samples plus 3,000 corrupted samples (gaussian noise and brightness, severity 4). Accuracies below are on this mixed stream.

| policy | coverage | acc on answered | escalated | unresolved escalation | review | abstain | mean latency (ms) | cost/sample |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| always_light | 1.000 | 0.6550 | 0.000 | 0.000 | 0.000 | 0.000 | 0.18 | 3.450 |
| always_advanced | 1.000 | 0.8456 | 0.000 | 0.000 | 0.000 | 0.000 | 2.05 | 2.544 |
| fixed_threshold | 0.279 | 0.9588 | 0.741 | 0.000 | 0.000 | 0.721 | 0.26 | 1.594 |
| uncertainty_only | 0.251 | 0.9602 | 0.000 | 0.000 | 0.000 | 0.749 | 0.18 | 1.598 |
| shift_aware | 1.000 | 0.8413 | 0.741 | 0.000 | 0.000 | 0.000 | 1.70 | 2.328 |
| aether | 0.653 | 0.9382 | 0.316 | 0.000 | 0.000 | 0.347 | 0.78 | 1.390 |

### Ablations

| variant | coverage | acc on answered | cost/sample |
| --- | --- | --- | --- |
| no_uncertainty | 0.000 | 0.0000 | 2.000 |
| no_drift_signal | 0.867 | 0.9056 | 1.577 |
| no_cost_awareness | 1.000 | 0.8413 | 2.328 |
| no_disagreement | 0.652 | 0.9387 | 1.387 |
| no_escalation | 0.481 | 0.9002 | 1.518 |
| clean_only_subset | 0.963 | 0.9460 | 1.062 |

## Notes

- Costs are relative units: error 10, escalation 1, review 4, abstention 2. An abstention is charged the abstention cost and counted as uncovered (never as a wrong answer); a review carries the review cost plus the expected residual error.
- Shift detection is measured without labels; accuracy impact on unlabelled traffic is not claimed.
- Corruption families and severity tables are documented in `aether/data/corruptions.py`.
- The official test split is used for reporting only and never for tuning thresholds.
