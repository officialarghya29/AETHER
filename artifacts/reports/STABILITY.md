# Stability

Risk-controlled thresholds, severity sweep and test-time augmentation, measured on
calibration and routing-development data. Regenerate with `make stability`.

## Risk-controlled acceptance

| tier | alpha | threshold | coverage (clean) | selective error (clean) | selective error (corrupted) |
| --- | --- | --- | --- | --- | --- |
| lightweight | 0.20 | 0.311 | 0.989 | 0.206 | 0.693 |
| lightweight | 0.10 | 0.716 | 0.694 | 0.097 | 0.577 |
| lightweight | 0.05 | 0.883 | 0.507 | 0.050 | 0.425 |
| lightweight | 0.02 | 0.969 | 0.276 | 0.020 | 0.229 |
| advanced | 0.20 | 0.322 | 0.999 | 0.066 | 0.354 |
| advanced | 0.10 | 0.322 | 0.999 | 0.066 | 0.354 |
| advanced | 0.05 | 0.534 | 0.987 | 0.056 | 0.307 |
| advanced | 0.02 | 0.947 | 0.869 | 0.018 | 0.134 |

## Severity sweep (gaussian noise)

| tier | alpha | severity | coverage | selective error | overall error |
| --- | --- | --- | --- | --- | --- |
| lightweight | 0.10 | 1 | 0.566 | 0.908 | 0.904 |
| lightweight | 0.10 | 3 | 0.382 | 0.880 | 0.893 |
| lightweight | 0.10 | 5 | 0.213 | 0.873 | 0.885 |
| lightweight | 0.05 | 1 | 0.364 | 0.907 | 0.904 |
| lightweight | 0.05 | 3 | 0.175 | 0.891 | 0.893 |
| lightweight | 0.05 | 5 | 0.066 | 0.864 | 0.885 |
| advanced | 0.10 | 1 | 0.999 | 0.921 | 0.921 |
| advanced | 0.10 | 3 | 0.980 | 0.908 | 0.909 |
| advanced | 0.10 | 5 | 0.933 | 0.897 | 0.897 |
| advanced | 0.05 | 1 | 0.965 | 0.921 | 0.921 |
| advanced | 0.05 | 3 | 0.827 | 0.918 | 0.909 |
| advanced | 0.05 | 5 | 0.556 | 0.897 | 0.897 |

## Test-time augmentation (advanced tier)

| stream | identity | TTA | delta | mirror pass |
| --- | --- | --- | --- | --- |
| clean | 0.9335 | 0.9435 | +0.0100 | 2.43 ms |
| corrupted_gaussian_noise_severity_3 | 0.6350 | 0.6450 | +0.0100 | 2.40 ms |

Drift gating: the mirror pass is paid on 0% of clean windows and 100% of severity-3 noisy windows, using the detector's measured rates.

Cost: one extra advanced forward pass per escalated request, skipped on clean windows.

## Protocol

Thresholds are fitted on the calibration split and evaluated on routing-development data; nothing here touches the official test split. TTA averages identity and horizontal-flip logits on the advanced tier only, and drift gating means the mirror pass is paid only when the monitor flags the window.
