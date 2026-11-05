# Stability

Risk-controlled thresholds and test-time augmentation, measured on calibration and
routing-development data. Regenerate with `make stability`.

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

## Test-time augmentation (advanced tier)

| stream | identity | TTA | delta | mirror pass |
| --- | --- | --- | --- | --- |
| clean | 0.9335 | 0.9435 | +0.0100 | 3.90 ms |
| corrupted_gaussian_noise_severity_3 | 0.6350 | 0.6450 | +0.0100 | 3.56 ms |

Cost: one extra advanced forward pass per escalated request. The pipeline enables it with `routing.tta = true`.

## Protocol

Thresholds are fitted on the calibration split and evaluated on routing-development data; nothing here touches the official test split. TTA averages identity and horizontal-flip logits on the advanced tier only.
