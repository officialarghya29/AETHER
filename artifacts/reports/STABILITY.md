# Stability

Risk-controlled thresholds and test-time augmentation, measured on calibration and
routing-development data. Regenerate with `make stability`.

## Conformal acceptance (clean routing-dev)

| tier | alpha | threshold | coverage | selective error |
| --- | --- | --- | --- | --- |
| lightweight | 0.20 | 0.395 | 0.953 | 0.192 |
| lightweight | 0.10 | 0.144 | 1.000 | 0.210 |
| lightweight | 0.05 | 0.053 | 1.000 | 0.210 |
| lightweight | 0.02 | 0.019 | 1.000 | 0.210 |
| advanced | 0.20 | 0.971 | 0.808 | 0.013 |
| advanced | 0.10 | 0.901 | 0.904 | 0.028 |
| advanced | 0.05 | 0.466 | 0.993 | 0.061 |
| advanced | 0.02 | 0.031 | 1.000 | 0.067 |

## Test-time augmentation (advanced tier)

| stream | identity | TTA | delta | mirror pass |
| --- | --- | --- | --- | --- |
| clean | 0.9335 | 0.9435 | +0.0100 | 2.44 ms |
| corrupted_gaussian_noise_severity_3 | 0.6350 | 0.6450 | +0.0100 | 2.56 ms |

Cost: one extra advanced forward pass per escalated request. The pipeline enables it with `routing.tta = true`.

## Protocol

Thresholds are fitted on the calibration split and evaluated on routing-development data; nothing here touches the official test split. TTA averages identity and horizontal-flip logits on the advanced tier only.
