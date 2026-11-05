# Efficiency

Measured, not asserted. Regenerate with `make efficiency`.

CPU only · 16 cores · torch 2.13.0+cpu · numpy 2.5.2 · Python 3.12.3 · 8 torch threads

## Component scope

| stage | per image | batch p50 | batch p95 |
| --- | --- | --- | --- |
| quality_analysis | 0.1510 ms | 2.291 ms | 2.651 ms |
| preprocess_batch | 0.0708 ms | 1.126 ms | 1.157 ms |
| light_inference_batch | 0.6113 ms | 9.490 ms | 11.293 ms |
| temperature_lookup | 0.0086 ms | 0.135 ms | 0.147 ms |
| calibrate_and_uncertainty | 0.0317 ms | 0.505 ms | 0.522 ms |
| risk_features | 0.0006 ms | 0.009 ms | 0.010 ms |
| risk_heads_two | 0.0076 ms | 0.120 ms | 0.128 ms |
| drift_bookkeeping | 0.0002 ms | 0.003 ms | 0.006 ms |
| routing_decide | 0.0200 ms | 0.319 ms | 0.332 ms |

## Batch scope

| batch | batch latency | ms / image | images / s |
| --- | --- | --- | --- |
| 1 | 4.170 ms | 4.170 | 239.8 |
| 2 | 7.664 ms | 3.832 | 260.9 |
| 4 | 22.185 ms | 5.546 | 180.3 |
| 8 | 26.771 ms | 3.346 | 298.8 |
| 16 | 27.571 ms | 1.723 | 580.3 |
| 32 | 39.852 ms | 1.245 | 803.0 |
| 64 | 89.099 ms | 1.392 | 718.3 |

## Feature scope (per image, end to end)

All configurations answer from the light tier (escalation disabled) so the comparison isolates the capability, not the routing mix.

| configuration | ms / image | Δ vs baseline |
| --- | --- | --- |
| baseline | 0.763 | — |
| without drift bookkeeping | 0.708 | +0.054 |
| heuristic heads | 0.704 | +0.059 |
| forced escalation (advanced call) | 2.777 | only escalated images pay this |

Composite drift evaluation: 197.1 ms per window, 0.9855 ms/image amortised over the window.

## Persistence

9.772 ms per prediction to write the prediction, routing decision and uncertainty records (sqlite in-memory via SQLAlchemy ORM).

## Memory

RSS grew +0.2 MB across 320 predictions; the drift window holds 200/200 observations.

## Protocol

Warmed before timing; repeated runs on CPU; medians across interleaved rounds. Per-image steady-state numbers exclude the composite drift evaluation, which runs once per window and is timed separately and amortised. The scope study disables exactly one capability at a time on the same pipeline instance.
