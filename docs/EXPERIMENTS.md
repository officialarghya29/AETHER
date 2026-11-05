# Experiments — AETHER-ML

## 1. Split discipline

CIFAR-10's 50,000-image training partition is split once, stratified, with seed 42:

| Split | Size | Used for |
| --- | --- | --- |
| train | 32,000 | weight updates |
| validation | 4,000 | checkpoint selection |
| calibration | 4,000 | temperature scaling only |
| routing_dev | 5,000 | risk models, policy development, drift thresholds |
| holdout | 5,000 | frozen drift reference window |

The official 10,000-image test split is **reporting only**. No threshold, temperature or policy
parameter is fitted on it. Split indices and dataset fingerprints are written to
`data/processed/splits.json` and `data/raw/cifar10.json`.

## 2. Models

| Role | Architecture | Initialisation | Image size | Epochs |
| --- | --- | --- | --- | --- |
| lightweight | MobileNetV3-Small | ImageNet weights, classifier replaced for 10 classes | 64 | 10 |
| advanced | ResNet50 | ImageNet weights, classifier replaced for 10 classes | 64 | 6 |

Training uses AdamW, cosine schedule, label smoothing and light augmentation. Checkpoints are
written with architecture, class list, input size, version and validation accuracy.

## 3. Commands

```bash
make data          # download + fingerprint CIFAR-10
make train         # fine-tune both models
make reference     # freeze the drift reference from the holdout split
make calibrate     # temperatures + risk models + learned policy artifact
make benchmarks    # clean, robustness, drift, routing and ablations
```

Individual stages:

```bash
python training/train.py --role lightweight --epochs 10
python training/train.py --role advanced --epochs 6
python training/build_reference.py --config configs/aether.yaml
python training/calibrate.py --config configs/aether.yaml
python experiments/run_benchmarks.py --config configs/aether.yaml
```

`run_benchmarks.py` writes `artifacts/reports/benchmarks.json` and
`artifacts/reports/BENCHMARKS.md`. Every table in the README is copied from that report; it is not
typed by hand.

## 4. What is measured

**Classification:** accuracy, macro precision/recall/F1, per-class support.

**Calibration:** ECE (15 bins), MCE, Brier, NLL, reliability curve, risk–coverage and AURC.

**Robustness:** 10 corruption families × 5 severities on a fixed stratified subsample of 2,500 test
images. Accuracy drop is measured against the same subsample's clean accuracy, and ECE is reported
per condition.

**Drift:** windows of the configured size over clean development data (false-positive reference) and
over corrupted development data at severities 1, 3 and 5. A window counts as detected when the
composite score reaches `moderate` or `severe`. This is shift detection — no accuracy claim is made
for unlabelled data.

**Routing:** 5,000 clean development samples plus 1,500 corrupted samples from two families. Policies:
always-lightweight, always-advanced, fixed confidence threshold, uncertainty-only, shift-aware and
AETHER. Reported: coverage, accuracy on answered samples, escalation/review/abstention rates, mean
latency and expected cost per sample.

**Ablations:** no uncertainty, no drift signal, no cost awareness, no disagreement, no escalation,
and a clean-only subset.

## 5. Reading the results honestly

- Latency numbers describe the recorded CPU and thread count in
  `artifacts/reports/benchmarks.json → environment`. They are not portable benchmarks.
- Routing cost is a configured utility, not currency. Changing `routing.costs` changes the optimum.
- Human review is simulated with `reviewer_error_rate = 0.02`; the real error rate of a reviewer is
  not measured here.
- The learned policy is a logistic model over six features; it is a defensible baseline, not a claim
  of a new routing method.
- Corruption families not implemented are listed explicitly in `docs/DECISIONS.md`.

## 6. Reproducibility checklist

1. Dataset fingerprints match `data/raw/cifar10.json`.
2. `runtime.seed = 42`, `data.split_seed = 42`, deterministic torch settings enabled.
3. Calibration artifact records the split and sample count.
4. Reference artifact records source, sample count and explained variance.
5. Benchmark report records timestamps, environment, seeds and protocol constants.
6. Re-running `make benchmarks` regenerates both report files.
