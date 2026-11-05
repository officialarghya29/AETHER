# Data

CIFAR-10 is downloaded by `scripts/download_cifar.py` and materialised as
`data/raw/cifar10.npz`, alongside `data/raw/cifar10.json`, which records the source, byte-level
fingerprints and split manifest. Datasets and derived artifacts are excluded from version control.

```
make data
```

Derived files:

| path | contents |
| --- | --- |
| `data/raw/cifar10.npz` | uint8 NHWC train/test images and int64 labels from the official release |
| `data/raw/cifar10.json` | source, SHA-256 fingerprints, split sizes and seed |
| `data/processed/splits.json` | stratified split indices for train/validation/calibration/routing_dev/holdout |

The official test split is used for reporting only. Calibration uses the calibration split, routing
policy development uses the routing_dev split, and the drift reference is built from the holdout
split, so no evaluation set leaks into tuning.
