#!/usr/bin/env python3
"""Fine-tune a torchvision classifier on the CIFAR-10 training split.

Design notes
------------
* ImageNet weights initialise the backbone; the classifier head is always fresh.
* The validation split drives model selection only. Calibration, routing and holdout splits are
  never touched here, and the official test split is never loaded.
* Batch-level augmentation keeps the CPU training loop fast enough to be practical.
* Every run writes a machine-readable report (config, split fingerprints, per-epoch metrics,
  wall clock, checkpoint digest) to ``artifacts/reports/``.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from aether.adapters.torchvision_adapter import build_architecture  # noqa: E402
from aether.config import load_config  # noqa: E402
from aether.data.cifar import (  # noqa: E402
    CIFAR10_CLASSES,
    build_and_persist_splits,
    load_cifar10,
)
from aether.data.preprocessing import (  # noqa: E402
    CifarSplitDataset,
    prepare_model_input,
)
from aether.runtime import configure_runtime  # noqa: E402
from aether.utils.io import atomic_write_json, seed_everything, sha256_file  # noqa: E402
from aether.utils.logging import configure_logging, get_logger  # noqa: E402

logger = get_logger("aether.train")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an AETHER-ML CIFAR-10 model.")
    parser.add_argument("--config", default=None, help="path to aether.yaml")
    parser.add_argument("--role", choices=["lightweight", "advanced"], default="lightweight")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3, help="classifier-head learning rate")
    parser.add_argument("--backbone-lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--limit-train", type=int, default=0, help="debug: cap training images")
    parser.add_argument("--limit-val", type=int, default=0, help="debug: cap validation images")
    parser.add_argument("--tag", default="")
    parser.add_argument("--device", default=None)
    return parser.parse_args(argv)


def build_loaders(
    config,
    splits,
    images: np.ndarray,
    labels: np.ndarray,
    batch_size: int,
    num_workers: int,
    seed: int,
    limit_train: int = 0,
    limit_val: int = 0,
) -> tuple[DataLoader, DataLoader]:
    train_indices = splits["train"]
    val_indices = splits["validation"]
    if limit_train:
        train_indices = train_indices[:limit_train]
    if limit_val:
        val_indices = val_indices[:limit_val]
    train_ds = CifarSplitDataset(images, labels, train_indices)
    val_ds = CifarSplitDataset(images, labels, val_indices)
    generator = torch.Generator()
    generator.manual_seed(seed)
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=False,
        persistent_workers=num_workers > 0,
        prefetch_factor=4 if num_workers > 0 else None,
        generator=generator,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config.data.eval_batch_size,
        shuffle=False,
        num_workers=max(1, num_workers // 2),
    )
    return train_loader, val_loader


def parameter_groups(model: nn.Module, head_lr: float, backbone_lr: float, weight_decay: float):
    head_names = {"fc", "classifier"}
    head_params, backbone_params = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if any(name.startswith(head) for head in head_names):
            head_params.append(param)
        else:
            backbone_params.append(param)
    return [
        {"params": backbone_params, "lr": backbone_lr, "weight_decay": weight_decay},
        {"params": head_params, "lr": head_lr, "weight_decay": weight_decay},
    ]


def _warmup_cosine(step: int, total: int, warmup: int) -> float:
    if step < warmup:
        return (step + 1) / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    image_size: int,
) -> dict[str, float]:
    model.eval()
    correct = 0
    total = 0
    loss_sum = 0.0
    criterion = nn.CrossEntropyLoss(reduction="sum")
    for images, labels in loader:
        batch = prepare_model_input(images, image_size).to(device)
        labels = labels.to(device)
        logits = model(batch)
        loss_sum += float(criterion(logits, labels))
        correct += int((logits.argmax(dim=1) == labels).sum())
        total += labels.numel()
    model.train()
    if total == 0:
        return {"accuracy": float("nan"), "loss": float("nan"), "samples": 0}
    return {"accuracy": correct / total, "loss": loss_sum / total, "samples": total}


def train(args: argparse.Namespace) -> dict:
    config = load_config(args.config)
    configure_logging(config.monitoring.log_level, config.monitoring.json_logs)
    runtime_config = (
        config.runtime
        if args.device is None
        else config.runtime.model_copy(update={"device": args.device})
    )
    runtime = configure_runtime(runtime_config)
    seed = args.seed if args.seed is not None else config.runtime.seed
    seed_everything(seed, config.runtime.deterministic)
    num_workers = args.num_workers if args.num_workers is not None else config.data.num_workers
    image_size = args.image_size or config.data.image_size
    spec = (
        config.models.lightweight if args.role == "lightweight" else config.models.advanced
    )

    data = load_cifar10(config.data_root)
    splits = build_and_persist_splits(
        config.data_root, data.train_labels, config.data.splits, config.data.split_seed,
        data.fingerprint,
    )
    train_loader, val_loader = build_loaders(
        config, splits, data.train_images, data.train_labels,
        args.batch_size, num_workers, seed, args.limit_train, args.limit_val,
    )

    model = build_architecture(spec.architecture, spec.num_classes, pretrained=spec.pretrained)
    model.to(runtime.device)
    model.train()
    parameters = int(sum(p.numel() for p in model.parameters()))
    logger.info(
        "training started",
        extra={
            "role": args.role,
            "architecture": spec.architecture,
            "parameters": parameters,
            "device": str(runtime.device),
            "train_images": len(train_loader.dataset),
            "val_images": len(val_loader.dataset),
            "image_size": image_size,
            "epochs": args.epochs,
        },
    )

    optimizer = torch.optim.AdamW(
        parameter_groups(model, args.lr, args.backbone_lr, args.weight_decay)
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    steps_per_epoch = max(1, len(train_loader))
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = max(10, steps_per_epoch // 2)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: _warmup_cosine(step, total_steps, warmup_steps)
    )
    amp_enabled = runtime.device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    augment_generator = torch.Generator()
    augment_generator.manual_seed(seed + 1)

    checkpoint_path = config.resolve(spec.checkpoint)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    report_path = config.resolve(f"artifacts/reports/training_{args.role}.json")
    history: list[dict] = []
    best_accuracy = -1.0
    best_epoch = -1
    started = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        seen = 0
        epoch_started = time.time()
        for step, (images, labels) in enumerate(train_loader, start=1):
            batch = prepare_model_input(
                images, image_size, train=True, generator=augment_generator
            ).to(runtime.device)
            labels = labels.to(runtime.device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=runtime.device.type, enabled=amp_enabled):
                logits = model(batch)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            running_loss += float(loss.detach()) * labels.numel()
            seen += labels.numel()
            if step % 20 == 0 or step == steps_per_epoch:
                logger.info(
                    "train step",
                    extra={
                        "epoch": epoch,
                        "step": step,
                        "steps": steps_per_epoch,
                        "loss": round(running_loss / max(1, seen), 4),
                        "lr": round(optimizer.param_groups[-1]["lr"], 6),
                    },
                )
            if args.limit_train and seen >= args.limit_train:
                break
        validation = evaluate(model, val_loader, runtime.device, image_size)
        epoch_record = {
            "epoch": epoch,
            "train_loss": running_loss / max(1, seen),
            "train_samples": seen,
            "val_accuracy": validation["accuracy"],
            "val_loss": validation["loss"],
            "epoch_seconds": round(time.time() - epoch_started, 2),
        }
        history.append(epoch_record)
        logger.info("epoch complete", extra=epoch_record)
        # Crash-safe progress: an interrupted run still leaves a truthful partial report and the
        # best checkpoint so far, instead of losing all evidence.
        atomic_write_json(
            report_path,
            {
                "status": "running",
                "role": args.role,
                "architecture": spec.architecture,
                "seed": seed,
                "image_size": image_size,
                "epochs_planned": args.epochs,
                "history": history,
                "best_val_accuracy": best_accuracy,
                "best_epoch": best_epoch,
                "checkpoint": str(checkpoint_path),
                "tag": args.tag,
            },
        )
        if validation["accuracy"] > best_accuracy:
            best_accuracy = validation["accuracy"]
            best_epoch = epoch
            payload = {
                "state_dict": model.state_dict(),
                "metadata": {
                    "architecture": spec.architecture,
                    "role": args.role,
                    "version": spec.version,
                    "num_classes": spec.num_classes,
                    "input_size": image_size,
                    "classes": list(CIFAR10_CLASSES),
                    "val_accuracy": best_accuracy,
                    "epoch": epoch,
                    "seed": seed,
                    "trained_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "dataset": "cifar10",
                },
            }
            torch.save(payload, checkpoint_path)

    wall_seconds = round(time.time() - started, 2)
    report = {
        "status": "completed",
        "role": args.role,
        "architecture": spec.architecture,
        "parameters": parameters,
        "device": runtime.to_dict(),
        "seed": seed,
        "image_size": image_size,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "optimizer": {
            "name": "AdamW",
            "head_lr": args.lr,
            "backbone_lr": args.backbone_lr,
            "weight_decay": args.weight_decay,
            "schedule": "linear-warmup + cosine",
            "label_smoothing": args.label_smoothing,
        },
        "splits": {
            "train": len(train_loader.dataset),
            "validation": len(val_loader.dataset),
        },
        "dataset_fingerprint": data.fingerprint,
        "history": history,
        "best_val_accuracy": best_accuracy,
        "best_epoch": best_epoch,
        "wall_clock_seconds": wall_seconds,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path) if checkpoint_path.exists() else None,
        "tag": args.tag,
    }
    atomic_write_json(report_path, report)
    logger.info(
        "training finished",
        extra={
            "best_val_accuracy": round(best_accuracy, 4),
            "best_epoch": best_epoch,
            "wall_clock_seconds": wall_seconds,
        },
    )

    if config.monitoring.experiment_tracking == "mlflow":
        _log_mlflow(config, report, checkpoint_path)
    return report


def _log_mlflow(config, report: dict, checkpoint_path: Path) -> None:
    """Log the run to MLflow when configured; failures are recorded, never hidden."""
    try:
        import mlflow

        mlflow.set_tracking_uri(config.monitoring.mlflow_tracking_uri)
        mlflow.set_experiment(config.monitoring.mlflow_experiment)
        with mlflow.start_run(run_name=f"train-{report['role']}-{report['seed']}"):
            mlflow.log_params(
                {
                    "role": report["role"],
                    "architecture": report["architecture"],
                    "epochs": report["epochs"],
                    "batch_size": report["batch_size"],
                    "image_size": report["image_size"],
                    "seed": report["seed"],
                    **report["optimizer"],
                }
            )
            for record in report["history"]:
                mlflow.log_metrics(
                    {
                        "train_loss": record["train_loss"],
                        "val_accuracy": record["val_accuracy"],
                        "val_loss": record["val_loss"],
                    },
                    step=record["epoch"],
                )
            mlflow.log_metric("best_val_accuracy", report["best_val_accuracy"])
            if checkpoint_path.exists():
                mlflow.log_artifact(str(checkpoint_path))
        logger.info("mlflow run recorded")
    except Exception as exc:  # tracking must never break training
        logger.warning("mlflow logging failed", extra={"error": str(exc)})
        report["tracking_error"] = str(exc)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = train(args)
    print(
        f"\nbest val accuracy: {report['best_val_accuracy']:.4f} "
        f"(epoch {report['best_epoch']}), wall {report['wall_clock_seconds']}s\n"
        f"checkpoint: {report['checkpoint']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
