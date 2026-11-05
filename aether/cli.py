"""AETHER-ML command line interface.

The CLI is a thin shell over the same ``ReliabilityPipeline`` the HTTP service uses, so a local
prediction and a served prediction produce identical evidence.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from aether.config import load_config
from aether.imaging import centre_crop_square, decode_image
from aether.pipeline import ReliabilityPipeline
from aether.runtime import configure_runtime, resolve_device
from aether.utils.logging import configure_logging


def _add_config(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", default=None, help="path to a YAML config (default: configs/aether.yaml)")


def _predict(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    configure_logging(config.monitoring.log_level, config.monitoring.json_logs)
    configure_runtime(config.runtime)
    payload = Path(args.image).read_bytes()
    decoded = decode_image(
        payload,
        max_bytes=config.storage.upload_max_bytes,
        max_pixels=config.storage.upload_max_pixels,
    )
    pipeline = ReliabilityPipeline(config, device=resolve_device(config.runtime.device))
    result = pipeline.predict_image(centre_crop_square(decoded.pixels))
    if args.json:
        print(json.dumps(result.to_dict(), indent=2))
    else:
        print(f"action           {result.action}")
        print(f"prediction       {result.predicted_label} ({result.predicted_index})")
        print(f"confidence       {result.uncertainty['max_probability']:.4f}")
        print(f"entropy (nats)   {result.uncertainty['predictive_entropy']:.4f}")
        print(f"error risk       {result.risk['value']:.4f} ({result.risk['source']})")
        print(f"drift            {result.drift['severity']} (score {result.drift['score']:.2f})")
        print(f"why              {result.decision.get('explanation', '')}")
        if result.diagnostics:
            print("diagnostics")
            for finding in result.diagnostics:
                print(f"  {finding['severity']:8s} {finding['code']}: {finding['explanation']}")
    pipeline.registry.close()
    return 0


def _status(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    checks = {
        "lightweight checkpoint": config.resolve(config.models.lightweight.checkpoint),
        "advanced checkpoint": config.resolve(config.models.advanced.checkpoint),
        "temperature artifact": config.resolve(config.uncertainty.temperature_artifact),
        "drift reference": config.resolve(config.drift.reference_artifact),
        "learned policy": config.resolve(config.routing.learned_policy.artifact),
    }
    missing = 0
    print(f"{config.application.name} {config.application.version} ({config.application.environment})")
    print(f"policy={config.routing.policy} device={config.runtime.device} seed={config.runtime.seed}")
    for label, path in checks.items():
        exists = path.exists()
        missing += 0 if exists else 1
        size = f"{path.stat().st_size / 1e6:.2f} MB" if exists else "missing"
        print(f"  {'ok     ' if exists else 'MISSING'} {label:24s} {size}")
    if missing:
        print("\nRun `make data train calibrate reference` to materialise the missing artifacts.")
    return 0 if missing == 0 else 1


def _serve(args: argparse.Namespace) -> int:
    import uvicorn

    uvicorn.run(
        "backend.app.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_config=None,
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="aether", description="AETHER-ML reliability tooling")
    sub = parser.add_subparsers(dest="command", required=True)

    predict = sub.add_parser("predict", help="run the reliability pipeline on one image")
    predict.add_argument("image")
    predict.add_argument("--json", action="store_true", help="emit the full evidence payload")
    _add_config(predict)
    predict.set_defaults(func=_predict)

    status = sub.add_parser("status", help="report which artifacts are present")
    _add_config(status)
    status.set_defaults(func=_status)

    serve = sub.add_parser("serve", help="start the HTTP API")
    serve.add_argument("--host", default="0.0.0.0")  # noqa: S104 - container default
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--reload", action="store_true")
    serve.set_defaults(func=_serve)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except FileNotFoundError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
