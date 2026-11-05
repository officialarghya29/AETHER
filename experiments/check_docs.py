"""Fail when README numbers drift from the recorded stability snapshot.

The stability study writes ``docs/data/stability-snapshot.json``; this check parses the README
tables and verifies every published threshold, coverage and error figure matches it. Regenerating
the snapshot is therefore a deliberate act: a number cannot quietly change in the prose without
the recorded evidence changing too.

Run in CI and locally with ``python experiments/check_docs.py``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT = ROOT / "docs" / "data" / "stability-snapshot.json"
README = ROOT / "README.md"


def _fmt(value: float | None) -> str:
    return "—" if value is None else f"{value:.3f}"


def _rows(readme: str) -> list[list[str]]:
    rows = []
    for line in readme.splitlines():
        if not line.startswith("|"):
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if cells and set("".join(cells)) <= {"-", " "}:
            continue
        rows.append(cells)
    return rows


def _compare(label: str, expected: str, actual: str, problems: list[str]) -> None:
    if expected != actual:
        problems.append(f"{label}: README has {actual!r}, snapshot has {expected!r}")


def main() -> int:
    if not SNAPSHOT.exists():
        print(f"missing snapshot: {SNAPSHOT}; run `make stability`")
        return 1
    snapshot: dict[str, Any] = json.loads(SNAPSHOT.read_text())
    rows = _rows(README.read_text())
    problems: list[str] = []

    for tier, entries in snapshot["risk_control"].items():
        for entry in entries:
            alpha = f"{entry['alpha']:.2f}"
            matches = [row for row in rows if row[:2] == [tier, alpha] and len(row) >= 6]
            if not matches:
                problems.append(f"risk control row missing for {tier} alpha={alpha}")
                continue
            row = matches[0]
            _compare(f"{tier} alpha={alpha} threshold", _fmt(entry["threshold"]), row[2], problems)
            _compare(
                f"{tier} alpha={alpha} clean coverage",
                _fmt(entry["clean_coverage"]),
                row[3],
                problems,
            )
            _compare(
                f"{tier} alpha={alpha} clean error", _fmt(entry["clean_error"]), row[4], problems
            )
            _compare(
                f"{tier} alpha={alpha} corrupted error",
                _fmt(entry["corrupted_error"]),
                row[5],
                problems,
            )

    for entry in snapshot["severity_sweep"]:
        if entry["alpha"] != 0.10:
            continue
        matches = [
            row
            for row in rows
            if row[0] == entry["tier"] and len(row) == 4 and "% / " in row[1]
        ]
        if not matches:
            problems.append(f"severity sweep row missing for {entry['tier']}")
            continue
        row = matches[0]
        for index, stream in enumerate(entry["streams"]):
            expected = f"{stream['coverage'] * 100:.0f}% / {_fmt(stream['error'])}"
            _compare(
                f"{entry['tier']} severity {stream['severity']}",
                expected,
                row[index + 1],
                problems,
            )

    labels = {
        "clean": "clean routing-dev",
        "corrupted_gaussian_noise_severity_3": "gaussian noise, severity 3",
    }
    for key, label in labels.items():
        if key not in snapshot["tta"]:
            continue
        entry = snapshot["tta"][key]
        matches = [row for row in rows if row and row[0] == label and len(row) >= 5]
        if not matches:
            problems.append(f"TTA row missing for {label}")
            continue
        row = matches[0]
        _compare(f"TTA {label} identity", f"{entry['identity']:.4f}", row[1], problems)
        _compare(f"TTA {label} accuracy", f"{entry['tta']:.4f}", row[2], problems)
        _compare(f"TTA {label} delta", f"{entry['delta']:+.4f}", row[3], problems)

    if problems:
        for problem in problems:
            print(f"DRIFT: {problem}")
        return 1
    print("README matches the recorded stability snapshot")
    return 0


if __name__ == "__main__":
    sys.exit(main())
