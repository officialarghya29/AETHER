"""Fail when README numbers drift from the recorded stability snapshot.

The stability study writes ``docs/data/stability-snapshot.json``; this check parses the README
tables and verifies every published threshold, coverage, error and mirror-pass figure matches it.
Regenerating the snapshot is therefore a deliberate act: a number cannot quietly change in the
prose without the recorded evidence changing too.

Inside CI (``GITHUB_STEP_SUMMARY`` set) the verified figures are also written to the job summary,
so a green run shows what was checked rather than only that something passed.

Run in CI and locally with ``python experiments/check_docs.py``.
"""

from __future__ import annotations

import json
import os
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


def _compare(
    label: str,
    expected: str,
    actual: str,
    problems: list[str],
    verified: list[tuple[str, str]],
) -> None:
    if expected == actual:
        verified.append((label, expected))
    else:
        problems.append(f"{label}: README has {actual!r}, snapshot has {expected!r}")


def _write_summary(verified: list[tuple[str, str]], generated_at: str) -> None:
    target = os.environ.get("GITHUB_STEP_SUMMARY")
    if not target:
        return
    lines = [
        "## Stability snapshot verified",
        "",
        "`README.md` matches `docs/data/stability-snapshot.json` "
        f"(stability run {generated_at}). Regenerate with `make stability`.",
        "",
        "| figure | value |",
        "| --- | --- |",
    ]
    lines += [f"| {label} | {value} |" for label, value in verified]
    with Path(target).open("a", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def main() -> int:
    if not SNAPSHOT.exists():
        print(f"missing snapshot: {SNAPSHOT}; run `make stability`")
        return 1
    snapshot: dict[str, Any] = json.loads(SNAPSHOT.read_text())
    readme = README.read_text()
    rows = _rows(readme)
    problems: list[str] = []
    verified: list[tuple[str, str]] = []

    for tier, entries in snapshot["risk_control"].items():
        for entry in entries:
            alpha = f"{entry['alpha']:.2f}"
            matches = [row for row in rows if row[:2] == [tier, alpha] and len(row) >= 6]
            if not matches:
                problems.append(f"risk control row missing for {tier} alpha={alpha}")
                continue
            row = matches[0]
            _compare(
                f"{tier} alpha={alpha} threshold",
                _fmt(entry["threshold"]),
                row[2],
                problems,
                verified,
            )
            _compare(
                f"{tier} alpha={alpha} clean coverage",
                _fmt(entry["clean_coverage"]),
                row[3],
                problems,
                verified,
            )
            _compare(
                f"{tier} alpha={alpha} clean error",
                _fmt(entry["clean_error"]),
                row[4],
                problems,
                verified,
            )
            _compare(
                f"{tier} alpha={alpha} corrupted error",
                _fmt(entry["corrupted_error"]),
                row[5],
                problems,
                verified,
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
                verified,
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
        _compare(f"TTA {label} identity", f"{entry['identity']:.4f}", row[1], problems, verified)
        _compare(f"TTA {label} accuracy", f"{entry['tta']:.4f}", row[2], problems, verified)
        _compare(f"TTA {label} delta", f"{entry['delta']:+.4f}", row[3], problems, verified)
        if "mirror_pass_ms" in entry:
            _compare(
                f"TTA {label} mirror pass",
                f"{entry['mirror_pass_ms']:.2f} ms",
                row[4],
                problems,
                verified,
            )

    gating = snapshot["tta"].get("drift_gating")
    if gating:
        phrase = (
            f"{gating['clean_extra_pass_rate'] * 100:.0f}% of clean windows and "
            f"{gating['corrupted_extra_pass_rate'] * 100:.0f}% of severity-3 noisy windows"
        )
        if phrase in " ".join(readme.split()):
            verified.append(
                (
                    "TTA gating rates",
                    f"{gating['clean_extra_pass_rate'] * 100:.0f}% clean / "
                    f"{gating['corrupted_extra_pass_rate'] * 100:.0f}% severity-3",
                )
            )
        else:
            problems.append(f"TTA gating prose: README does not state {phrase!r}")

    if problems:
        for problem in problems:
            print(f"DRIFT: {problem}")
        return 1
    print(f"README matches the recorded stability snapshot ({len(verified)} figures)")
    _write_summary(verified, str(snapshot.get("generated_at", "recorded run")))
    return 0


if __name__ == "__main__":
    sys.exit(main())
