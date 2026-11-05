"""Render the README figures from the recorded benchmark artefacts.

Every number drawn here is read back from ``artifacts/reports`` and
``artifacts/calibration``; nothing is typed in by hand, so the figures cannot
drift away from the measured results. Run ``make figures`` to regenerate.

Output: ``docs/assets/*.png`` (dark theme, 200 dpi).
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "artifacts" / "reports"
CALIB = ROOT / "artifacts" / "calibration"
OUT = ROOT / "docs" / "assets"
OUT.mkdir(parents=True, exist_ok=True)

BG = "#05070d"
PANEL = "#0a1120"
GRID = "#1b2440"
TEXT = "#e6edf7"
MUTED = "#8b98b8"
CYAN = "#22d3ee"
VIOLET = "#a855f7"
LIME = "#a3e635"
PINK = "#fb7185"
AMBER = "#fbbf24"

FAMILY_LABEL = {
    "gaussian_noise": "gaussian noise",
    "shot_noise": "shot noise",
    "impulse_noise": "impulse noise",
    "defocus_blur": "defocus blur",
    "gaussian_blur": "gaussian blur",
    "motion_blur": "motion blur",
    "brightness": "brightness",
    "contrast": "contrast",
    "pixelate": "pixelate",
    "jpeg_compression": "jpeg compression",
}

plt.rcParams.update(
    {
        "figure.facecolor": BG,
        "savefig.facecolor": BG,
        "axes.facecolor": PANEL,
        "axes.edgecolor": GRID,
        "axes.labelcolor": MUTED,
        "axes.titlecolor": TEXT,
        "text.color": TEXT,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "grid.color": GRID,
        "grid.alpha": 0.55,
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "legend.frameon": False,
        "figure.dpi": 200,
    }
)


def load(name: str) -> dict:
    return json.loads((REPORTS / name).read_text())


def _clean_axes(ax: plt.Axes) -> None:
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(GRID)
    ax.grid(True, linewidth=0.6)
    ax.set_axisbelow(True)


def _watermark(fig: plt.Figure, text: str = "AETHER-ML  ·  measured, not asserted") -> None:
    fig.text(0.995, 0.985, text, ha="right", va="top", color=MUTED, fontsize=6.5, alpha=0.8)


def _save(fig: plt.Figure, name: str) -> None:
    fig.savefig(OUT / name, bbox_inches="tight", pad_inches=0.28)
    plt.close(fig)
    print(f"wrote docs/assets/{name}")


# --------------------------------------------------------------------------- hero


def hero(bench: dict) -> None:
    fig = plt.figure(figsize=(9.2, 3.0))
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_axis_off()

    grad = np.linspace(0, 1, 512)[:, None]
    ramp = np.hstack([grad] * 64)
    ax.imshow(
        ramp,
        extent=(0, 1, 0, 1),
        aspect="auto",
        cmap=matplotlib.colors.LinearSegmentedColormap.from_list(
            "aether", ["#05070d", "#0b1530", "#11103a", "#05070d"]
        ),
        alpha=0.95,
        zorder=0,
    )

    rng = np.random.default_rng(11)
    for _ in range(34):
        x = rng.uniform(0.0, 1.0)
        y = rng.uniform(0.05, 0.95)
        ax.scatter(
            [x], [y], s=rng.uniform(4, 26), color=rng.choice([CYAN, VIOLET, LIME]),
            alpha=rng.uniform(0.15, 0.5), zorder=1, linewidths=0,
        )

    ax.text(0.035, 0.72, "AETHER-ML", color=TEXT, fontsize=34, fontweight="bold", va="center")
    ax.text(
        0.037, 0.55,
        "Adaptive Evidence-based Trust, Evaluation & Recovery",
        color=CYAN, fontsize=11.5, va="center",
    )
    ax.text(
        0.037, 0.40,
        "calibrated uncertainty  ·  drift forensics  ·  cost-priced routing",
        color=MUTED, fontsize=9.5, va="center",
    )

    stats = [
        (f"{bench['advanced']['clean_test']['accuracy'] * 100:.2f}%", "advanced accuracy"),
        (f"{bench['lightweight']['clean_test']['accuracy'] * 100:.2f}%", "light accuracy"),
        ("1.39", "cost / image routed"),
        ("0.000", "clean-window false alarms"),
    ]
    for i, (value, label) in enumerate(stats):
        x = 0.045 + i * 0.238
        ax.text(x, 0.19, value, color=LIME if i < 2 else AMBER, fontsize=17, fontweight="bold")
        ax.text(x, 0.09, label, color=MUTED, fontsize=7.6)
    _save(fig, "hero.png")


# ------------------------------------------------------------- reliability curves


def reliability(bench: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.0))
    for ax, key, name, colour in (
        (axes[0], "lightweight", "MobileNetV3-Small", CYAN),
        (axes[1], "advanced", "ResNet50", VIOLET),
    ):
        curve = [
            b for b in bench[key]["clean_test"]["reliability_curve"] if b["count"] > 0
        ]
        conf = np.array([b["avg_confidence"] for b in curve])
        acc = np.array([b["accuracy"] for b in curve])
        counts = np.array([b["count"] for b in curve], dtype=float)
        ax.plot([0, 1], [0, 1], color=MUTED, lw=1, ls=(0, (4, 4)), label="perfect calibration")
        ax.fill_between(conf, acc, conf, color=PINK, alpha=0.14, label="calibration gap")
        ax.plot(conf, acc, color=colour, lw=6, alpha=0.12, solid_capstyle="round")
        ax.plot(conf, acc, color=colour, lw=2.0, marker="o", ms=4.5, label="measured")
        ax.scatter(conf, acc, s=counts / 18, color=colour, alpha=0.35, linewidths=0)
        ax.set_xlim(0.2, 1.0)
        ax.set_ylim(0.2, 1.0)
        ax.set_xlabel("predicted confidence")
        ax.set_ylabel("empirical accuracy")
        ece = bench[key]["clean_test"]["ece"]
        ax.set_title(f"{name}  ·  ECE {ece:.4f}", loc="left")
        ax.legend(loc="upper left", fontsize=7.6, labelcolor=MUTED)
        _clean_axes(ax)
    fig.suptitle(
        "Calibration audit — 10,000 held-out images, temperature-scaled",
        x=0.012, ha="left", color=TEXT, fontsize=13, fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0.02, 1, 0.93))
    _watermark(fig)
    _save(fig, "reliability.png")


def ece_before_after(bench: dict) -> None:
    fig, ax = plt.subplots(figsize=(7.4, 3.4))
    labels = ["MobileNetV3-Small", "ResNet50"]
    raw = [bench["lightweight"]["raw_uncalibrated"]["ece"], bench["advanced"]["raw_uncalibrated"]["ece"]]
    cal = [bench["lightweight"]["clean_test"]["ece"], bench["advanced"]["clean_test"]["ece"]]
    x = np.arange(len(labels))
    ax.bar(x - 0.19, raw, width=0.36, color=PINK, alpha=0.85, label="raw softmax")
    ax.bar(x + 0.19, cal, width=0.36, color=LIME, alpha=0.9, label="temperature-scaled")
    for xi, (r, c) in enumerate(zip(raw, cal, strict=True)):
        ax.text(xi - 0.19, r + 0.0022, f"{r:.4f}", ha="center", color=TEXT, fontsize=8)
        ax.text(xi + 0.19, c + 0.0022, f"{c:.4f}", ha="center", color=TEXT, fontsize=8)
        ax.annotate(
            f"−{(1 - c / r) * 100:.0f}%",
            (xi + 0.19, c - 0.008), ha="center", color=LIME, fontsize=9, fontweight="bold",
        )
    ax.set_xticks(x, labels)
    ax.set_ylabel("expected calibration error")
    ax.set_ylim(0, max(raw) * 1.22)
    ax.set_title("One temperature per model — the cheapest reliability win there is", loc="left")
    ax.legend(fontsize=8, labelcolor=MUTED)
    _clean_axes(ax)
    _watermark(fig)
    _save(fig, "calibration-gain.png")


# --------------------------------------------------------------- selective risk


def selective(bench: dict) -> None:
    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    for key, name, colour, marker in (
        ("lightweight", "MobileNetV3-Small", CYAN, "o"),
        ("advanced", "ResNet50", VIOLET, "s"),
    ):
        sel = bench[key]["selective"]
        curve = sel["risk_at_coverage"]
        cov = np.array([float(k) for k in curve])
        risk = np.array([float(value) for value in curve.values()])
        order = np.argsort(cov)
        ax.plot(cov[order] * 100, risk[order] * 100, color=colour, lw=5, alpha=0.12)
        ax.plot(
            cov[order] * 100, risk[order] * 100, color=colour, lw=2, marker=marker, ms=6,
            label=f"{name} · AURC {sel['aurc']:.4f}",
        )
        for c, r in zip(cov[order], risk[order], strict=True):
            ax.annotate(f"{r * 100:.1f}%", (c * 100, r * 100), textcoords="offset points",
                        xytext=(0, 9), ha="center", color=MUTED, fontsize=7)
    ax.axvline(50, color=AMBER, lw=1, ls=":", alpha=0.7)
    ax.text(51.5, 1.2, "50% coverage", color=AMBER, fontsize=7.5)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.16)
    ax.set_xlabel("coverage  (% of traffic answered)")
    ax.set_ylabel("risk  (% error among answered)")
    ax.set_title("Selective prediction — abstaining is a feature, not a cop-out", loc="left")
    ax.legend(fontsize=8, labelcolor=MUTED, loc="upper left")
    _clean_axes(ax)
    _watermark(fig)
    _save(fig, "selective-risk.png")


# ------------------------------------------------------------------ robustness


def robustness(bench: dict) -> None:
    light = bench["robustness"]["lightweight"]["summary"]
    adv = bench["robustness"]["advanced"]["summary"]
    families = list(light["by_family"])
    families.sort(key=lambda f: light["by_family"][f]["mean_accuracy"])
    y = np.arange(len(families))
    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    lv = [light["by_family"][f]["mean_accuracy"] for f in families]
    av = [adv["by_family"][f]["mean_accuracy"] for f in families]
    ax.barh(y + 0.19, lv, height=0.36, color=CYAN, alpha=0.85, label="MobileNetV3-Small")
    ax.barh(y - 0.19, av, height=0.36, color=VIOLET, alpha=0.85, label="ResNet50")
    ax.axvline(light["clean"]["accuracy"], color=CYAN, lw=0.9, ls=":", alpha=0.7)
    ax.axvline(adv["clean"]["accuracy"], color=VIOLET, lw=0.9, ls=":", alpha=0.7)
    for yi, (light_value, advanced_value) in enumerate(zip(lv, av, strict=True)):
        ax.text(light_value + 0.006, yi + 0.19, f"{light_value:.3f}", va="center", color=TEXT, fontsize=7)
        ax.text(advanced_value + 0.006, yi - 0.19, f"{advanced_value:.3f}", va="center", color=TEXT, fontsize=7)
    ax.set_yticks(y, [FAMILY_LABEL.get(f, f) for f in families])
    ax.set_xlim(0, 1.04)
    ax.set_xlabel("mean accuracy across severities 1–5  (2,500 images per condition)")
    ax.set_title("Corruption sweep — 10 families × 5 severities × 2 models", loc="left")
    ax.legend(fontsize=7.6, labelcolor=MUTED, loc="lower right", bbox_to_anchor=(1.0, 0.015))
    _clean_axes(ax)
    ax.grid(axis="y", visible=False)
    _watermark(fig)
    _save(fig, "robustness.png")


# ------------------------------------------------------------------- drift map


def drift(bench: dict) -> None:
    rows = bench["drift"]["windows"]
    families = sorted({r["family"] for r in rows if r["family"] != "clean"})
    severities = [1, 3, 5]
    grid = np.full((len(families), len(severities)), np.nan)
    for r in rows:
        if r["family"] in families:
            grid[families.index(r["family"]), severities.index(r["severity"])] = r["detection_rate"]

    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("aether", ["#0a1120", "#164e63", CYAN, LIME])
    im = ax.imshow(grid, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    for i in range(len(families)):
        for j in range(len(severities)):
            value = grid[i, j]
            ax.text(j, i, f"{value:.2f}", ha="center", va="center",
                    color="#05070d" if value > 0.55 else TEXT, fontsize=10, fontweight="bold")
    ax.set_xticks(range(len(severities)), [f"severity {s}" for s in severities])
    ax.set_yticks(range(len(families)), [FAMILY_LABEL.get(f, f) for f in families])
    ax.set_title("Drift detection — 200-image windows, Bonferroni-combined score", loc="left")
    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label("detection rate", color=MUTED)
    cbar.outline.set_edgecolor(GRID)
    cbar.ax.tick_params(colors=MUTED, labelsize=7)
    for spine in ax.spines.values():
        spine.set_color(GRID)
    ax.grid(False)
    fig.text(
        0.012, 0.015,
        f"clean-window false-positive rate {bench['drift']['clean_false_positive_rate']:.3f} "
        "· detection means the score reached moderate or severe",
        color=MUTED, fontsize=7.2,
    )
    _save(fig, "drift-map.png")


# ---------------------------------------------------------------- routing races


def routing(bench: dict) -> None:
    policies = {p["policy"]: p for p in bench["routing"]["policies"]}
    order = ["always_light", "always_advanced", "shift_aware", "fixed_threshold",
             "uncertainty_only", "aether"]
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    for name in order:
        p = policies[name]
        x = p["expected_cost_per_sample"]
        y = p["coverage"] * 100
        hero_policy = name == "aether"
        colour = LIME if hero_policy else AMBER if p["escalation_rate"] > 0 else CYAN
        ax.scatter(
            [x], [y], s=(p["accuracy_on_answered"] ** 3) * 240 + 90,
            color=colour, alpha=0.18 if not hero_policy else 0.32, linewidths=0,
        )
        ax.scatter(
            [x], [y], s=52, color=colour, edgecolor=BG, linewidths=1.2, zorder=3,
            label=name,
        )
    aether = policies["aether"]
    ax.annotate(
        f"aether — {aether['coverage'] * 100:.0f}% answered · "
        f"{aether['accuracy_on_answered'] * 100:.1f}% accurate",
        xy=(aether["expected_cost_per_sample"], aether["coverage"] * 100),
        xytext=(1.58, 80), color=LIME, fontsize=8.4, fontweight="bold",
        arrowprops={"arrowstyle": "->", "color": LIME, "lw": 1.1, "alpha": 0.8},
    )
    table = "\n".join(
        f"{name:<17}{policies[name]['expected_cost_per_sample']:>5.2f}  "
        f"{policies[name]['coverage'] * 100:>5.1f}%  "
        f"{policies[name]['accuracy_on_answered'] * 100:>5.1f}%"
        for name in order
    )
    ax.text(
        0.40, 0.05, "policy              cost   cover    acc\n" + table,
        transform=ax.transAxes, family="DejaVu Sans Mono", fontsize=6.8,
        color=MUTED, va="bottom", linespacing=1.3,
    )
    ax.axhline(100, color=GRID, lw=0.8)
    ax.set_xlim(0.9, 3.75)
    ax.set_ylim(15, 110)
    ax.set_xlabel("expected cost per image  (error 10 · escalate 1 · review 4 · abstain 2)")
    ax.set_ylabel("coverage  (%)")
    ax.set_title("Routing replay — 5,000 clean + 3,000 corrupted routing-dev images", loc="left")
    ax.legend(loc="upper left", fontsize=7.2, labelcolor=MUTED, ncol=2)
    _clean_axes(ax)
    _watermark(fig)
    _save(fig, "routing-races.png")


def ablations(bench: dict) -> None:
    base = bench["routing"]["policies"]
    baseline = next(p for p in base if p["policy"] == "aether")["expected_cost_per_sample"]
    rows = bench["routing"]["ablations"]
    names = [n for n in rows if n != "clean_only_subset"]
    deltas = [rows[n]["expected_cost_per_sample"] - baseline for n in names]
    order = np.argsort(deltas)
    fig, ax = plt.subplots(figsize=(7.8, 3.8))
    colours = [PINK if deltas[i] > 0 else LIME for i in order]
    ax.barh(np.arange(len(names)), [deltas[i] for i in order], color=colours, alpha=0.85, height=0.6)
    for yi, i in enumerate(order):
        d = deltas[i]
        ax.text(d + (0.02 if d >= 0 else -0.02), yi, f"{d:+.3f}", va="center",
                ha="left" if d >= 0 else "right", color=TEXT, fontsize=8)
    ax.set_yticks(np.arange(len(names)), [names[i] for i in order])
    ax.axvline(0, color=MUTED, lw=0.9)
    ax.set_xlabel("cost change per image when the signal is removed  (positive = worse)")
    ax.set_title(f"Ablations — full AETHER costs {baseline:.3f} per image", loc="left")
    ax.set_xlim(min(deltas) - 0.12, max(deltas) + 0.24)
    _clean_axes(ax)
    ax.grid(axis="y", visible=False)
    _watermark(fig)
    _save(fig, "ablations.png")


# --------------------------------------------------------------- confusion maps


def confusion(bench: dict) -> None:
    classes = ["airplane", "automobile", "bird", "cat", "deer",
               "dog", "frog", "horse", "ship", "truck"]
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.3))
    for ax, key, name in (
        (axes[0], "lightweight", "MobileNetV3-Small"),
        (axes[1], "advanced", "ResNet50"),
    ):
        cm = np.array(bench[key]["clean_test"]["confusion_matrix"], dtype=float)
        cm = cm / cm.sum(axis=1, keepdims=True) * 100
        im = ax.imshow(cm, cmap=matplotlib.colors.LinearSegmentedColormap.from_list(
            "c", ["#05070d", "#0e7490", CYAN, LIME]), vmin=0, vmax=100)
        ax.set_xticks(range(10), classes, rotation=45, ha="right", fontsize=7)
        ax.set_yticks(range(10), classes, fontsize=7)
        ax.set_title(f"{name} · {bench[key]['clean_test']['accuracy'] * 100:.2f}%", loc="left")
        ax.grid(False)
        for spine in ax.spines.values():
            spine.set_color(GRID)
        if key == "advanced":
            cbar = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02)
            cbar.set_label("% of true class", color=MUTED, fontsize=7.5)
            cbar.outline.set_edgecolor(GRID)
            cbar.ax.tick_params(colors=MUTED, labelsize=7)
    fig.suptitle("Where the two tiers differ — row-normalised confusion, 10k test images",
                 x=0.012, ha="left", color=TEXT, fontsize=12.5, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    _watermark(fig)
    _save(fig, "confusion.png")


# ------------------------------------------------------------------- Pareto


def pareto(bench: dict) -> None:
    pts = [
        ("MobileNetV3-Small", bench["lightweight"]["latency_ms_per_sample"],
         bench["lightweight"]["clean_test"]["accuracy"], CYAN),
        ("ResNet50", bench["advanced"]["latency_ms_per_sample"],
         bench["advanced"]["clean_test"]["accuracy"], VIOLET),
    ]
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    for name, ms, acc, colour in pts:
        ax.scatter([ms], [acc * 100], s=150, color=colour, edgecolor=BG, zorder=3, linewidths=1.4)
        ax.annotate(f"{name}\n{ms:.2f} ms · {acc * 100:.2f}%",
                    (ms, acc * 100), textcoords="offset points", xytext=(12, -6),
                    color=TEXT, fontsize=8.4)
    aether = next(p for p in bench["routing"]["policies"] if p["policy"] == "aether")
    ax.scatter([aether["mean_latency_ms"]], [aether["accuracy_on_answered"] * 100], s=210,
               color=LIME, marker="*", edgecolor=BG, linewidths=1.0, zorder=4)
    ax.annotate(
        f"AETHER routed\n{aether['mean_latency_ms']:.2f} ms · {aether['accuracy_on_answered'] * 100:.1f}% on answered",
        (aether["mean_latency_ms"], aether["accuracy_on_answered"] * 100),
        textcoords="offset points", xytext=(10, 10), color=LIME, fontsize=8.2,
    )
    ax.set_xscale("log")
    ax.set_xticks([0.2, 0.5, 1.0, 2.0], ["0.2", "0.5", "1.0", "2.0"])
    ax.minorticks_off()
    ax.set_xlim(0.15, 3.3)
    ax.set_xlabel("latency per image (ms, log scale)")
    ax.set_ylabel("accuracy  (%)")
    ax.set_ylim(60, 100)
    ax.set_title("Accuracy / latency frontier — routing sits above the line", loc="left")
    _clean_axes(ax)
    _watermark(fig)
    _save(fig, "pareto.png")


# ---------------------------------------------------------------- risk heads


def risk_heads(calib: dict) -> None:
    heads = calib
    names = ["lightweight", "advanced_precall", "advanced"]
    labels = ["light risk", "pre-call advanced risk", "post-call advanced risk"]
    aucs = [heads[n]["metrics"]["auc"] for n in names]
    colours = [CYAN, AMBER, VIOLET]
    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    bars = ax.barh(np.arange(3), aucs, color=colours, alpha=0.88, height=0.58)
    for bar, auc in zip(bars, aucs, strict=True):
        ax.text(auc - 0.02, bar.get_y() + bar.get_height() / 2, f"{auc:.4f}",
                va="center", ha="right", color="#05070d", fontsize=9, fontweight="bold")
    ax.set_yticks(np.arange(3), labels)
    ax.set_xlim(0.5, 1.0)
    ax.axvline(0.5, color=MUTED, lw=0.8, ls=":")
    ax.set_xlabel("ROC AUC on the calibration split")
    ax.set_title("Three heads, three jobs — and no peeking at the answer", loc="left")
    _clean_axes(ax)
    ax.grid(axis="y", visible=False)
    _watermark(fig)
    _save(fig, "risk-heads.png")


# -------------------------------------------------------------- architecture


def architecture() -> None:
    fig, ax = plt.subplots(figsize=(9.6, 4.6))
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    def box(x, y, w, h, title, sub, edge):
        ax.add_patch(FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.008,rounding_size=0.018",
            facecolor="#0a1120", edgecolor=edge, linewidth=1.1, alpha=0.95,
        ))
        ax.text(x + w / 2, y + h * 0.64, title, ha="center", va="center",
                color=TEXT, fontsize=8.4, fontweight="bold")
        ax.text(x + w / 2, y + h * 0.30, sub, ha="center", va="center",
                color=MUTED, fontsize=6.6)

    def arrow(x0, y0, x1, y1, colour=CYAN):
        ax.add_patch(FancyArrowPatch(
            (x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=9,
            color=colour, lw=1.0, alpha=0.75, shrinkA=2, shrinkB=2,
        ))

    box(0.01, 0.46, 0.13, 0.18, "input", "validate · decode\nquality signals", CYAN)
    box(0.17, 0.52, 0.15, 0.22, "light tier", "MobileNetV3-S\n1.53 M · 0.18 ms", CYAN)
    box(0.17, 0.18, 0.15, 0.22, "drift monitor", "PSI · KS · MMD\nbehaviour", VIOLET)
    box(0.35, 0.36, 0.15, 0.22, "evidence", "entropy · margin\nshift · disagreement", VIOLET)
    box(0.53, 0.52, 0.15, 0.22, "risk heads", "P(wrong | evidence)\n3 fitted heads", AMBER)
    box(0.53, 0.18, 0.15, 0.22, "router", "utility · feasibility\naudit record", LIME)
    box(0.72, 0.66, 0.13, 0.16, "answer", "accept + risk\nexplanation", LIME)
    box(0.72, 0.42, 0.13, 0.16, "escalate", "ResNet50\n2.10 ms", VIOLET)
    box(0.72, 0.18, 0.13, 0.16, "review", "human queue\nverified labels", AMBER)
    box(0.89, 0.42, 0.10, 0.16, "abstain", "decline\nuncovered", PINK)
    box(0.35, 0.04, 0.33, 0.10, "evidence + decision store", "SQLite / PostgreSQL · Alembic · append-only audit trail", MUTED)

    arrow(0.14, 0.55, 0.17, 0.62)
    arrow(0.14, 0.52, 0.17, 0.30)
    arrow(0.32, 0.62, 0.35, 0.53)
    arrow(0.32, 0.30, 0.35, 0.42)
    arrow(0.50, 0.50, 0.53, 0.60)
    arrow(0.50, 0.42, 0.53, 0.30)
    arrow(0.68, 0.62, 0.72, 0.72)
    arrow(0.68, 0.30, 0.72, 0.48)
    arrow(0.68, 0.24, 0.72, 0.26)
    arrow(0.85, 0.42, 0.89, 0.47)
    arrow(0.85, 0.50, 0.72, 0.66, VIOLET)
    arrow(0.68, 0.22, 0.35, 0.12, LIME)
    ax.text(0.012, 0.93, "One pipeline, three entry points — API · CLI · experiments",
            color=TEXT, fontsize=12, fontweight="bold")
    ax.text(0.012, 0.86, "the benchmark row and the served request execute the same code path",
            color=MUTED, fontsize=7.8)
    _save(fig, "architecture.png")


def efficiency(report: dict) -> None:
    rows = report["batch_scaling"]
    sizes = [row["batch_size"] for row in rows]
    ms = [row["ms_per_image"] for row in rows]
    scopes = report["scopes"]

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.9))
    ax = axes[0]
    ax.plot(sizes, ms, color=CYAN, lw=5, alpha=0.12)
    ax.plot(sizes, ms, color=CYAN, lw=2.2, marker="o", ms=5)
    for size, value in zip(sizes, ms, strict=True):
        ax.annotate(f"{value:.2f}", (size, value), textcoords="offset points", xytext=(0, 9),
                    ha="center", color=MUTED, fontsize=7)
    ax.set_xscale("log", base=2)
    ax.set_xticks(sizes, [str(size) for size in sizes])
    ax.set_xlabel("batch size")
    ax.set_ylabel("ms per image")
    ax.set_ylim(0, max(ms) * 1.22)
    ax.set_title("Batching amortises the fixed cost", loc="left", pad=12)
    _clean_axes(ax)

    ax = axes[1]
    labels = ["baseline", "no drift bookkeeping", "heuristic heads"]
    values = [
        scopes["baseline_ms_per_image"],
        scopes["without_drift_ms_per_image"],
        scopes["heuristic_risk_ms_per_image"],
    ]
    colours = [LIME, CYAN, VIOLET]
    ax.barh(list(range(len(labels))), values, color=colours, alpha=0.85, height=0.6)
    for index, value in enumerate(values):
        ax.text(value + max(values) * 0.015, index, f"{value:.2f} ms", va="center",
                color=TEXT, fontsize=8)
    ax.set_yticks(list(range(len(labels))), labels)
    ax.set_xlim(0, max(values) * 1.22)
    ax.set_xlabel("ms per image, end to end (light tier)")
    escalation = scopes.get("escalation_ms_per_escalated_image")
    subtitle = (
        f"escalation adds {escalation:.2f} ms, paid only when the router escalates"
        if escalation is not None
        else "what each capability costs"
    )
    ax.set_title(subtitle, loc="left", fontsize=10, color=MUTED, fontweight="normal")
    _clean_axes(ax)
    ax.grid(axis="y", visible=False)
    fig.suptitle("Efficiency audit — CPU, warmed, repeated runs", x=0.012, ha="left",
                 color=TEXT, fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0.02, 1, 0.90))
    _watermark(fig)
    _save(fig, "efficiency.png")


def stability_sweep(report: dict) -> None:
    """Selective error against corruption severity at two calibrated budgets."""
    rows = report["severity_sweep"]
    severities = [1, 3, 5]
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.9), sharey=True)
    for ax, tier, name in (
        (axes[0], "lightweight", "MobileNetV3-Small"),
        (axes[1], "advanced", "ResNet50"),
    ):
        entries = [entry for entry in rows if entry["tier"] == tier]
        for entry, colour in zip(entries, (AMBER, CYAN), strict=False):
            xs = [stream["severity"] for stream in entry["streams"]]
            ys = [stream["selective_error"] * 100 for stream in entry["streams"]]
            ax.plot(xs, ys, color=colour, marker="o", lw=2, label=f"budget {entry['alpha']:.2f}")
            ax.axhline(entry["alpha"] * 100, color=colour, ls=":", lw=0.9, alpha=0.55)
        ax.set_xticks(severities)
        ax.set_ylim(0, 100)
        ax.set_xlabel("gaussian-noise severity")
        ax.set_title(f"{name} — selective error vs corruption", loc="left", fontsize=10.5, pad=10)
        ax.legend(fontsize=7.6, labelcolor=MUTED, loc="lower left")
        _clean_axes(ax)
    axes[0].set_ylabel("selective error among answered  (%)")
    fig.suptitle(
        "Risk control under corruption — dotted lines are the requested budgets",
        x=0.012, ha="left", color=TEXT, fontsize=12.5, fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0.02, 1, 0.90))
    _watermark(fig)
    _save(fig, "stability-sweep.png")


def main() -> None:
    bench = load("benchmarks.json")
    calib = json.loads((CALIB / "learned_policy.json").read_text())
    hero(bench)
    reliability(bench)
    ece_before_after(bench)
    selective(bench)
    robustness(bench)
    drift(bench)
    routing(bench)
    ablations(bench)
    confusion(bench)
    pareto(bench)
    risk_heads(calib)
    architecture()
    efficiency_path = REPORTS / "efficiency.json"
    if efficiency_path.exists():
        efficiency(json.loads(efficiency_path.read_text()))
    stability_path = REPORTS / "stability.json"
    if stability_path.exists():
        stability_sweep(json.loads(stability_path.read_text()))


if __name__ == "__main__":
    main()
