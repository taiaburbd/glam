"""
Build the lead figure: task formulation, not model capacity, drives reported
performance in visual field progression prediction.

Panel A  variance explained under the leaky and time-split formulations, in
         both cohorts and for both model classes
Panel B  internal test-set error for every model, against the naive mean
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
EXTERNAL = ROOT / "reports" / "forecast" / "external_validation.json"
METRICS = ROOT / "reports" / "forecast" / "forecast_metrics.json"
OUT = ROOT / "paper" / "JAMA Ophthalmology" / "FigLeakage.png"

LEAKY_COLOR = "#B03A2E"
SPLIT_COLOR = "#2E5E8E"

DISPLAY = {
    "glam": "GLAM (deep learning)",
    "gbm_full_features": "Gradient boosting, 13 features",
    "ridge_full_features": "Ridge, 13 features",
    "ridge_baseline_md": "Ridge, baseline MD",
    "naive_mean": "Naive cohort mean",
    "ols_early_3visits": "OLS of first 3 visits",
}


def main() -> None:
    ext = json.loads(EXTERNAL.read_text())
    met = json.loads(METRICS.read_text())

    fig, (ax_a, ax_b) = plt.subplots(
        1, 2, figsize=(13.5, 5.4), gridspec_kw={"width_ratios": [1.05, 1.0]}
    )

    # ---- Panel A: the collapse -------------------------------------------
    groups = [
        ("UWHVF\nridge", "uwhvf", "ridge"),
        ("UWHVF\ngrad. boost", "uwhvf", "gbm"),
        ("GRAPE\nridge", "grape", "ridge"),
        ("GRAPE\ngrad. boost", "grape", "gbm"),
    ]
    leaky = [ext[c]["leaky"][m]["r2"] for _, c, m in groups]
    split = [ext[c]["forecast"][m]["r2"] for _, c, m in groups]

    x = np.arange(len(groups))
    w = 0.38
    b1 = ax_a.bar(x - w / 2, leaky, w, color=LEAKY_COLOR,
                  label="Leaky: label from the input window")
    b2 = ax_a.bar(x + w / 2, split, w, color=SPLIT_COLOR,
                  label="Time-split: label from later visits only")

    for bars in (b1, b2):
        for rect in bars:
            h = rect.get_height()
            ax_a.annotate(
                f"{h:+.2f}",
                (rect.get_x() + rect.get_width() / 2, h),
                textcoords="offset points",
                xytext=(0, 4 if h >= 0 else -13),
                ha="center",
                fontsize=9.5,
            )

    ax_a.axhline(0, color="black", lw=1.0)
    ax_a.set_xticks(x)
    ax_a.set_xticklabels([g[0] for g in groups], fontsize=10)
    ax_a.set_ylabel("Variance explained in subsequent rate (R\u00b2)", fontsize=11)
    ax_a.set_ylim(-0.45, 1.12)
    ax_a.legend(frameon=False, fontsize=10, loc="upper left")
    ax_a.set_title(
        "A  Identical models and features; only the label window differs",
        fontsize=11.5, loc="left", fontweight="bold",
    )
    ax_a.spines[["top", "right"]].set_visible(False)

    n_note = (
        f"UWHVF n={ext['uwhvf']['leaky']['n']}/{ext['uwhvf']['forecast']['n']}   "
        f"GRAPE n={ext['grape']['leaky']['n']}/{ext['grape']['forecast']['n']}"
    )
    ax_a.text(0.01, -0.155, n_note, transform=ax_a.transAxes, fontsize=9, color="#555")

    # ---- Panel B: nobody beats the mean ----------------------------------
    rows = [("glam", met["glam"]["md_mae"])]
    rows += [(b["name"], b["md_mae"]) for b in met["baselines"]]
    rows.sort(key=lambda r: r[1], reverse=True)

    labels = [DISPLAY.get(n, n) for n, _ in rows]
    vals = [v for _, v in rows]
    naive = dict(rows)["naive_mean"]
    colors = ["#6B7B8C" if n != "glam" else "#1F4E79" for n, _ in rows]

    y = np.arange(len(rows))
    ax_b.barh(y, vals, color=colors, height=0.62)
    ax_b.axvline(naive, color=LEAKY_COLOR, ls="--", lw=1.4)
    ax_b.set_ylim(-0.6, len(rows) - 0.1)
    ax_b.text(
        naive + 0.012, len(rows) - 0.22, "predicting the cohort mean",
        color=LEAKY_COLOR, fontsize=9.5, ha="left", va="center",
    )
    for yi, v in zip(y, vals):
        ax_b.annotate(f"{v:.3f}", (v, yi), xytext=(4, 0),
                      textcoords="offset points", va="center", fontsize=9.5)

    ax_b.set_yticks(y)
    ax_b.set_yticklabels(labels, fontsize=10)
    ax_b.set_xlabel("Mean absolute error in subsequent MD rate (dB/y)", fontsize=11)
    ax_b.set_xlim(0, max(vals) * 1.18)
    ax_b.set_title(
        "B  Internal test set: no model improves on the mean",
        fontsize=11.5, loc="left", fontweight="bold",
    )
    ax_b.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    fig.savefig(OUT, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
