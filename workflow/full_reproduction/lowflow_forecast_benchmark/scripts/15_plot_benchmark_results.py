#!/usr/bin/env python3
"""Plot the full-confirmation operational and trajectory results (not manuscript figures)."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from lib.common import RESULTS_ROOT  # noqa: E402


COLORS = {"survival_mixture": "#007C91", "operational_direct_logistic": "#D55E00"}
LABELS = {
    "survival_mixture": "Dry-spell survival mixture",
    "operational_direct_logistic": "Direct hydrograph-memory model",
    "operational_qbin": "Flow-state (q-bin)",
    "operational_persistence": "Persistence",
    "operational_training_climatology": "Training climatology",
}


def style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 10,
        "axes.labelsize": 9.5, "axes.linewidth": 0.8, "xtick.direction": "out",
        "ytick.direction": "out", "svg.fonttype": "none",
    })


def save(fig, stem: Path) -> None:
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".png"), dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def operational_plot(frame: pd.DataFrame, output: Path) -> None:
    data = frame[frame["threshold_quantile"].eq(0.1)].copy()
    references = ["operational_qbin", "operational_persistence",
                  "operational_training_climatology"]
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.35), sharex=True)
    for axis, reference in zip(axes, references):
        subset = data[data["reference_model"].eq(reference)]
        for model in ["survival_mixture", "operational_direct_logistic"]:
            values = subset[subset["model"].eq(model)].sort_values("lead_days")
            x = values["lead_days"].to_numpy(float)
            y = values["weighted_brier_improvement"].to_numpy(float)
            error = np.vstack([
                y - values["bootstrap_ci_low"].to_numpy(float),
                values["bootstrap_ci_high"].to_numpy(float) - y,
            ])
            axis.errorbar(
                x, y, yerr=error, color=COLORS[model], marker="o", markersize=4.2,
                linewidth=1.8, elinewidth=0.9, capsize=2.2, label=LABELS[model],
            )
        axis.axhline(0, color="#383838", linewidth=0.8, linestyle=(0, (3, 2)))
        axis.set_xscale("log")
        axis.set_xticks([1, 3, 7, 14, 30, 60, 90])
        axis.get_xaxis().set_major_formatter(plt.ScalarFormatter())
        axis.set_title(f"Versus {LABELS[reference].lower()}")
        axis.grid(axis="y", color="#D9D9D9", linewidth=0.6)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Brier-score improvement")
    axes[1].set_xlabel("Forecast lead (days)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.03),
               ncol=2, frameon=False)
    fig.suptitle("Q10 endpoint forecasts initialized during observed dry periods", y=1.13,
                 fontsize=11, fontweight="semibold")
    fig.text(0.01, -0.02, "Positive values favor the candidate; bars are 95% spatially hierarchical bootstrap intervals.",
             fontsize=8, color="#4A4A4A")
    fig.tight_layout()
    save(fig, output / "full_confirmation_operational_skill")


def trajectory_plot(frame: pd.DataFrame, output: Path) -> None:
    data = frame[frame["threshold_quantile"].eq(0.1)].copy()
    scores = ["first_passage_brier", "first_passage_discrete_crps", "deficit_volume_crps"]
    titles = ["First-passage probability", "First-passage timing", "Cumulative deficit"]
    colors = ["#3B528B", "#21918C", "#D1495B"]
    fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.35), sharex=True)
    for axis, score, title, color in zip(axes, scores, titles, colors):
        values = data[data["score"].eq(score)].sort_values("lead_days")
        x = values["lead_days"].to_numpy(float)
        y = values["weighted_improvement"].to_numpy(float)
        error = np.vstack([
            y - values["bootstrap_ci_low"].to_numpy(float),
            values["bootstrap_ci_high"].to_numpy(float) - y,
        ])
        axis.errorbar(x, y, yerr=error, color=color, marker="o", markersize=4.5,
                      linewidth=1.9, elinewidth=0.9, capsize=2.2)
        axis.axhline(0, color="#383838", linewidth=0.8, linestyle=(0, (3, 2)))
        axis.set_title(title)
        axis.set_xscale("log")
        axis.set_xticks([3, 7, 14, 30])
        axis.get_xaxis().set_major_formatter(plt.ScalarFormatter())
        axis.grid(axis="y", color="#D9D9D9", linewidth=0.6)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Score improvement")
    axes[1].set_xlabel("Forecast horizon (days)")
    fig.suptitle("Value of coherent Q10 paths over identical shuffled marginals", y=1.03,
                 fontsize=11, fontweight="semibold")
    fig.text(0.01, -0.02, "Positive values favor coherent paths; bars are 95% spatially hierarchical bootstrap intervals.",
             fontsize=8, color="#4A4A4A")
    fig.tight_layout()
    save(fig, output / "full_confirmation_trajectory_skill")


def main() -> None:
    style()
    root = RESULTS_ROOT / "08_full_confirmation"
    operational = pd.read_csv(root / "operational_full_comparison.csv")
    trajectory = pd.read_csv(root / "trajectory_full_comparison.csv")
    output = root / "figures"
    output.mkdir(parents=True, exist_ok=True)
    operational_plot(operational, output)
    trajectory_plot(trajectory, output)
    print(f"Saved full-confirmation figures to {output}")


if __name__ == "__main__":
    main()

