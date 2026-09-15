#!/usr/bin/env python3
"""Create diagnostic figures for the independently confirmed extended-range result."""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/lowflow_extended_mplconfig")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from lib.common import RESULTS_ROOT  # noqa: E402


COLORS = {
    "event_brier": "#3B528B", "timing_crps": "#21918C",
    "duration_crps": "#D55E00", "deficit_crps": "#CC4678",
}
LABELS = {
    "event_brier": "Onset probability", "timing_crps": "Onset timing",
    "duration_crps": "Low-flow duration", "deficit_crps": "Cumulative deficit",
    "hydrograph_marginal_shuffle": "Identical shuffled marginals",
    "seasonal_climatology_path": "Seasonal trajectory climatology",
    "constant_persistence_path": "Constant persistence",
}


def style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9.5,
        "axes.titlesize": 10.5, "axes.labelsize": 10,
        "axes.linewidth": 0.8, "xtick.direction": "out",
        "ytick.direction": "out", "svg.fonttype": "none",
    })


def save(fig, output: Path, name: str) -> None:
    output.mkdir(parents=True, exist_ok=True)
    fig.savefig(output / f"{name}.svg", bbox_inches="tight", facecolor="white")
    fig.savefig(output / f"{name}.png", dpi=500, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def paired_relative_skill(metrics: pd.DataFrame, reference: str,
                          score: str) -> pd.DataFrame:
    keys = [
        "GAGE_ID", "spatial_group", "threshold_name", "lead_days", "target"
    ]
    candidate = metrics[
        metrics["model"].eq("hydrograph_analog")
        & metrics["threshold_name"].eq("Q10")
        & metrics["target"].eq("onset")
    ][keys + ["n", score]].dropna(subset=[score])
    baseline = metrics[
        metrics["model"].eq(reference)
        & metrics["threshold_name"].eq("Q10")
        & metrics["target"].eq("onset")
    ][keys + [score]].dropna(subset=[score])
    paired = candidate.merge(
        baseline, on=keys, suffixes=("_candidate", "_reference")
    )
    rows = []
    for lead, group in paired.groupby("lead_days"):
        candidate_score = np.average(
            group[f"{score}_candidate"], weights=group["n"]
        )
        reference_score = np.average(
            group[f"{score}_reference"], weights=group["n"]
        )
        rows.append({
            "lead_days": int(lead),
            "relative_skill": 100 * (reference_score - candidate_score) / reference_score,
            "candidate_score": candidate_score, "reference_score": reference_score,
            "basins": int(len(group)),
        })
    return pd.DataFrame(rows)


def coherence_by_outcome(metrics: pd.DataFrame, output: Path) -> None:
    scores = ["event_brier", "timing_crps", "duration_crps", "deficit_crps"]
    fig, axes = plt.subplots(2, 2, figsize=(9.2, 6.3), sharex=True)
    for axis, score in zip(axes.flat, scores):
        values = paired_relative_skill(
            metrics, "hydrograph_marginal_shuffle", score
        )
        values = values[values["lead_days"].ge(3)]
        axis.plot(
            values["lead_days"], values["relative_skill"], marker="o",
            linewidth=2.1, markersize=4.5, color=COLORS[score],
        )
        axis.axhline(0, color="#424242", linewidth=0.8, linestyle=(0, (3, 2)))
        axis.set_title(LABELS[score])
        axis.set_xscale("log")
        axis.set_xticks([3, 7, 14, 30, 45, 60, 90])
        axis.get_xaxis().set_major_formatter(plt.ScalarFormatter())
        axis.grid(axis="y", color="#DDDDDD", linewidth=0.6)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0, 0].set_ylabel("Skill from coherence (%)")
    axes[1, 0].set_ylabel("Skill from coherence (%)")
    axes[1, 0].set_xlabel("Forecast horizon (days)")
    axes[1, 1].set_xlabel("Forecast horizon (days)")
    fig.suptitle(
        "Temporal coherence adds increasing value to Q10 hazard prediction",
        fontsize=12, fontweight="semibold", y=1.01,
    )
    fig.text(
        0.01, -0.01,
        "Skill is relative score reduction versus forecasts with identical daily marginals but shuffled member paths.",
        fontsize=8.2, color="#4A4A4A",
    )
    fig.tight_layout()
    save(fig, output, "extended_coherence_by_outcome")


def onset_brier_references(metrics: pd.DataFrame, output: Path) -> None:
    references = [
        "hydrograph_marginal_shuffle", "seasonal_climatology_path",
        "constant_persistence_path",
    ]
    colors = ["#3B528B", "#21918C", "#D55E00"]
    fig, axis = plt.subplots(figsize=(7.2, 4.25))
    for reference, color in zip(references, colors):
        values = paired_relative_skill(metrics, reference, "event_brier")
        values = values[values["lead_days"].ge(3)]
        axis.plot(
            values["lead_days"], values["relative_skill"], marker="o",
            markersize=4.5, linewidth=2, color=color, label=LABELS[reference],
        )
    axis.axhline(0, color="#424242", linewidth=0.8, linestyle=(0, (3, 2)))
    axis.set_xscale("log")
    axis.set_xticks([3, 7, 14, 30, 45, 60, 90])
    axis.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    axis.set_xlabel("Forecast horizon (days)")
    axis.set_ylabel("Q10 onset Brier skill (%)")
    axis.set_title("Coherent hydrograph analog versus operationally relevant references")
    axis.legend(frameon=False, loc="best")
    axis.grid(axis="y", color="#DDDDDD", linewidth=0.6)
    axis.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    save(fig, output, "extended_onset_skill_references")


def ecoregion_heatmap(regional: pd.DataFrame, output: Path) -> None:
    selected = regional[regional["score"].isin(COLORS)].copy()
    summary = selected.groupby(
        ["spatial_group", "lead_days"], as_index=False
    )["basin_fraction_improved"].mean()
    table = summary.pivot(
        index="spatial_group", columns="lead_days", values="basin_fraction_improved"
    ).sort_values(90, ascending=False)
    values = 100 * table.to_numpy(float)
    fig, axis = plt.subplots(figsize=(6.8, 4.8))
    image = axis.imshow(values, aspect="auto", cmap="YlGnBu", vmin=80, vmax=100)
    axis.set_xticks(np.arange(len(table.columns)), table.columns.astype(str))
    axis.set_yticks(np.arange(len(table.index)), table.index)
    axis.set_xlabel("Forecast horizon (days)")
    axis.set_title("Basins improved by coherent paths across ecoregions")
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            color = "white" if values[row, column] > 94 else "#222222"
            axis.text(column, row, f"{values[row, column]:.0f}",
                      ha="center", va="center", fontsize=8, color=color)
    bar = fig.colorbar(image, ax=axis, pad=0.02)
    bar.set_label("Mean fraction improved (%)")
    fig.tight_layout()
    save(fig, output, "extended_ecoregion_consistency")


def sensitivity_plot(sensitivity: pd.DataFrame, output: Path) -> None:
    order = [
        "base_age3_p0.1", "members_31", "members_51", "members_151",
        "dry_age1", "dry_age7", "precip_0.0", "precip_1.0",
        "seasonal_q10", "shuffle_repeat_1", "shuffle_repeat_2",
        "shuffle_repeat_3", "shuffle_repeat_4", "shuffle_repeat_5",
    ]
    labels = {
        "base_age3_p0.1": "Primary configuration", "members_31": "31 members",
        "members_51": "51 members", "members_151": "151 members",
        "dry_age1": "Dry age ≥1 day", "dry_age7": "Dry age ≥7 days",
        "precip_0.0": "Dry cutoff: 0.0 mm", "precip_1.0": "Dry cutoff: 1.0 mm",
        "seasonal_q10": "Seasonally varying Q10",
        "shuffle_repeat_1": "Shuffle seed 1", "shuffle_repeat_2": "Shuffle seed 2",
        "shuffle_repeat_3": "Shuffle seed 3", "shuffle_repeat_4": "Shuffle seed 4",
        "shuffle_repeat_5": "Shuffle seed 5",
    }
    work = sensitivity.set_index("variant").reindex(order).dropna().reset_index()
    y = np.arange(len(work))
    fig, axis = plt.subplots(figsize=(7.2, 5.4))
    axis.scatter(
        100 * work["mean_basin_fraction_improved"], y, s=43,
        c=work["significant_positive"], cmap="viridis", vmin=12, vmax=16,
        edgecolor="white", linewidth=0.5,
    )
    axis.set_yticks(y, [labels[value] for value in work["variant"]])
    axis.set_xlim(70, 101)
    axis.set_xlabel("Mean fraction of basins improved across 30–90-day outcomes (%)")
    axis.set_title("Temporal-coherence result is insensitive to analysis choices")
    axis.axvline(75, color="#555555", linewidth=0.8, linestyle=(0, (3, 2)))
    axis.grid(axis="x", color="#DDDDDD", linewidth=0.6)
    axis.spines[["top", "right", "left"]].set_visible(False)
    axis.invert_yaxis()
    fig.tight_layout()
    save(fig, output, "extended_sensitivity_summary")


def main() -> None:
    style()
    root = RESULTS_ROOT / "09_extended_forecast"
    output = root / "figures"
    metrics = pd.read_csv(
        root / "extended_confirmation_metrics.csv.gz", dtype={"GAGE_ID": str}
    )
    regional = pd.read_csv(root / "extended_ecoregion_consistency.csv")
    sensitivity = pd.read_csv(root / "sensitivities" / "sensitivity_summary.csv")
    coherence_by_outcome(metrics, output)
    onset_brier_references(metrics, output)
    ecoregion_heatmap(regional, output)
    sensitivity_plot(sensitivity, output)
    print(f"Saved extended-range figures to {output}")


if __name__ == "__main__":
    main()
