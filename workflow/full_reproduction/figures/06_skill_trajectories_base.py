#!/usr/bin/env python3
"""Basin-level forecast-skill trajectories for the six reported windows.

The figure follows the frozen analysis protocol: results at 30, 45, 60, and
90 days come from the original confirmation archive, whereas 105- and 120-day
results come from the separately archived extension. Q10 basin-level values
are shown directly; Q5 and Q20 appear as threshold-sensitivity medians. Only
basins with paired scores at all six windows contribute to a threshold curve,
so every trajectory follows one fixed basin population.
"""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "output"
SOURCE = HERE / "source_tables"
CACHE = HERE / ".cache"
for folder in (OUTPUT, SOURCE, CACHE):
    folder.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(CACHE / "matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


ORIGINAL_INPUT = (
    ROOT
    / "lowflow_forecast_benchmark"
    / "results"
    / "09_extended_forecast"
    / "extended_confirmation_metrics.csv.gz"
)
EXTENSION_INPUT = (
    ROOT
    / "lowflow_forecast_benchmark"
    / "results"
    / "10_extension_120"
    / "extension_120_metrics.csv.gz"
)
STEM = "figure_06_basin_skill_trajectories_base"
LEADS = np.array([30, 45, 60, 90, 105, 120], dtype=int)
THRESHOLDS = (0.05, 0.10, 0.20)
SCORES = (
    ("event_brier", "Occurrence", "#245A7A"),
    ("timing_crps", "Joint onset or no event", "#23766B"),
    ("duration_crps", "Low-flow-day count", "#B87920"),
    ("deficit_crps", "Cumulative deficit", "#A74445"),
)

TEXT = "#1D2830"
BLACK = "#1F1F1F"
MID = "#626D74"
GRID = "#D8DDE0"
LIGHT = "#F0F2F3"
LOWER_DISPLAY = -100.0
UPPER_DISPLAY = 85.0


def style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": [
                "Times New Roman",
                "Times",
                "Liberation Serif",
                "DejaVu Serif",
            ],
            "font.size": 8.5,
            "axes.labelsize": 9.2,
            "axes.titlesize": 9.5,
            "xtick.labelsize": 7.6,
            "ytick.labelsize": 7.6,
            "axes.linewidth": 0.75,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "xtick.major.size": 3.1,
            "ytick.major.size": 3.1,
            "text.color": TEXT,
            "axes.labelcolor": TEXT,
            "axes.edgecolor": BLACK,
            "xtick.color": BLACK,
            "ytick.color": BLACK,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )


def load_pairs() -> pd.DataFrame:
    columns = [
        "GAGE_ID",
        "spatial_group",
        "threshold_quantile",
        "lead_days",
        "target",
        "model",
        *[score for score, _, _ in SCORES],
    ]
    original = pd.read_csv(ORIGINAL_INPUT, usecols=columns)
    extension = pd.read_csv(EXTENSION_INPUT, usecols=columns)
    original = original[original["lead_days"].isin(LEADS[:4])]
    extension = extension[extension["lead_days"].isin(LEADS[4:])]
    data = pd.concat([original, extension], ignore_index=True)
    data = data.loc[
        data["target"].eq("onset")
        & data["lead_days"].isin(LEADS)
        & data["threshold_quantile"].isin(THRESHOLDS)
        & data["model"].isin(
            ["hydrograph_analog", "seasonal_climatology_path"]
        )
    ].copy()
    data["GAGE_ID"] = data["GAGE_ID"].astype(str).str.zfill(8)
    keys = ["GAGE_ID", "spatial_group", "threshold_quantile", "lead_days"]
    analog = data.loc[data["model"].eq("hydrograph_analog")].set_index(keys)
    seasonal = data.loc[
        data["model"].eq("seasonal_climatology_path")
    ].set_index(keys)
    paired = analog[[s[0] for s in SCORES]].join(
        seasonal[[s[0] for s in SCORES]],
        lsuffix="_analog",
        rsuffix="_seasonal",
        how="inner",
    ).reset_index()
    for score, _, _ in SCORES:
        denominator = paired[f"{score}_seasonal"].replace(0, np.nan)
        paired[f"{score}_skill"] = 100.0 * (
            paired[f"{score}_seasonal"] - paired[f"{score}_analog"]
        ) / denominator
    return paired.replace([np.inf, -np.inf], np.nan)


def complete_threshold_data(
    paired: pd.DataFrame, score: str, threshold: float
) -> pd.DataFrame:
    value = f"{score}_skill"
    part = paired.loc[
        paired["threshold_quantile"].eq(threshold),
        ["GAGE_ID", "spatial_group", "threshold_quantile", "lead_days", value],
    ].dropna(subset=[value])
    counts = part.groupby("GAGE_ID", observed=True)["lead_days"].nunique()
    complete = counts.index[counts.eq(len(LEADS))]
    return part.loc[part["GAGE_ID"].isin(complete)].copy()


def make_sources(paired: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    all_values = []
    summaries = []
    for score, label, _ in SCORES:
        value = f"{score}_skill"
        for threshold in THRESHOLDS:
            part = complete_threshold_data(paired, score, threshold)
            part = part.rename(columns={value: "relative_skill_percent"})
            part["score"] = score
            part["outcome"] = label
            all_values.append(part)
            for lead, values in part.groupby("lead_days", observed=True):
                x = values["relative_skill_percent"].to_numpy(float)
                summaries.append(
                    {
                        "score": score,
                        "outcome": label,
                        "threshold_quantile": threshold,
                        "lead_days": int(lead),
                        "basins": int(len(x)),
                        "fraction_improved": float(np.mean(x > 0)),
                        "q10": float(np.quantile(x, 0.10)),
                        "q25": float(np.quantile(x, 0.25)),
                        "median": float(np.median(x)),
                        "q75": float(np.quantile(x, 0.75)),
                        "q90": float(np.quantile(x, 0.90)),
                    }
                )
    values = pd.concat(all_values, ignore_index=True)
    summary = pd.DataFrame(summaries).sort_values(
        ["score", "threshold_quantile", "lead_days"]
    )
    return values, summary


def panel_label(axis: plt.Axes, letter: str) -> None:
    axis.text(
        -0.105,
        1.035,
        letter,
        transform=axis.transAxes,
        ha="left",
        va="bottom",
        fontsize=10.2,
        fontweight="bold",
        color=TEXT,
        clip_on=False,
    )


def draw(values: pd.DataFrame, summary: pd.DataFrame) -> None:
    style()
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(7.45, 5.55),
        sharex=True,
        sharey=True,
        gridspec_kw={"hspace": 0.29, "wspace": 0.15},
    )
    rng = np.random.default_rng(20260913)

    for index, (axis, (score, label, color)) in enumerate(zip(axes.flat, SCORES)):
        q10 = values.loc[
            values["score"].eq(score)
            & values["threshold_quantile"].eq(0.10)
        ]
        wide = q10.pivot(index="GAGE_ID", columns="lead_days", values="relative_skill_percent")
        wide = wide.reindex(columns=LEADS).dropna()
        trajectories = wide.to_numpy(float)
        clipped = np.clip(trajectories, LOWER_DISPLAY, UPPER_DISPLAY)

        # Thousands of faint paired basin trajectories provide the raw-data layer.
        axis.plot(
            LEADS,
            clipped.T,
            color=color,
            lw=0.24,
            alpha=0.018,
            solid_capstyle="round",
            zorder=1,
        )
        # A lightly jittered point cloud keeps individual basin results visible.
        for lead_index, lead in enumerate(LEADS):
            jitter = rng.uniform(-1.25, 1.25, size=len(clipped))
            axis.scatter(
                np.full(len(clipped), lead) + jitter,
                clipped[:, lead_index],
                s=1.0,
                color=color,
                alpha=0.045,
                linewidths=0,
                rasterized=True,
                zorder=2,
            )

        core = summary.loc[
            summary["score"].eq(score)
            & summary["threshold_quantile"].eq(0.10)
        ].sort_values("lead_days")
        x = core["lead_days"].to_numpy(float)
        axis.fill_between(
            x,
            core["q25"].to_numpy(float),
            core["q75"].to_numpy(float),
            color=color,
            alpha=0.18,
            linewidth=0,
            zorder=3,
        )
        axis.plot(
            x,
            core["median"].to_numpy(float),
            color=color,
            lw=2.05,
            marker="o",
            ms=3.2,
            mfc="white",
            mec=color,
            mew=0.85,
            zorder=5,
        )

        threshold_styles = {
            0.05: ("#5F6670", (0, (1.2, 1.8)), "Q5 median"),
            0.20: (BLACK, (0, (5, 2.6)), "Q20 median"),
        }
        for threshold, (line_color, dash, _) in threshold_styles.items():
            other = summary.loc[
                summary["score"].eq(score)
                & summary["threshold_quantile"].eq(threshold)
            ].sort_values("lead_days")
            axis.plot(
                other["lead_days"],
                other["median"],
                color=line_color,
                lw=1.12,
                ls=dash,
                zorder=4,
            )

        q5_n = int(
            summary.loc[
                summary["score"].eq(score)
                & summary["threshold_quantile"].eq(0.05),
                "basins",
            ].iloc[0]
        )
        q10_n = int(core["basins"].iloc[0])
        q20_n = int(
            summary.loc[
                summary["score"].eq(score)
                & summary["threshold_quantile"].eq(0.20),
                "basins",
            ].iloc[0]
        )
        end = core.loc[core["lead_days"].eq(120)].iloc[0]
        axis.text(
            0.018,
            0.035,
            f"complete basins: Q5 {q5_n:,}  ·  Q10 {q10_n:,}  ·  Q20 {q20_n:,}",
            transform=axis.transAxes,
            ha="left",
            va="bottom",
            fontsize=6.55,
            color=MID,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.80, "pad": 1.0},
            zorder=8,
        )
        axis.annotate(
            f"{end['median']:+.1f}%\n{100 * end['fraction_improved']:.0f}% improved",
            xy=(120, end["median"]),
            xytext=(-7, 17),
            textcoords="offset points",
            ha="right",
            va="bottom",
            fontsize=6.7,
            color=color,
            arrowprops={"arrowstyle": "-", "color": color, "lw": 0.65},
            zorder=9,
        )

        axis.axhline(0, color=BLACK, lw=0.85, zorder=4)
        axis.axvspan(97.5, 122.5, color=LIGHT, alpha=0.50, lw=0, zorder=0)
        axis.axvline(97.5, color=MID, lw=0.7, ls=(0, (1.5, 2.0)), zorder=4)
        axis.text(
            0.985,
            0.965,
            "separately tested\nextension",
            transform=axis.transAxes,
            ha="right",
            va="top",
            fontsize=6.4,
            color=MID,
            linespacing=1.2,
        )
        axis.set_title(label, loc="left", pad=4.0, fontweight="normal")
        panel_label(axis, chr(97 + index))
        axis.set_xlim(27, 123)
        axis.set_ylim(LOWER_DISPLAY, UPPER_DISPLAY)
        axis.set_xticks(LEADS)
        axis.set_yticks([-100, -75, -50, -25, 0, 25, 50, 75])
        axis.grid(axis="y", color=GRID, lw=0.55, alpha=0.82, zorder=0)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.spines["left"].set_linewidth(0.75)
        axis.spines["bottom"].set_linewidth(0.75)

    for axis in axes[:, 0]:
        axis.set_ylabel("Score reduction vs seasonal climatology (%)")
    for axis in axes[-1, :]:
        axis.set_xlabel("Forecast window (days)")

    legend = [
        Line2D(
            [], [], color=SCORES[0][2], lw=2.0, marker="o", ms=3.2,
            mfc="white", mec=SCORES[0][2], label="Q10 median and interquartile range",
        ),
        Line2D([], [], color="#5F6670", lw=1.15, ls=(0, (1.2, 1.8)), label="Q5 median"),
        Line2D([], [], color=BLACK, lw=1.15, ls=(0, (5, 2.6)), label="Q20 median"),
        Line2D([], [], color=BLACK, lw=0.85, label="No improvement"),
    ]
    fig.legend(
        handles=legend,
        frameon=False,
        ncol=4,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.002),
        columnspacing=1.30,
        handlelength=2.8,
        fontsize=7.45,
    )
    fig.subplots_adjust(left=0.10, right=0.995, top=0.975, bottom=0.115)

    for suffix, kwargs in (
        ("png", {"dpi": 600}),
        ("pdf", {}),
        ("svg", {}),
    ):
        fig.savefig(
            OUTPUT / f"{STEM}.{suffix}",
            bbox_inches="tight",
            facecolor="white",
            **kwargs,
        )
    plt.close(fig)


def main() -> None:
    for path in (ORIGINAL_INPUT, EXTENSION_INPUT):
        if not path.exists():
            raise FileNotFoundError(path)
    paired = load_pairs()
    values, summary = make_sources(paired)
    values.to_csv(SOURCE / f"{STEM}_basin_values.csv.gz", index=False)
    summary.to_csv(SOURCE / f"{STEM}_summary.csv", index=False)
    draw(values, summary)
    print(f"Created {OUTPUT / f'{STEM}.png'}")


if __name__ == "__main__":
    main()
