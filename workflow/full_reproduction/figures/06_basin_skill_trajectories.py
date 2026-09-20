#!/usr/bin/env python3
"""Classical journal treatment of the basin-level skill trajectories."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

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


BASE_SCRIPT = HERE / "06_skill_trajectories_base.py"
spec = importlib.util.spec_from_file_location("skill_trajectory_base", BASE_SCRIPT)
base = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(base)

STEM = "figure_06_basin_skill_trajectories"
LEADS = base.LEADS
SCORES = tuple((score, label) for score, label, _ in base.SCORES)
LOWER_DISPLAY = base.LOWER_DISPLAY
UPPER_DISPLAY = base.UPPER_DISPLAY

NAVY = "#17365D"
BLACK = "#202020"
DARK_GRAY = "#555555"
MID_GRAY = "#7A7A7A"
TRACE_GRAY = "#68737A"
RIBBON_GRAY = "#C9D0D5"
GRID = "#DADADA"
EXTENSION = "#F1F1F1"
TEXT = "#222222"


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": [
                "Times New Roman",
                "Times",
                "Liberation Serif",
                "DejaVu Serif",
            ],
            "font.size": 9.5,
            "font.weight": "bold",
            "axes.labelsize": 10.2,
            "axes.labelweight": "bold",
            "axes.titlesize": 10.5,
            "axes.titleweight": "bold",
            "xtick.labelsize": 8.6,
            "ytick.labelsize": 8.6,
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "xtick.major.size": 3.2,
            "ytick.major.size": 3.2,
            "text.color": TEXT,
            "axes.labelcolor": TEXT,
            "axes.edgecolor": BLACK,
            "xtick.color": BLACK,
            "ytick.color": BLACK,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )


def load_sources() -> tuple[pd.DataFrame, pd.DataFrame]:
    value_path = SOURCE / "figure_06_basin_skill_trajectories_basin_values.csv.gz"
    summary_path = SOURCE / "figure_06_basin_skill_trajectories_summary.csv"
    if not value_path.exists() or not summary_path.exists():
        paired = base.load_pairs()
        values, summary = base.make_sources(paired)
        values.to_csv(value_path, index=False)
        summary.to_csv(summary_path, index=False)
    return pd.read_csv(value_path), pd.read_csv(summary_path)


def panel_label(axis: plt.Axes, letter: str) -> None:
    axis.text(
        -0.105,
        1.035,
        letter,
        transform=axis.transAxes,
        ha="left",
        va="bottom",
        fontsize=11.2,
        fontweight="bold",
        color=TEXT,
        clip_on=False,
    )


def draw(
    values: pd.DataFrame,
    summary: pd.DataFrame,
    output_stem: str = STEM,
) -> None:
    configure_style()
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(7.45, 5.55),
        sharex=True,
        sharey=True,
        gridspec_kw={"hspace": 0.25, "wspace": 0.12},
    )
    rng = np.random.default_rng(20260913)

    for index, (axis, (score, label)) in enumerate(zip(axes.flat, SCORES)):
        q10 = values.loc[
            values["score"].eq(score)
            & values["threshold_quantile"].eq(0.10)
        ]
        wide = q10.pivot(
            index="GAGE_ID", columns="lead_days", values="relative_skill_percent"
        ).reindex(columns=LEADS).dropna()
        trajectories = wide.to_numpy(float)
        clipped = np.clip(trajectories, LOWER_DISPLAY, UPPER_DISPLAY)

        # Raw paired basin histories remain visible but do not compete with summaries.
        axis.plot(
            LEADS,
            clipped.T,
            color=TRACE_GRAY,
            lw=0.22,
            alpha=0.016,
            solid_capstyle="round",
            zorder=1,
        )
        for lead_index, lead in enumerate(LEADS):
            jitter = rng.uniform(-1.18, 1.18, size=len(clipped))
            axis.scatter(
                np.full(len(clipped), lead) + jitter,
                clipped[:, lead_index],
                s=0.95,
                color=TRACE_GRAY,
                alpha=0.042,
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
            color=RIBBON_GRAY,
            alpha=0.58,
            linewidth=0,
            zorder=3,
        )
        axis.plot(
            x,
            core["median"].to_numpy(float),
            color=NAVY,
            lw=2.05,
            marker="o",
            ms=3.25,
            mfc="white",
            mec=NAVY,
            mew=0.9,
            zorder=6,
        )

        q5 = summary.loc[
            summary["score"].eq(score)
            & summary["threshold_quantile"].eq(0.05)
        ].sort_values("lead_days")
        q20 = summary.loc[
            summary["score"].eq(score)
            & summary["threshold_quantile"].eq(0.20)
        ].sort_values("lead_days")
        axis.plot(
            q5["lead_days"],
            q5["median"],
            color=MID_GRAY,
            lw=1.2,
            ls=(0, (1.3, 1.8)),
            zorder=5,
        )
        axis.plot(
            q20["lead_days"],
            q20["median"],
            color=BLACK,
            lw=1.2,
            ls=(0, (5.2, 2.6)),
            zorder=5,
        )

        q5_n = int(q5["basins"].iloc[0])
        q10_n = int(core["basins"].iloc[0])
        q20_n = int(q20["basins"].iloc[0])
        end = core.loc[core["lead_days"].eq(120)].iloc[0]
        axis.text(
            0.018,
            0.035,
            f"complete basins: Q5 {q5_n:,}  ·  Q10 {q10_n:,}  ·  Q20 {q20_n:,}",
            transform=axis.transAxes,
            ha="left",
            va="bottom",
            fontsize=7.55,
            fontweight="bold",
            color=DARK_GRAY,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.86, "pad": 1.0},
            zorder=8,
        )
        axis.annotate(
            f"{end['median']:+.1f}%\n{100 * end['fraction_improved']:.0f}% improved",
            xy=(120, end["median"]),
            xytext=(-7, 17),
            textcoords="offset points",
            ha="right",
            va="bottom",
            fontsize=7.7,
            fontweight="bold",
            color=NAVY,
            arrowprops={"arrowstyle": "-", "color": NAVY, "lw": 0.65},
            zorder=9,
        )

        axis.axhline(0, color=BLACK, lw=0.9, zorder=4)
        axis.axvspan(97.5, 122.5, color=EXTENSION, alpha=0.72, lw=0, zorder=0)
        axis.axvline(97.5, color=DARK_GRAY, lw=0.72, ls=(0, (1.5, 2.0)), zorder=4)
        axis.set_title(label, loc="left", pad=4.0, fontweight="bold")
        panel_label(axis, chr(97 + index))
        axis.set_xlim(27, 123)
        axis.set_ylim(LOWER_DISPLAY, UPPER_DISPLAY)
        axis.set_xticks(LEADS)
        axis.set_yticks([-100, -75, -50, -25, 0, 25, 50, 75])
        axis.grid(axis="y", color=GRID, lw=0.55, alpha=0.9, zorder=0)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.spines["left"].set_linewidth(0.8)
        axis.spines["bottom"].set_linewidth(0.8)
        for tick_label in [*axis.get_xticklabels(), *axis.get_yticklabels()]:
            tick_label.set_fontweight("bold")

    fig.text(
        0.018, 0.545, "Score reduction vs seasonal climatology (%)",
        ha="center", va="center", rotation=90,
        fontsize=10.2, fontweight="bold", color=TEXT,
    )
    for axis in axes[-1, :]:
        axis.set_xlabel("Forecast window (days)")

    legend = [
        Line2D(
            [], [], color=NAVY, lw=2.0, marker="o", ms=3.2,
            mfc="white", mec=NAVY, label="Q10 median and interquartile range",
        ),
        Line2D([], [], color=MID_GRAY, lw=1.2, ls=(0, (1.3, 1.8)), label="Q5 median"),
        Line2D([], [], color=BLACK, lw=1.2, ls=(0, (5.2, 2.6)), label="Q20 median"),
        Line2D([], [], color=BLACK, lw=0.9, label="No improvement"),
    ]
    figure_legend = fig.legend(
        handles=legend,
        frameon=False,
        ncol=4,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.002),
        columnspacing=1.30,
        handlelength=2.8,
        fontsize=7.45,
    )
    for legend_text in figure_legend.get_texts():
        legend_text.set_fontweight("bold")
    fig.subplots_adjust(left=0.10, right=0.995, top=0.975, bottom=0.115)

    for suffix, kwargs in (
        ("png", {"dpi": 600}),
        ("pdf", {}),
        ("svg", {}),
    ):
        fig.savefig(
            OUTPUT / f"{output_stem}.{suffix}",
            bbox_inches="tight",
            facecolor="white",
            **kwargs,
        )
    plt.close(fig)


def main() -> None:
    values, summary = load_sources()
    draw(values, summary)
    print(f"Created {OUTPUT / f'{STEM}.png'}")


if __name__ == "__main__":
    main()
