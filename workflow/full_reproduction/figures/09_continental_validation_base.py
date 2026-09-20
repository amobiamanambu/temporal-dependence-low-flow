#!/usr/bin/env python3
"""Core plotting functions for continental temporal validation.

The plot is built from the archived basin-first monthly validation table.

Run from the project root:

    python figures/09_continental_validation_base.py
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path


FIGURE_ROOT = Path(__file__).resolve().parent
SOURCE = (
    FIGURE_ROOT / "source_tables" /
    "figure_07_continental_temporal_validation.csv"
)
OUTPUT = FIGURE_ROOT / "output"
CACHE = FIGURE_ROOT / ".cache"
OUTPUT.mkdir(parents=True, exist_ok=True)
CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(CACHE / "matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd


LEADS = (30, 60, 90, 120)
TEXT = "#202624"
SECONDARY = "#5F6864"
FORECAST = "#176B9B"
OBSERVED = "#252A28"
UNDER = "#C44E3B"
OVER = "#4D86AA"
GRID = "#D8DEDB"
YEAR_SHADE = "#F4F5F3"
AXIS = "#747C78"


def apply_style() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
        "font.size": 8.0,
        "axes.labelsize": 8.3,
        "axes.titlesize": 9.2,
        "xtick.labelsize": 9.1,
        "ytick.labelsize": 9.1,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.dpi": 600,
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
    })


def load_data() -> pd.DataFrame:
    if not SOURCE.exists():
        raise FileNotFoundError(SOURCE)
    data = pd.read_csv(SOURCE, parse_dates=["month"])
    data = data[data["lead_days"].isin(LEADS)].copy()
    return data.sort_values(["lead_days", "month"]).reset_index(drop=True)


def metrics(part: pd.DataFrame) -> tuple[float, float, float]:
    forecast = part["forecast_probability"].to_numpy(float)
    observed = part["observed_frequency"].to_numpy(float)
    correlation = float(np.corrcoef(forecast, observed)[0, 1])
    bias_pp = 100.0 * float(np.mean(forecast - observed))
    mae_pp = 100.0 * float(np.mean(np.abs(forecast - observed)))
    return correlation, bias_pp, mae_pp


def signed(value: float) -> str:
    return f"{value:+.1f}".replace("-", "−")


def shade_years(axis: plt.Axes) -> None:
    for year in range(2016, 2026, 2):
        axis.axvspan(
            pd.Timestamp(f"{year}-01-01"),
            pd.Timestamp(f"{year + 1}-01-01"),
            color=YEAR_SHADE, linewidth=0, zorder=0,
        )


def finish_axis(axis: plt.Axes, *, show_bottom: bool) -> None:
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_color(AXIS)
    axis.spines["left"].set_linewidth(0.55)
    axis.spines["bottom"].set_color(AXIS)
    axis.spines["bottom"].set_linewidth(0.55)
    axis.tick_params(
        colors=SECONDARY, length=2.7, width=0.50,
        bottom=show_bottom, labelbottom=show_bottom,
    )
    for label in [*axis.get_xticklabels(), *axis.get_yticklabels()]:
        label.set_fontweight("bold")


def build(data: pd.DataFrame):
    figure = plt.figure(figsize=(7.45, 6.65))
    outer = figure.add_gridspec(
        2, 2, left=0.083, right=0.992, bottom=0.145, top=0.975,
        wspace=0.17, hspace=0.20,
    )
    signal_axes: list[plt.Axes] = []
    error_axes: list[plt.Axes] = []

    for index, lead in enumerate(LEADS):
        row, column = divmod(index, 2)
        inner = outer[row, column].subgridspec(
            2, 1, height_ratios=[3.2, 1.25], hspace=0.045,
        )
        signal = figure.add_subplot(inner[0])
        error = figure.add_subplot(inner[1], sharex=signal)
        signal_axes.append(signal)
        error_axes.append(error)

        part = data[data["lead_days"].eq(lead)]
        x = part["month"]
        forecast = part["forecast_probability"].to_numpy(float)
        observed = part["observed_frequency"].to_numpy(float)
        difference_pp = 100.0 * (forecast - observed)
        correlation, bias_pp, mae_pp = metrics(part)

        shade_years(signal)
        signal.grid(axis="y", color=GRID, linewidth=0.45, alpha=0.78)
        signal.set_axisbelow(True)
        signal.plot(
            x, observed, color=OBSERVED, lw=1.05,
            ls=(0, (3.2, 2.0)), zorder=3,
        )
        signal.plot(x, forecast, color=FORECAST, lw=1.55, zorder=4)
        signal.set_ylim(0, 0.74)
        signal.set_yticks([0, 0.2, 0.4, 0.6])
        signal.text(
            -0.06, 1.07, "abcd"[index], transform=signal.transAxes,
            ha="left", va="bottom", fontsize=10.8, fontweight="bold",
            color=TEXT, clip_on=False,
        )
        signal.text(
            0.0, 1.07, f"{lead} days", transform=signal.transAxes,
            ha="left", va="bottom", fontsize=9.4, fontweight="bold",
            color=TEXT, clip_on=False,
        )
        signal.text(
            0.018, 0.965,
            f"r = {correlation:.2f}   ·   MAE = {mae_pp:.1f} pp",
            transform=signal.transAxes, ha="left", va="top",
            fontsize=7.1, color="#000000", clip_on=True,
        )
        finish_axis(signal, show_bottom=False)

        shade_years(error)
        error.axhline(0, color=TEXT, lw=0.70, zorder=3)
        error.fill_between(
            x, 0, difference_pp, where=difference_pp < 0,
            color=UNDER, alpha=0.70, interpolate=True,
            linewidth=0, zorder=2,
        )
        error.fill_between(
            x, 0, difference_pp, where=difference_pp >= 0,
            color=OVER, alpha=0.58, interpolate=True,
            linewidth=0, zorder=2,
        )
        error.plot(x, difference_pp, color=SECONDARY, lw=0.45, zorder=4)
        error.axhline(
            bias_pp, color=UNDER if bias_pp < 0 else OVER,
            lw=0.72, ls=(0, (2.2, 2.0)), zorder=3,
        )
        error.set_ylim(-30, 15)
        error.set_yticks([-20, 0, 10])
        error.set_yticklabels(["−20", "0", "+10"])
        error.text(
            0.015, 0.08, f"mean bias {signed(bias_pp)} pp",
            transform=error.transAxes, ha="left", va="bottom",
            fontsize=6.8, color=TEXT,
        )
        error.xaxis.set_major_locator(mdates.YearLocator(2))
        error.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        finish_axis(error, show_bottom=row == 1)

    signal_axes[0].set_ylabel("Q10 event frequency")
    signal_axes[2].set_ylabel("Q10 event frequency")
    error_axes[0].set_ylabel("Forecast −\nobserved (pp)", labelpad=3)
    error_axes[2].set_ylabel("Forecast −\nobserved (pp)", labelpad=3)
    error_axes[2].set_xlabel("Forecast initialization month")
    error_axes[3].set_xlabel("Forecast initialization month")

    minimum = pd.Timestamp("2015-12-01")
    maximum = pd.Timestamp("2025-11-01")
    for axis in error_axes:
        axis.set_xlim(minimum, maximum)

    legend = [
        Line2D([0], [0], color=FORECAST, lw=1.7, label="Forecast probability"),
        Line2D(
            [0], [0], color=OBSERVED, lw=1.1, ls=(0, (3.2, 2.0)),
            label="Observed USGS frequency",
        ),
        Patch(
            facecolor=UNDER, alpha=0.70, edgecolor="none",
            label="Underforecasting: predicted risk too low",
        ),
        Patch(
            facecolor=OVER, alpha=0.58, edgecolor="none",
            label="Overforecasting: predicted risk too high",
        ),
    ]
    figure.legend(
        handles=legend, frameon=False, ncol=4, loc="lower center",
        bbox_to_anchor=(0.52, 0.018), columnspacing=1.18,
        handlelength=2.3, handletextpad=0.55, fontsize=7.15,
    )
    return figure


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preview-output", type=Path, default=None)
    parser.add_argument(
        "--output-stem", default="figure_07_nature_signal_error_strips",
        help="Filename stem written inside figures/output.",
    )
    args = parser.parse_args()

    apply_style()
    data = load_data()
    figure = build(data)
    stem = OUTPUT / args.output_stem
    figure.savefig(
        stem.with_suffix(".png"), dpi=600,
        bbox_inches="tight", pad_inches=0.04,
    )
    figure.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.04)
    figure.savefig(stem.with_suffix(".svg"), bbox_inches="tight", pad_inches=0.04)
    if args.preview_output is not None:
        args.preview_output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(
            args.preview_output, dpi=190,
            bbox_inches="tight", pad_inches=0.04,
        )
    plt.close(figure)
    print(f"Wrote {stem}.png/.pdf/.svg")


if __name__ == "__main__":
    main()
