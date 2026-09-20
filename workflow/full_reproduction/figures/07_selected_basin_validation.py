#!/usr/bin/env python3
"""Create Figure 7: temporal validation for selected basins.

This four-gage figure reads the archived source table and writes PNG, PDF, and
SVG versions.

Run from the project root:

    python figures/07_selected_basin_validation.py
"""

from __future__ import annotations

import os
from pathlib import Path


FIGURE_DIR = Path(__file__).resolve().parent
SOURCE = FIGURE_DIR / "source_tables" / "figure_05_station_temporal_validation.csv"
OUTPUT = FIGURE_DIR / "output"
CACHE = FIGURE_DIR / ".cache"
OUTPUT.mkdir(parents=True, exist_ok=True)
CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(CACHE / "matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd


STEM = "figure_07_selected_basin_validation"
GAGES = ("01013500", "13313000", "09404115", "02479300")
LEADS = (30, 60, 90, 120)
REGIONS = {
    "NorthEast": "Northeast",
    "WestMnts": "Western Mountains",
    "WestXeric": "Western Xeric",
    "SEPlains": "Southeast Plains",
}
LEAD_COLORS = {
    30: "#176B9B",
    60: "#00827C",
    90: "#D48616",
    120: "#C94D59",
}
INK = "#172B3A"
TEXT = "#2C3E4B"
MUTED = "#687983"
GRID = "#DCE4E7"
BLACK = "#202020"


def configure_style() -> None:
    """Apply the final print-safe typography."""
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
            "font.weight": "bold",
            "axes.titlesize": 9.5,
            "axes.titleweight": "bold",
            "axes.labelsize": 9.7,
            "axes.labelweight": "bold",
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "legend.fontsize": 8.2,
            "axes.linewidth": 0.55,
            "axes.edgecolor": MUTED,
            "axes.labelcolor": TEXT,
            "xtick.color": TEXT,
            "ytick.color": TEXT,
            "xtick.major.width": 0.45,
            "ytick.major.width": 0.45,
            "xtick.major.size": 2.8,
            "ytick.major.size": 2.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.axisbelow": True,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.dpi": 600,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )


def draw() -> None:
    if not SOURCE.exists():
        raise FileNotFoundError(f"Missing archived source table: {SOURCE}")

    configure_style()
    data = pd.read_csv(SOURCE, dtype={"GAGE_ID": str})
    data["GAGE_ID"] = data["GAGE_ID"].str.zfill(8)
    data["issue_date"] = pd.to_datetime(data["issue_date"])

    fig, axes = plt.subplots(
        4,
        4,
        figsize=(7.45, 7.25),
        sharex=True,
        sharey=True,
        gridspec_kw={"hspace": 0.20, "wspace": 0.10},
    )
    # The one-point tick increase needs a quieter four-year cadence so adjacent
    # panel-edge years do not merge when the figure is reduced to journal width.
    years = mdates.YearLocator(4)

    for row_index, gage in enumerate(GAGES):
        gage_data = data.loc[data["GAGE_ID"].eq(gage)]
        if gage_data.empty:
            raise ValueError(f"No temporal-validation data for USGS {gage}")
        group = str(gage_data["spatial_group"].iloc[0])
        region = REGIONS.get(group, group)

        for column_index, lead in enumerate(LEADS):
            axis = axes[row_index, column_index]
            part = gage_data.loc[gage_data["lead_days"].eq(lead)].sort_values(
                "issue_date"
            )
            axis.plot(
                part["issue_date"],
                part["forecast_moving_mean"],
                color=LEAD_COLORS[lead],
                lw=1.45,
                solid_capstyle="round",
            )
            axis.plot(
                part["issue_date"],
                part["observed_moving_mean"],
                color=BLACK,
                lw=1.05,
                ls=(0, (4, 2)),
            )
            event_dates = part.loc[part["observed_event"].eq(1), "issue_date"]
            axis.vlines(event_dates, 0, 0.035, color=BLACK, lw=0.35, alpha=0.28)
            axis.text(
                0.98,
                0.94,
                f"BS = {part['brier_score'].iloc[0]:.3f}",
                transform=axis.transAxes,
                ha="right",
                va="top",
                fontsize=6.5,
                fontweight="bold",
                color=MUTED,
                bbox={
                    "facecolor": "white",
                    "edgecolor": "none",
                    "alpha": 0.78,
                    "pad": 0.5,
                },
            )
            axis.set_ylim(0, 1)
            axis.set_yticks([0, 0.5, 1.0])
            axis.xaxis.set_major_locator(years)
            axis.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
            axis.grid(axis="y", color=GRID, lw=0.55, alpha=0.85)

            for tick_label in [*axis.get_xticklabels(), *axis.get_yticklabels()]:
                tick_label.set_fontweight("bold")

            if row_index == 0:
                axis.set_title(f"{lead}-day window", loc="center", pad=6)
            if column_index == 0:
                axis.text(
                    0.02,
                    0.94,
                    f"{chr(97 + row_index)}  USGS {gage}\n{region}",
                    transform=axis.transAxes,
                    ha="left",
                    va="top",
                    fontsize=7.7,
                    fontweight="bold",
                    color=INK,
                    bbox={
                        "facecolor": "white",
                        "edgecolor": "none",
                        "alpha": 0.82,
                        "pad": 0.8,
                    },
                )
            if row_index == len(GAGES) - 1:
                axis.tick_params(axis="x", rotation=0)

    fig.text(
        0.022,
        0.53,
        "Probability / frequency",
        ha="center",
        va="center",
        rotation=90,
        fontsize=9.7,
        fontweight="bold",
        color=TEXT,
    )
    fig.text(
        0.55,
        0.068,
        "Forecast initialization date",
        ha="center",
        va="center",
        fontsize=9.7,
        fontweight="bold",
        color=TEXT,
    )

    handles = [
        Line2D(
            [],
            [],
            color=MUTED,
            lw=1.6,
            label="Forecast probability (color denotes window)",
        ),
        Line2D(
            [],
            [],
            color=BLACK,
            lw=1.1,
            ls=(0, (4, 2)),
            label="Observed USGS event frequency",
        ),
        Line2D(
            [],
            [],
            color=BLACK,
            marker="|",
            ls="",
            alpha=0.45,
            label="Initialization with observed event",
        ),
    ]
    legend = fig.legend(
        handles=handles,
        frameon=False,
        ncol=3,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.021),
        columnspacing=1.15,
    )
    for legend_text in legend.get_texts():
        legend_text.set_fontweight("bold")

    fig.subplots_adjust(bottom=0.12, top=0.95, left=0.085, right=0.99)
    for suffix, settings in (
        ("png", {"dpi": 600}),
        ("pdf", {}),
        ("svg", {}),
    ):
        fig.savefig(
            OUTPUT / f"{STEM}.{suffix}",
            bbox_inches="tight",
            facecolor="white",
            pad_inches=0.04,
            **settings,
        )
    plt.close(fig)
    print(f"Created Figure 7 files in {OUTPUT}")


if __name__ == "__main__":
    draw()
