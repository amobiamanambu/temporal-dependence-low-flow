#!/usr/bin/env python3
"""Create Figure 8: ecoregional occurrence calibration.

Only the horizontal placement of the three subplot columns is changed. The
data, point sizes, colors, limits, typography, annotations, and legend remain
identical to the core figure.

Run from the project root:

    python figures/08_ecoregional_calibration.py
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path


SCRIPT = Path(__file__).resolve()
FIGURE_ROOT = SCRIPT.parent
IMPLEMENTATION = FIGURE_ROOT / "03_04_05_08_core_figures.py"
OUTPUT = FIGURE_ROOT / "output"
CACHE = FIGURE_ROOT / ".cache"
OUTPUT.mkdir(parents=True, exist_ok=True)
CACHE.mkdir(parents=True, exist_ok=True)
os.environ["MPLCONFIGDIR"] = str(CACHE / "matplotlib")


def load_implementation():
    specification = importlib.util.spec_from_file_location(
        "figure8_accepted_implementation", IMPLEMENTATION
    )
    if specification is None or specification.loader is None:
        raise ImportError(f"Could not load plotting implementation: {IMPLEMENTATION}")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def main() -> None:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    import pandas as pd

    figures = load_implementation()
    figures.configure_style()

    data = pd.read_csv(figures.BASIN_CALIBRATION, dtype={"GAGE_ID": str})
    data["GAGE_ID"] = data["GAGE_ID"].map(figures.normalize_gage)
    data = data[data["lead_days"].isin(figures.LEADS)].copy()
    data["ecoregion"] = data["spatial_group"].map(figures.ECOREGIONS)
    regions = sorted(
        data["spatial_group"].unique(), key=lambda code: figures.ECOREGIONS[code]
    )

    fig, axes = plt.subplots(
        3,
        3,
        figsize=(7.55, 6.45),
        sharex=True,
        sharey=True,
        gridspec_kw={"hspace": 0.26, "wspace": 0.05},
    )
    column_anchors = (0.84, 0.50, 0.16)
    for index, (axis, region) in enumerate(zip(axes.flat, regions)):
        part = data[data["spatial_group"].eq(region)]
        axis.plot(
            [0, 1],
            [0, 1],
            color=figures.INK,
            lw=0.85,
            ls=(0, (4, 2)),
            zorder=1,
        )
        for lead in figures.LEADS:
            lead_data = part[part["lead_days"].eq(lead)]
            axis.scatter(
                lead_data["observed_event_frequency"],
                lead_data["mean_forecast_probability"],
                s=2.55,
                color=figures.LEAD_COLORS[lead],
                alpha=0.115,
                linewidths=0,
                rasterized=True,
                zorder=2,
            )
            axis.scatter(
                [lead_data["observed_event_frequency"].mean()],
                [lead_data["mean_forecast_probability"].mean()],
                s=35,
                marker="D",
                facecolor=figures.LEAD_COLORS[lead],
                edgecolor=figures.INK,
                linewidth=0.72,
                zorder=5,
            )

        axis.set_title(figures.ECOREGIONS[region], loc="left", pad=3)
        axis.text(
            0.97,
            0.05,
            f"{part['GAGE_ID'].nunique():,} basins",
            transform=axis.transAxes,
            ha="right",
            va="bottom",
            fontsize=7.0,
            color=figures.MIDGRAY,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 0.6},
        )
        axis.set_xlim(0, 1)
        axis.set_ylim(0, 1)
        axis.set_xticks([0, 0.25, 0.50, 0.75, 1.0])
        axis.set_yticks([0, 0.25, 0.50, 0.75, 1.0])
        axis.set_aspect("equal", adjustable="box")
        axis.set_anchor((column_anchors[index % 3], 0.5))
        figures.finish_axis(axis, grid_axis="both")
        figures.panel(axis, chr(97 + index), -0.12)

    fig.text(
        0.51,
        0.075,
        "Observed USGS Q10 event frequency",
        ha="center",
        va="center",
        fontsize=9.8,
        fontweight="bold",
    )
    fig.text(
        0.090,
        0.54,
        "Forecast Q10 event probability",
        rotation=90,
        ha="center",
        va="center",
        fontsize=9.8,
        fontweight="bold",
    )

    handles = [
        Line2D(
            [],
            [],
            marker="o",
            ls="",
            color=figures.LEAD_COLORS[lead],
            ms=4.8,
            label=f"{lead} days",
        )
        for lead in figures.LEADS
    ]
    handles.extend(
        [
            Line2D(
                [],
                [],
                marker="D",
                ls="",
                mfc=figures.MIDGRAY,
                mec=figures.INK,
                color=figures.MIDGRAY,
                ms=5.2,
                label="Ecoregion mean",
            ),
            Line2D(
                [],
                [],
                color=figures.INK,
                lw=0.85,
                ls=(0, (4, 2)),
                label="Perfect calibration",
            ),
        ]
    )
    legend = fig.legend(
        handles=handles,
        frameon=False,
        ncol=8,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.006),
        fontsize=6.9,
        columnspacing=0.8,
        handletextpad=0.3,
    )
    for label in legend.get_texts():
        label.set_fontweight("bold")

    fig.subplots_adjust(left=0.085, right=0.995, top=0.975, bottom=0.115)
    original_output = figures.OUTPUT
    try:
        figures.OUTPUT = OUTPUT
        figures.save(fig, "figure_08_ecoregional_calibration")
    finally:
        figures.OUTPUT = original_output
        plt.close(fig)

    print(f"Figure 8 written to {OUTPUT}")


if __name__ == "__main__":
    main()
