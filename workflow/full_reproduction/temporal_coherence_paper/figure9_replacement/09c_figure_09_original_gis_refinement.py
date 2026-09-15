#!/usr/bin/env python3
"""Create the accepted Figure 9 Option 1b with a GIS-led map design.

This figure preserves the scientific encoding and ecoregional distribution
from ``09a_figure_09_conditional_skill_span.py``. Panel a uses a large
hydrologic GIS basemap; panel b gives the complete distribution of strict
skill-span categories within each ecoregion. This is the author-selected
Option 1b. The active manuscript is not modified by this script.

Run from the project root:

    python temporal_coherence_paper/figure9_replacement/09c_figure_09_original_gis_refinement.py
"""

from __future__ import annotations

import os
from pathlib import Path


PAPER_ROOT = Path(__file__).resolve().parents[1]
FIGURE_ROOT = PAPER_ROOT / "figure9_replacement"
OUTPUT = FIGURE_ROOT / "output"
SOURCES = FIGURE_ROOT / "source_tables"
CACHE = FIGURE_ROOT / ".cache"

MAP_SOURCE = SOURCES / "figure_09_skill_span_basin_map.csv"
REGION_SOURCE = SOURCES / "figure_09_skill_span_ecoregions.csv"

OUTPUT.mkdir(parents=True, exist_ok=True)
CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(CACHE / "matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


SPAN_LEVELS = (0, 30, 45, 60, 90, 105, 120)
SPAN_COLORS = {
    0: "#CCD1CF",
    30: "#A85E4B",
    45: "#C58A45",
    60: "#AAA568",
    90: "#70A28E",
    105: "#2E8878",
    120: "#006B5D",
}

INK = "#17211F"
TEXT = "#303936"
GRID = "#D9DEDB"
LAND = "#F3F0E7"
OCEAN = "#F8FBFC"
LAKE = "#E8F3F5"
WATER = "#9FC3CD"
STATE = "#B9BDB5"
BOUNDARY = "#4C5551"


def set_style() -> None:
    """Set a compact journal visual system with editable vector text."""

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": [
            "Arial", "Helvetica Neue", "Helvetica", "Liberation Sans",
            "DejaVu Sans",
        ],
        "font.size": 8.0,
        "axes.titlesize": 9.2,
        "axes.labelsize": 8.2,
        "xtick.labelsize": 7.2,
        "ytick.labelsize": 7.2,
        "legend.fontsize": 7.0,
        "axes.linewidth": 0.60,
        "axes.edgecolor": INK,
        "axes.labelcolor": TEXT,
        "xtick.color": TEXT,
        "ytick.color": TEXT,
        "xtick.major.width": 0.50,
        "ytick.major.width": 0.50,
        "xtick.major.size": 2.8,
        "ytick.major.size": 2.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.axisbelow": True,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.dpi": 600,
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
    })


def panel_label(axis, label: str, x: float, y: float) -> None:
    axis.text(
        x, y, label, transform=axis.transAxes, ha="left", va="bottom",
        fontsize=11.2, fontweight="bold", color=INK, clip_on=False,
    )


def load_sources() -> tuple[pd.DataFrame, pd.DataFrame]:
    if not MAP_SOURCE.exists() or not REGION_SOURCE.exists():
        raise FileNotFoundError(
            "Archived Figure 9 source tables are missing; run "
            "09a_figure_09_conditional_skill_span.py first."
        )
    mapped = pd.read_csv(MAP_SOURCE, dtype={"GAGE_ID": str})
    regional = pd.read_csv(REGION_SOURCE)
    mapped["GAGE_ID"] = mapped["GAGE_ID"].str.zfill(8)
    return mapped, regional


def draw_gis_base(axis) -> None:
    """Draw a quiet hydrologic atlas beneath the skill-span observations."""

    import cartopy
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    cartopy_cache = Path.home() / ".local" / "share" / "cartopy"
    if cartopy_cache.exists():
        cartopy.config["pre_existing_data_dir"] = str(cartopy_cache)

    axis.set_extent([-125, -66.5, 24, 50], crs=ccrs.PlateCarree())
    axis.set_facecolor(OCEAN)
    axis.add_feature(
        cfeature.LAND.with_scale("50m"), facecolor=LAND,
        edgecolor="none", zorder=0,
    )
    axis.add_feature(
        cfeature.OCEAN.with_scale("50m"), facecolor=OCEAN,
        edgecolor="none", zorder=0,
    )
    axis.add_feature(
        cfeature.LAKES.with_scale("50m"), facecolor=LAKE,
        edgecolor=WATER, linewidth=0.32, zorder=1,
    )
    axis.add_feature(
        cfeature.RIVERS.with_scale("110m"), facecolor="none",
        edgecolor=WATER, linewidth=0.34, alpha=0.78, zorder=1,
    )
    axis.add_feature(
        cfeature.STATES.with_scale("50m"), facecolor="none",
        edgecolor=STATE, linewidth=0.38, zorder=2,
    )
    axis.add_feature(
        cfeature.COASTLINE.with_scale("50m"), edgecolor=BOUNDARY,
        linewidth=0.62, zorder=2,
    )
    axis.add_feature(
        cfeature.BORDERS.with_scale("50m"), edgecolor=BOUNDARY,
        linewidth=0.54, zorder=2,
    )
    axis.spines["geo"].set_edgecolor("#8E9995")
    axis.spines["geo"].set_linewidth(0.62)


def add_north_arrow(axis) -> None:
    """Add a discreet cartographic orientation mark."""

    axis.annotate(
        "N", xy=(0.958, 0.905), xytext=(0.958, 0.805),
        xycoords="axes fraction", textcoords="axes fraction",
        ha="center", va="center", fontsize=7.0, fontweight="bold",
        color=BOUNDARY,
        arrowprops={
            "arrowstyle": "-|>", "color": BOUNDARY, "lw": 0.75,
            "shrinkA": 0, "shrinkB": 0, "mutation_scale": 7.5,
        },
        zorder=8,
    )


def add_scale_bar(axis) -> None:
    """Add a horizontal 1,000-km locator bar at the lower-left margin."""

    # Axes coordinates keep the graphic horizontal in the projected map.  The
    # displayed length is calibrated for the central conterminous United States
    # and is intended as a compact locator scale.
    x0, x1, y = 0.055, 0.182, 0.082
    axis.plot(
        [x0, x1], [y, y], transform=axis.transAxes,
        color=BOUNDARY, linewidth=1.25, solid_capstyle="butt", zorder=8,
    )
    for x in (x0, x1):
        axis.plot(
            [x, x], [y - 0.010, y + 0.010], transform=axis.transAxes,
            color=BOUNDARY, linewidth=0.72, zorder=8,
        )
    axis.text(
        (x0 + x1) / 2, y + 0.014, "1,000 km",
        transform=axis.transAxes, ha="center", va="bottom",
        fontsize=6.1, color=BOUNDARY, zorder=8,
    )


def draw_map(axis, mapped: pd.DataFrame) -> None:
    import cartopy.crs as ccrs

    draw_gis_base(axis)
    axis.set_anchor("S")
    plate_carree = ccrs.PlateCarree()

    # Longest-span sites are plotted first. Shorter-span sites are then still
    # visible in dense regions; the 120-day class remains distinct by shape.
    for span in reversed(SPAN_LEVELS):
        part = mapped[mapped["skill_span_days"].eq(span)]
        axis.scatter(
            part["longitude"], part["latitude"], transform=plate_carree,
            s=11.2 if span == 120 else 7.6,
            marker="D" if span == 120 else "o",
            facecolor=SPAN_COLORS[span], edgecolor="white",
            linewidth=0.22, alpha=0.92 if span else 0.74,
            rasterized=True, zorder=4 if span == 120 else 5,
        )

    # Intentionally no panel-a title: the figure caption carries the full
    # scientific definition and the visual is allowed to function as a map.
    panel_label(axis, "a", x=0.006, y=1.006)
    add_north_arrow(axis)
    add_scale_bar(axis)


def draw_legend(axis) -> None:
    axis.axis("off")
    handles = [
        Line2D(
            [], [], linestyle="", marker="D" if span == 120 else "o",
            markersize=5.2, markerfacecolor=SPAN_COLORS[span],
            markeredgecolor="white", markeredgewidth=0.35, label=str(span),
        )
        for span in SPAN_LEVELS
    ]
    axis.legend(
        handles=handles,
        title="Conditional four-outcome skill span (days)",
        loc="center", ncol=7, frameon=False,
        handletextpad=0.35, columnspacing=1.15, borderaxespad=0,
    )


def draw_ecoregions(axis, regional: pd.DataFrame) -> None:
    """Draw the complete skill-span distribution within each ecoregion."""

    order = (
        regional[["spatial_group", "ecoregion", "display_order", "region_total"]]
        .drop_duplicates().sort_values("display_order")
    )
    y_positions = np.arange(len(order))
    left = np.zeros(len(order), dtype=float)

    for span in SPAN_LEVELS:
        values = []
        for group in order["spatial_group"]:
            row = regional[
                regional["spatial_group"].eq(group)
                & regional["skill_span_days"].eq(span)
            ]
            values.append(float(row["percent"].iloc[0]))
        values = np.asarray(values)
        bars = axis.barh(
            y_positions, values, left=left, height=0.60,
            color=SPAN_COLORS[span], edgecolor="white", linewidth=0.45,
        )
        for bar, value in zip(bars, values):
            if value >= 8.0:
                label_color = "white" if span in (30, 105, 120) else TEXT
                axis.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_y() + bar.get_height() / 2,
                    f"{value:.0f}", ha="center", va="center",
                    fontsize=6.4, color=label_color,
                )
        left += values

    labels = [
        f"{row.ecoregion}  (n = {int(row.region_total):,})"
        for row in order.itertuples()
    ]
    axis.set_yticks(y_positions, labels)
    axis.invert_yaxis()
    axis.set_xlim(0, 100)
    axis.set_xticks([0, 20, 40, 60, 80, 100])
    axis.set_xlabel("Catchments within ecoregion (%)")
    axis.set_title("Ecoregional distribution", loc="left", pad=4.5)
    axis.grid(axis="x", color=GRID, linewidth=0.46, alpha=0.82)
    axis.spines["left"].set_visible(False)
    axis.spines["bottom"].set_color(INK)
    axis.spines["bottom"].set_linewidth(0.60)
    axis.tick_params(axis="y", length=0, pad=4.5, labelsize=6.9)
    panel_label(axis, "b", x=-0.205, y=1.022)


def build_figure(mapped: pd.DataFrame, regional: pd.DataFrame) -> plt.Figure:
    import cartopy.crs as ccrs

    projection = ccrs.AlbersEqualArea(
        central_longitude=-96,
        central_latitude=37.5,
        standard_parallels=(29.5, 45.5),
    )
    figure = plt.figure(figsize=(7.48, 6.72))

    # Independent axes let the map use the full page width while the lower
    # panel retains the left margin required by long ecoregion labels.
    map_axis = figure.add_axes([0.035, 0.445, 0.95, 0.535], projection=projection)
    legend_axis = figure.add_axes([0.12, 0.375, 0.82, 0.052])
    bar_axis = figure.add_axes([0.205, 0.073, 0.765, 0.270])

    draw_map(map_axis, mapped)
    draw_legend(legend_axis)
    draw_ecoregions(bar_axis, regional)
    return figure


def save(figure: plt.Figure) -> Path:
    stem = OUTPUT / "figure_09_option_1b_gis_refined_review"
    figure.savefig(stem.with_suffix(".png"), dpi=600)
    figure.savefig(stem.with_suffix(".pdf"))
    figure.savefig(stem.with_suffix(".svg"))
    return stem.with_suffix(".png")


def main() -> None:
    set_style()
    mapped, regional = load_sources()
    figure = build_figure(mapped, regional)
    output = save(figure)
    plt.close(figure)
    print(f"Wrote {output}")
    print(f"Mapped complete six-window sample: {len(mapped):,} basins")


if __name__ == "__main__":
    main()
