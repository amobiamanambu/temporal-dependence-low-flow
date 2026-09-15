#!/usr/bin/env python3
"""Create the map-led manuscript Figure 1 for the primary Q10 cohort.

The map and ecoregional bars use the 3,402-basin primary Q10-onset forecast
cohort. The accounting panel begins with the 5,227-basin accepted matched-data
inventory and distinguishes model development, confirmation, extension
eligibility, and the primary cohort.

Run from the project root:

    python temporal_coherence_paper/figure1_options/01b_figure_01_nature_study_area.py
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path


PAPER_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = PAPER_ROOT.parent
OPTION_ROOT = PAPER_ROOT / "figure1_options"
OUTPUT = OPTION_ROOT / "output"
SOURCES = OPTION_ROOT / "source_tables"
CACHE = OPTION_ROOT / ".cache"

ANALYSIS_SAMPLE = (
    PAPER_ROOT / "figures_v2" / "source_tables" / "figure_01_analysis_sample.csv"
)
ECOREGIONS = (
    PAPER_ROOT / "figures_v2" / "source_tables" / "figure_01_ecoregions.csv"
)
PRIMARY_METRICS = (
    PAPER_ROOT / "reviewer_strengthening" / "full"
    / "reviewer_strengthening_metrics.csv.gz"
)
SHAPEFILE = PROJECT_ROOT / "gagesII_9322_sept30_2011.shp"

for folder in (OUTPUT, SOURCES, CACHE):
    folder.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(CACHE / "matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


# Colorblind-safe, print-resilient palette.  Reference catchments receive the
# warm accent so that they remain legible above the denser non-reference layer.
INK = "#17211F"
TEXT = "#2D3734"
SECONDARY = "#66706D"
LAND = "#F1F3F0"
OCEAN = "#FFFFFF"
LAKE = "#F4FAFB"
STATE = "#C7CECA"
BOUNDARY = "#626D69"
GRID = "#DDE2DF"
NON_REFERENCE = "#356F82"
REFERENCE = "#D56A3A"
EXCLUDED = "#C99843"
ATTEMPTED = "#D6DCDA"


DESIGNS = {
    "editorial": {
        "stem": "figure_01_option_1_editorial_atlas",
        "land": "#F1F3F0",
        "ocean": "#FFFFFF",
        "lake": "#F4FAFB",
        "state": "#C7CECA",
        "boundary": "#626D69",
        "water_edge": "#B8CFD5",
        "non_reference": "#356F82",
        "reference": "#D56A3A",
        "attempted": "#D6DCDA",
        "excluded": "#C99843",
        "reference_face": "#D56A3A",
        "reference_marker": "o",
        "reference_size": 15.5,
        "reference_linewidth": 0.32,
        "non_reference_size": 7.0,
        "non_reference_alpha": 0.72,
        "rivers": False,
        "graticule": False,
    },
    "hydrologic": {
        "stem": "figure_01_option_2_hydrologic_atlas",
        "land": "#F3F0E7",
        "ocean": "#F8FBFC",
        "lake": "#E8F3F5",
        "state": "#B9BDB5",
        "boundary": "#4C5551",
        "water_edge": "#9FC3CD",
        "non_reference": "#245B6A",
        "reference": "#A83F2D",
        "attempted": "#D9D7CF",
        "excluded": "#B7893B",
        "reference_face": "#F3F0E7",
        "reference_marker": "o",
        "reference_size": 19.0,
        "reference_linewidth": 0.72,
        "non_reference_size": 5.6,
        "non_reference_alpha": 0.66,
        "rivers": True,
        "graticule": False,
    },
    "reference_forward": {
        "stem": "figure_01_option_3_reference_forward",
        "land": "#FAFAF7",
        "ocean": "#FFFFFF",
        "lake": "#F7FAFA",
        "state": "#C6C8C5",
        "boundary": "#424846",
        "water_edge": "#C5D5D8",
        "non_reference": "#8C9692",
        "reference": "#006B5D",
        "attempted": "#D8DCDA",
        "excluded": "#7A8581",
        "reference_face": "#006B5D",
        "reference_marker": "D",
        "reference_size": 13.0,
        "reference_linewidth": 0.28,
        "non_reference_size": 5.4,
        "non_reference_alpha": 0.46,
        "rivers": False,
        "graticule": True,
    },
}


ECOREGION_LABELS = {
    "CntlPlains": "Central Plains",
    "EastHghlnds": "Eastern Highlands",
    "MxWdShld": "Mixed Wood Shield",
    "NorthEast": "Northeast",
    "SEC": "Northern Plains",
    "SECstPlain": "Southeast Coastal Plain",
    "SEPlains": "Southeast Plains",
    "WestMnts": "Western Mountains",
    "WestPlains": "Western Plains",
    "WestXeric": "Western Xeric",
}


def set_style() -> None:
    """Set a restrained, journal-scale visual system."""

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": [
            "Arial", "Helvetica Neue", "Helvetica", "Liberation Sans", "DejaVu Sans"
        ],
        "font.size": 8.0,
        "axes.titlesize": 9.2,
        "axes.labelsize": 8.2,
        "xtick.labelsize": 7.2,
        "ytick.labelsize": 7.2,
        "legend.fontsize": 7.4,
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


def panel_label(axis, label: str, x: float = -0.02, y: float = 1.03) -> None:
    axis.text(
        x, y, label, transform=axis.transAxes, ha="left", va="bottom",
        fontsize=11.2, fontweight="bold", color=INK, clip_on=False,
    )


def clean_axis(axis, grid_axis: str = "x") -> None:
    axis.set_facecolor("white")
    for spine in ("left", "bottom"):
        axis.spines[spine].set_color(INK)
        axis.spines[spine].set_linewidth(0.60)
    axis.grid(axis=grid_axis, color=GRID, linewidth=0.48, alpha=0.90)


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    """Load the nested samples and attach coordinates to the Q10 cohort."""

    import geopandas as gpd

    sample = pd.read_csv(ANALYSIS_SAMPLE, dtype={"GAGE_ID": str})
    sample["GAGE_ID"] = sample["GAGE_ID"].str.zfill(8)
    sample["is_reference"] = (
        sample["is_reference"].astype(str).str.strip().str.lower().eq("true")
    )

    points = gpd.read_file(SHAPEFILE).to_crs("EPSG:4326")
    points["GAGE_ID"] = points["STAID"].astype(str).str.zfill(8)
    points["longitude"] = points.geometry.x
    points["latitude"] = points.geometry.y

    extension = sample[sample["status"].eq("Scored")].copy()
    primary_metrics = pd.read_csv(PRIMARY_METRICS, usecols=["GAGE_ID"], dtype={"GAGE_ID": str})
    primary_ids = set(primary_metrics["GAGE_ID"].str.zfill(8).unique())
    primary = extension[extension["GAGE_ID"].isin(primary_ids)].copy()
    mapped = primary.merge(
        points[["GAGE_ID", "longitude", "latitude"]],
        on="GAGE_ID", how="inner", validate="one_to_one",
    )
    if len(mapped) != len(primary):
        raise RuntimeError(
            f"Mapped {len(mapped):,} of {len(primary):,} primary Q10 catchments"
        )

    ecoregions = (
        mapped.groupby("spatial_group", as_index=False)
        .agg(basins=("GAGE_ID", "nunique"))
    )
    ecoregions["ecoregion"] = (
        ecoregions["spatial_group"].map(ECOREGION_LABELS)
        .fillna(ecoregions["spatial_group"])
    )
    ecoregions = ecoregions.sort_values("basins", ascending=True).reset_index(drop=True)

    counts = {
        "accepted": 5227,
        "development": 192,
        "attempted": int(len(sample)),
        "extension": int(len(extension)),
        "primary": int(len(primary)),
        "no_extension": int(sample["status"].eq("Excluded").sum()),
        "other_target_only": int(len(extension) - len(primary)),
        "reference": int(mapped["is_reference"].sum()),
        "non_reference": int((~mapped["is_reference"]).sum()),
    }
    expected = {
        "accepted": 5227,
        "development": 192,
        "attempted": 5035,
        "extension": 3733,
        "primary": 3402,
        "no_extension": 1302,
        "other_target_only": 331,
        "reference": 619,
        "non_reference": 2783,
    }
    if counts != expected:
        raise RuntimeError(f"Unexpected analysis-population counts: {counts}")
    if counts["accepted"] - counts["development"] != counts["attempted"]:
        raise RuntimeError("Accepted, development, and confirmation counts do not reconcile")

    mapped.to_csv(SOURCES / "figure_01_nature_map_source.csv", index=False)
    pd.DataFrame({
        "analysis_population": [
            "Accepted matched-data inventory",
            "Development sample excluded from confirmation",
            "Confirmation attempted",
            "Extension processing successful",
            "Primary Q10 forecast cohort",
            "No extension result",
            "Other target only",
        ],
        "basins": [
            counts["accepted"],
            counts["development"],
            counts["attempted"],
            counts["extension"],
            counts["primary"],
            counts["no_extension"],
            counts["other_target_only"],
        ],
    }).to_csv(SOURCES / "figure_01_nature_accounting_source.csv", index=False)
    ecoregions.to_csv(SOURCES / "figure_01_nature_ecoregion_source.csv", index=False)
    return mapped, ecoregions, counts


def draw_base_map(axis, design: dict) -> None:
    import cartopy
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    cartopy_cache = Path.home() / ".local" / "share" / "cartopy"
    if cartopy_cache.exists():
        cartopy.config["pre_existing_data_dir"] = str(cartopy_cache)

    axis.set_extent([-125, -66.5, 24, 50], crs=ccrs.PlateCarree())
    axis.set_facecolor(design["ocean"])
    axis.add_feature(
        cfeature.LAND.with_scale("50m"), facecolor=design["land"],
        edgecolor="none", zorder=0,
    )
    axis.add_feature(
        cfeature.OCEAN.with_scale("50m"), facecolor=design["ocean"],
        edgecolor="none", zorder=0,
    )
    axis.add_feature(
        cfeature.LAKES.with_scale("50m"), facecolor=design["lake"],
        edgecolor=design["water_edge"], linewidth=0.30, zorder=1,
    )
    if design["rivers"]:
        axis.add_feature(
            cfeature.RIVERS.with_scale("110m"), facecolor="none",
            edgecolor=design["water_edge"], linewidth=0.35,
            alpha=0.80, zorder=1,
        )
    axis.add_feature(
        cfeature.STATES.with_scale("50m"), facecolor="none",
        edgecolor=design["state"], linewidth=0.38, zorder=2,
    )
    axis.add_feature(
        cfeature.COASTLINE.with_scale("50m"), edgecolor=design["boundary"],
        linewidth=0.58, zorder=2,
    )
    axis.add_feature(
        cfeature.BORDERS.with_scale("50m"), edgecolor=design["boundary"],
        linewidth=0.52, zorder=2,
    )
    if design["graticule"]:
        axis.gridlines(
            crs=ccrs.PlateCarree(), draw_labels=False,
            xlocs=[-120, -105, -90, -75], ylocs=[30, 40, 50],
            linewidth=0.30, color="#AAB1AE", alpha=0.38,
            linestyle=(0, (1.4, 2.8)), zorder=1,
        )
    axis.spines["geo"].set_edgecolor("#AEB7B3")
    axis.spines["geo"].set_linewidth(0.50)


def build_figure(
    mapped: pd.DataFrame,
    ecoregions: pd.DataFrame,
    counts: dict[str, int],
    design: dict,
) -> plt.Figure:
    """Assemble the large map and two aligned lower summary panels."""

    import cartopy.crs as ccrs

    projection = ccrs.AlbersEqualArea(
        central_longitude=-96,
        central_latitude=37.5,
        standard_parallels=(29.5, 45.5),
    )
    figure = plt.figure(figsize=(7.48, 6.05))
    grid = figure.add_gridspec(
        2, 2,
        height_ratios=[2.13, 1.18],
        width_ratios=[0.88, 1.12],
        left=0.055, right=0.985, bottom=0.082, top=0.986,
        hspace=0.11, wspace=0.36,
    )

    map_axis = figure.add_subplot(grid[0, :], projection=projection)
    accounting_axis = figure.add_subplot(grid[1, 0])
    ecoregion_axis = figure.add_subplot(grid[1, 1])

    draw_base_map(map_axis, design)
    map_axis.set_anchor("S")
    plate_carree = ccrs.PlateCarree()
    non_reference = mapped[~mapped["is_reference"]].copy()
    reference = mapped[mapped["is_reference"]].copy()

    # Dense sites form a quiet blue field; the less numerous reference sites
    # are slightly larger, opaque, and outlined so both classes survive print.
    map_axis.scatter(
        non_reference["longitude"], non_reference["latitude"],
        transform=plate_carree, s=design["non_reference_size"],
        c=design["non_reference"], alpha=design["non_reference_alpha"],
        edgecolors="white", linewidths=0.14,
        rasterized=True, zorder=3,
    )
    map_axis.scatter(
        reference["longitude"], reference["latitude"],
        transform=plate_carree, s=design["reference_size"],
        marker=design["reference_marker"],
        facecolors=design["reference_face"],
        edgecolors=(
            design["reference"]
            if design["reference_face"] == design["land"]
            else "white"
        ),
        alpha=0.94, linewidths=design["reference_linewidth"],
        rasterized=True, zorder=4,
    )

    legend = map_axis.legend(
        handles=[
            Line2D(
                [], [], marker=design["reference_marker"], linestyle="", markersize=5.5,
                markerfacecolor=design["reference_face"],
                markeredgecolor=(
                    design["reference"]
                    if design["reference_face"] == design["land"]
                    else "white"
                ),
                markeredgewidth=max(0.45, design["reference_linewidth"]),
                label=f"Reference (n = {counts['reference']:,})",
            ),
            Line2D(
                [], [], marker="o", linestyle="", markersize=5.0,
                markerfacecolor=design["non_reference"], markeredgecolor="white",
                markeredgewidth=0.35,
                label=f"Non-reference (n = {counts['non_reference']:,})",
            ),
        ],
        loc="lower left", bbox_to_anchor=(-0.002, 0.008),
        ncol=1, frameon=True, fancybox=False, framealpha=0.96,
        facecolor="white", edgecolor="#CDD4D1",
        handletextpad=0.45, labelspacing=0.30, borderpad=0.43,
    )
    legend.get_frame().set_linewidth(0.50)
    panel_label(map_axis, "a", x=0.002, y=1.008)

    # Panel b: nested population sequence. The two reductions are described
    # precisely in the caption and source table rather than by a vague
    # "scored/excluded" dichotomy.
    accounting_labels = [
        "Accepted matched-data\ninventory",
        "Confirmation\nattempted",
        "120-day extension\neligible",
        "Primary Q10\ncohort",
    ]
    accounting_values = [
        counts["accepted"], counts["attempted"],
        counts["extension"], counts["primary"]
    ]
    accounting_colors = [
        "#ECE9E0", design["attempted"], "#6F929B", design["non_reference"]
    ]
    y_positions = np.arange(len(accounting_labels))[::-1]
    bars = accounting_axis.barh(
        y_positions, accounting_values,
        height=0.50, color=accounting_colors,
        edgecolor=["#C8C3B8", "#AEB7B3", "#5D818A", design["non_reference"]],
        linewidth=0.60,
    )
    accounting_axis.set_yticks(y_positions, accounting_labels)
    accounting_axis.set_xlim(0, 5700)
    accounting_axis.set_xticks([0, 2000, 4000])
    accounting_axis.set_xlabel("Number of catchments")
    accounting_axis.set_title("Analysis populations", loc="left", pad=5.0)
    clean_axis(accounting_axis, "x")
    accounting_axis.spines["left"].set_visible(False)
    accounting_axis.tick_params(axis="y", length=0, pad=4)
    for bar, value in zip(bars, accounting_values):
        accounting_axis.text(
            value + 95, bar.get_y() + bar.get_height() / 2,
            f"{value:,}", ha="left", va="center",
            fontsize=7.5, color=TEXT,
        )
    accounting_axis.text(
        3850, y_positions[0],
        "192 used for development;\nexcluded from confirmation",
        ha="center", va="center", fontsize=6.4,
        color=SECONDARY, linespacing=1.05,
    )
    panel_label(accounting_axis, "b", x=-0.12, y=1.02)

    # Panel c: classical horizontal bars preserve readable ecoregion names.
    y_eco = np.arange(len(ecoregions))
    normalized = ecoregions["basins"] / ecoregions["basins"].max()
    eco_colors = [
        (*matplotlib.colors.to_rgb(design["non_reference"]), 0.50 + 0.45 * value)
        for value in normalized
    ]
    eco_bars = ecoregion_axis.barh(
        y_eco, ecoregions["basins"], height=0.58,
        color=eco_colors, edgecolor=design["non_reference"], linewidth=0.35,
    )
    ecoregion_axis.set_yticks(y_eco, ecoregions["ecoregion"])
    ecoregion_axis.set_xlim(0, 750)
    ecoregion_axis.set_xticks([0, 200, 400, 600])
    ecoregion_axis.set_xlabel("Primary Q10 catchments")
    ecoregion_axis.set_title("Ecoregional coverage", loc="left", pad=6.0)
    clean_axis(ecoregion_axis, "x")
    ecoregion_axis.spines["left"].set_visible(False)
    ecoregion_axis.tick_params(axis="y", length=0, pad=3.5, labelsize=6.9)
    for bar, value in zip(eco_bars, ecoregions["basins"]):
        ecoregion_axis.text(
            value + 13, bar.get_y() + bar.get_height() / 2,
            f"{int(value):,}", ha="left", va="center",
            fontsize=6.8, color=TEXT,
        )
    panel_label(ecoregion_axis, "c", x=-0.36, y=1.02)

    return figure


def save(
    figure: plt.Figure,
    stem_name: str,
    preview_output: Path | None = None,
) -> Path:
    stem = OUTPUT / stem_name
    figure.savefig(
        stem.with_suffix(".png"), dpi=600,
        bbox_inches="tight", pad_inches=0.045,
    )
    figure.savefig(
        stem.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.045,
    )
    figure.savefig(
        stem.with_suffix(".svg"), bbox_inches="tight", pad_inches=0.045,
    )
    if preview_output is not None:
        preview_output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(
            preview_output, dpi=190,
            bbox_inches="tight", pad_inches=0.045,
        )
    return stem.with_suffix(".png")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--design", choices=["editorial", "hydrologic", "reference_forward", "all"],
        default="all", help="Cartographic design to render (default: all)",
    )
    parser.add_argument("--preview-output", type=Path, default=None)
    args = parser.parse_args()
    set_style()
    mapped, ecoregions, counts = load_data()
    selected = DESIGNS if args.design == "all" else {args.design: DESIGNS[args.design]}
    for name, design in selected.items():
        figure = build_figure(mapped, ecoregions, counts, design)
        preview = args.preview_output if len(selected) == 1 else None
        output = save(figure, design["stem"], preview)
        plt.close(figure)
        print(f"Wrote {name}: {output}")


if __name__ == "__main__":
    main()
