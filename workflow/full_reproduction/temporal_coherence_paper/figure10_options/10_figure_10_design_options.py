#!/usr/bin/env python3
"""Create five publication-design options for manuscript Figure 10.

All options display the same archived basin-level Q10 occurrence-calibration
bias at 30, 60, 90, and 120 days. They differ only in typography,
cartographic treatment, marker rendering, and legend design. No option places
median statistics or an explanatory point-count sentence inside the figure.

Run from the project root:

    python temporal_coherence_paper/figure10_options/10_figure_10_design_options.py

Use ``--design 1`` through ``--design 5`` to regenerate one option.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path


PAPER_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = PAPER_ROOT.parent
OPTION_ROOT = PAPER_ROOT / "figure10_options"
OUTPUT = OPTION_ROOT / "output"
CACHE = OPTION_ROOT / ".cache"
DATA = PAPER_ROOT / "temporal_validation" / "basin_horizon_calibration.csv.gz"
SHAPEFILE = PROJECT_ROOT / "gagesII_9322_sept30_2011.shp"
OUTPUT.mkdir(parents=True, exist_ok=True)
CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(CACHE / "matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap, ListedColormap, TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd


LEADS = (30, 60, 90, 120)
CLIP = 35.0


THEMES = {
    1: {
        "name": "classic_traditional",
        "family": "serif",
        "fonts": ["Times New Roman", "Times", "Liberation Serif", "DejaVu Serif"],
        "text": "#202020",
        "secondary": "#555555",
        "land": "#F7F5EF",
        "ocean": "#FFFFFF",
        "lake": "#FFFFFF",
        "water": "#B7C5CB",
        "state": "#A9A9A9",
        "boundary": "#4E4E4E",
        "colors": ["#8A2E2E", "#D29A8E", "#F7F4EE", "#8EAAC2", "#17365D"],
        "graticule": True,
        "coordinate_labels": True,
        "rivers": False,
        "state_labels": False,
        "categorical": False,
        "halo": False,
        "north_scale": False,
        "point_size": 5.2,
        "alpha": 0.83,
        "figure_size": (7.45, 5.55),
        "hspace": 0.08,
        "wspace": 0.03,
        "title_style": "traditional",
    },
    2: {
        "name": "nature_clean",
        "family": "sans-serif",
        "fonts": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
        "text": "#1D2422",
        "secondary": "#58615D",
        "land": "#F1F3F1",
        "ocean": "#FFFFFF",
        "lake": "#FFFFFF",
        "water": "#B8CFD5",
        "state": "#C2C8C5",
        "boundary": "#5E6864",
        "colors": ["#9E3C32", "#D58B7D", "#F5F3EE", "#88ABC5", "#245A84"],
        "graticule": False,
        "coordinate_labels": False,
        "rivers": False,
        "state_labels": False,
        "categorical": False,
        "halo": False,
        "north_scale": False,
        "point_size": 6.3,
        "alpha": 0.87,
        "figure_size": (7.25, 5.25),
        "hspace": 0.13,
        "wspace": 0.035,
        "title_style": "editorial",
    },
    3: {
        "name": "arcgis_light_canvas",
        "family": "sans-serif",
        "fonts": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
        "text": "#283033",
        "secondary": "#626D72",
        "land": "#ECEDEB",
        "ocean": "#F8FAFB",
        "lake": "#F8FAFB",
        "water": "#B8CDD5",
        "state": "#AEB5B6",
        "boundary": "#687174",
        "colors": ["#8C2D26", "#C85A4D", "#E8A196", "#F4F1E9", "#9BBFD2", "#4A88AC", "#1E5578"],
        "graticule": True,
        "coordinate_labels": False,
        "rivers": True,
        "state_labels": True,
        "categorical": True,
        "halo": True,
        "north_scale": False,
        "point_size": 8.0,
        "alpha": 0.92,
        "figure_size": (7.45, 5.60),
        "hspace": 0.09,
        "wspace": 0.025,
        "title_style": "gis",
    },
    4: {
        "name": "arcgis_hydrologic_atlas",
        "family": "sans-serif",
        "fonts": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
        "text": "#24302D",
        "secondary": "#596561",
        "land": "#E9E8DD",
        "ocean": "#EAF2F4",
        "lake": "#EAF2F4",
        "water": "#82AEB9",
        "state": "#9DA69F",
        "boundary": "#52605B",
        "colors": ["#93362F", "#D4836F", "#F4F0DF", "#79A8BD", "#1B5A7A"],
        "graticule": False,
        "coordinate_labels": False,
        "rivers": True,
        "state_labels": True,
        "categorical": False,
        "halo": True,
        "north_scale": True,
        "point_size": 7.2,
        "alpha": 0.91,
        "figure_size": (7.45, 5.52),
        "hspace": 0.08,
        "wspace": 0.025,
        "title_style": "gis",
    },
    5: {
        "name": "editorial_endgame",
        "family": "sans-serif",
        "fonts": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
        "text": "#172B35",
        "secondary": "#53636A",
        "land": "#F0F2EF",
        "ocean": "#F7FAFA",
        "lake": "#F7FAFA",
        "water": "#AFC9D0",
        "state": "#B6BFBB",
        "boundary": "#4F625F",
        "colors": ["#A43B32", "#D98975", "#F6F1E8", "#82AEC6", "#155B82"],
        "graticule": True,
        "coordinate_labels": False,
        "rivers": True,
        "state_labels": False,
        "categorical": False,
        "halo": True,
        "north_scale": False,
        "point_size": 7.8,
        "alpha": 0.94,
        "figure_size": (7.48, 5.20),
        "hspace": 0.045,
        "wspace": 0.018,
        "title_style": "endgame",
    },
}


def apply_style(theme: dict) -> None:
    font_key = "font.serif" if theme["family"] == "serif" else "font.sans-serif"
    plt.rcParams.update({
        "font.family": theme["family"],
        font_key: theme["fonts"],
        "font.size": 8.2,
        "axes.titlesize": 9.6,
        "axes.labelsize": 8.5,
        "xtick.labelsize": 7.0,
        "ytick.labelsize": 7.0,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.dpi": 600,
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
    })


def load_data() -> pd.DataFrame:
    import geopandas as gpd

    if not DATA.exists():
        raise FileNotFoundError(DATA)
    if not SHAPEFILE.exists():
        raise FileNotFoundError(SHAPEFILE)
    calibration = pd.read_csv(DATA, dtype={"GAGE_ID": str})
    calibration["GAGE_ID"] = calibration["GAGE_ID"].str.zfill(8)
    calibration = calibration[calibration["lead_days"].isin(LEADS)].copy()
    calibration["bias_pp"] = 100.0 * calibration["forecast_minus_observed"]
    calibration["bias_clipped_pp"] = calibration["bias_pp"].clip(-CLIP, CLIP)

    points = gpd.read_file(SHAPEFILE).to_crs("EPSG:4326")
    points["GAGE_ID"] = points["STAID"].astype(str).str.zfill(8)
    mapped = points[["GAGE_ID", "geometry"]].merge(
        calibration, on="GAGE_ID", how="inner", validate="one_to_many"
    )
    mapped["longitude"] = mapped.geometry.x
    mapped["latitude"] = mapped.geometry.y

    source = mapped.drop(columns="geometry")
    source.to_csv(OUTPUT / "figure_10_design_options_source.csv.gz", index=False)
    metrics = source.groupby("lead_days", as_index=False).agg(
        basins=("GAGE_ID", "size"),
        median_bias_pp=("bias_pp", "median"),
        mean_bias_pp=("bias_pp", "mean"),
        fraction_predicted_risk_too_low=("bias_pp", lambda values: float((values < 0).mean())),
    )
    metrics.to_csv(OUTPUT / "figure_10_caption_metrics.csv", index=False)
    return mapped


def add_state_labels(axis, color: str) -> None:
    import cartopy.crs as ccrs
    from cartopy.io import shapereader

    path = (
        Path.home() / ".local" / "share" / "cartopy" / "shapefiles" /
        "natural_earth" / "cultural" /
        "ne_10m_admin_1_states_provinces_lakes.shp"
    )
    if not path.exists():
        return
    excluded = {"AK", "HI", "PR", "VI", "GU", "MP", "AS"}
    for record in shapereader.Reader(str(path)).records():
        attributes = record.attributes
        postal = attributes.get("postal")
        if attributes.get("adm0_a3") != "USA" or not postal or postal in excluded:
            continue
        longitude = attributes.get("longitude")
        latitude = attributes.get("latitude")
        if longitude is None or latitude is None:
            continue
        axis.text(
            float(longitude), float(latitude), postal,
            transform=ccrs.PlateCarree(), ha="center", va="center",
            fontsize=3.6, fontweight="bold", color=color, alpha=0.44,
            zorder=2.4,
        )


def draw_base(axis, theme: dict, row: int, column: int) -> None:
    import cartopy
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    cache = Path.home() / ".local" / "share" / "cartopy"
    if cache.exists():
        cartopy.config["pre_existing_data_dir"] = str(cache)
    axis.set_extent([-125, -66.5, 24, 50], crs=ccrs.PlateCarree())
    axis.set_facecolor(theme["ocean"])
    axis.add_feature(
        cfeature.LAND.with_scale("50m"), facecolor=theme["land"],
        edgecolor="none", zorder=0,
    )
    axis.add_feature(
        cfeature.OCEAN.with_scale("50m"), facecolor=theme["ocean"],
        edgecolor="none", zorder=0,
    )
    axis.add_feature(
        cfeature.LAKES.with_scale("50m"), facecolor=theme["lake"],
        edgecolor=theme["water"], linewidth=0.27, zorder=1,
    )
    if theme["rivers"]:
        axis.add_feature(
            cfeature.RIVERS.with_scale("110m"), facecolor="none",
            edgecolor=theme["water"], linewidth=0.30, alpha=0.72, zorder=1,
        )
    axis.add_feature(
        cfeature.STATES.with_scale("50m"), facecolor="none",
        edgecolor=theme["state"], linewidth=0.34, zorder=2,
    )
    axis.add_feature(
        cfeature.COASTLINE.with_scale("50m"), edgecolor=theme["boundary"],
        linewidth=0.55, zorder=2,
    )
    axis.add_feature(
        cfeature.BORDERS.with_scale("50m"), edgecolor=theme["boundary"],
        linewidth=0.50, zorder=2,
    )
    if theme["graticule"]:
        gridliner = axis.gridlines(
            crs=ccrs.PlateCarree(), draw_labels=theme["coordinate_labels"],
            xlocs=[-120, -105, -90, -75], ylocs=[25, 35, 45],
            linewidth=0.27, color="#9AA5A5", alpha=0.38,
            linestyle=(0, (1.2, 2.6)), zorder=1,
        )
        if theme["coordinate_labels"]:
            gridliner.top_labels = False
            gridliner.right_labels = False
            gridliner.bottom_labels = row == 1
            gridliner.left_labels = column == 0
            gridliner.xlabel_style = {"size": 6.2, "color": theme["secondary"]}
            gridliner.ylabel_style = {"size": 6.2, "color": theme["secondary"]}
    if theme["state_labels"]:
        add_state_labels(axis, theme["secondary"])
    axis.spines["geo"].set_edgecolor(theme["boundary"])
    axis.spines["geo"].set_linewidth(0.48 if theme["title_style"] != "gis" else 0.62)


def add_orientation(axis, theme: dict) -> None:
    axis.annotate(
        "N", xy=(0.957, 0.91), xytext=(0.957, 0.82),
        xycoords="axes fraction", textcoords="axes fraction",
        ha="center", va="center", fontsize=6.4, fontweight="bold",
        color=theme["boundary"],
        arrowprops={
            "arrowstyle": "-|>", "color": theme["boundary"], "lw": 0.65,
            "shrinkA": 0, "shrinkB": 0, "mutation_scale": 6.7,
        },
        zorder=8,
    )
    x0, x1, y = 0.055, 0.182, 0.080
    axis.plot(
        [x0, x1], [y, y], transform=axis.transAxes,
        color=theme["boundary"], lw=1.05, solid_capstyle="butt", zorder=8,
    )
    for xvalue in (x0, x1):
        axis.plot(
            [xvalue, xvalue], [y - 0.009, y + 0.009],
            transform=axis.transAxes, color=theme["boundary"], lw=0.62,
            zorder=8,
        )
    axis.text(
        (x0 + x1) / 2, y + 0.012, "1,000 km",
        transform=axis.transAxes, ha="center", va="bottom",
        fontsize=5.4, color=theme["boundary"], zorder=8,
    )


def panel_heading(axis, index: int, lead: int, theme: dict) -> None:
    letter = "abcd"[index]
    if theme["title_style"] == "traditional":
        axis.text(
            0.002, 1.016, letter, transform=axis.transAxes,
            ha="left", va="bottom", fontsize=11.2, fontweight="bold",
            color=theme["text"], clip_on=False,
        )
        axis.set_title(
            f"{lead}-day forecast window", loc="left", x=0.070, pad=4,
            fontsize=9.7, fontweight="normal", color=theme["text"],
        )
    else:
        axis.text(
            0.002, 1.022, letter, transform=axis.transAxes,
            ha="left", va="bottom", fontsize=11.2, fontweight="bold",
            color=theme["text"], clip_on=False,
        )
        axis.text(
            0.078, 1.022, f"{lead} days", transform=axis.transAxes,
            ha="left", va="bottom", fontsize=9.8, fontweight="bold",
            color=theme["text"], clip_on=False,
        )


def build(mapped: pd.DataFrame, theme: dict):
    import cartopy.crs as ccrs

    projection = ccrs.AlbersEqualArea(
        central_longitude=-96, central_latitude=37.5,
        standard_parallels=(29.5, 45.5),
    )
    figure, axes = plt.subplots(
        2, 2, figsize=theme["figure_size"], subplot_kw={"projection": projection},
        gridspec_kw={"hspace": theme["hspace"], "wspace": theme["wspace"]},
    )
    plate_carree = ccrs.PlateCarree()
    scatter = None

    if theme["categorical"]:
        boundaries = np.array([-35.1, -20, -10, -3, 3, 10, 20, 35.1])
        colormap = ListedColormap(theme["colors"], name="arcgis_bias_classes")
        normalization = BoundaryNorm(boundaries, colormap.N)
    else:
        colormap = LinearSegmentedColormap.from_list(
            f"{theme['name']}_bias", theme["colors"], N=256
        )
        normalization = TwoSlopeNorm(vmin=-CLIP, vcenter=0, vmax=CLIP)

    for index, (axis, lead) in enumerate(zip(axes.flat, LEADS)):
        row, column = divmod(index, 2)
        part = mapped[mapped["lead_days"].eq(lead)].copy()
        part["absolute_bias"] = part["bias_clipped_pp"].abs()
        part = part.sort_values("absolute_bias")
        draw_base(axis, theme, row, column)
        if theme["halo"]:
            axis.scatter(
                part["longitude"], part["latitude"], transform=plate_carree,
                s=theme["point_size"] + 3.4, facecolor="white", edgecolor="none",
                alpha=0.72, rasterized=True, zorder=2.8,
            )
        scatter = axis.scatter(
            part["longitude"], part["latitude"],
            c=part["bias_clipped_pp"], cmap=colormap, norm=normalization,
            transform=plate_carree, s=theme["point_size"],
            alpha=theme["alpha"], edgecolor="none", rasterized=True, zorder=3,
        )
        panel_heading(axis, index, lead, theme)
        if theme["north_scale"] and index == 0:
            add_orientation(axis, theme)

    if theme["categorical"]:
        labels = ["≤−20", "−20 to −10", "−10 to −3", "−3 to +3",
                  "+3 to +10", "+10 to +20", "≥+20"]
        handles = [Patch(facecolor=color, edgecolor="none", label=label)
                   for color, label in zip(theme["colors"], labels)]
        legend = figure.legend(
            handles=handles, ncol=7, frameon=False, loc="lower center",
            bbox_to_anchor=(0.5, 0.040), fontsize=6.8,
            handlelength=1.4, handleheight=0.8, columnspacing=0.65,
            handletextpad=0.32,
        )
        figure.text(
            0.5, 0.088, "Calibration bias (percentage points)",
            ha="center", va="bottom", fontsize=8.0, fontweight="bold",
            color=theme["text"],
        )
        figure.text(
            0.025, 0.047, "Predicted risk too low", ha="left", va="center",
            fontsize=6.6, color=theme["colors"][0],
        )
        figure.text(
            0.975, 0.047, "Predicted risk too high", ha="right", va="center",
            fontsize=6.6, color=theme["colors"][-1],
        )
        for text in legend.get_texts():
            text.set_color(theme["text"])
        bottom = 0.120
    else:
        colorbar_y = 0.085 if theme["name"] == "editorial_endgame" else 0.055
        color_axis = figure.add_axes([0.205, colorbar_y, 0.59, 0.020])
        colorbar = figure.colorbar(
            scatter, cax=color_axis, orientation="horizontal", extend="both",
            ticks=[-35, -20, 0, 20, 35],
        )
        colorbar.ax.set_xticklabels(["≤−35", "−20", "0", "+20", "≥+35"])
        colorbar.set_label(
            "Calibration bias (percentage points)", labelpad=2.6,
            color=theme["text"], fontweight="bold" if theme["title_style"] == "endgame" else "normal",
        )
        colorbar.outline.set_linewidth(0.42)
        colorbar.outline.set_edgecolor(theme["boundary"])
        colorbar.ax.tick_params(length=2.4, width=0.45, colors=theme["text"])
        side_label_y = colorbar_y + 0.010
        figure.text(
            0.195, side_label_y, "Predicted risk\ntoo low", ha="right", va="center",
            fontsize=6.7, color=theme["colors"][0],
        )
        figure.text(
            0.805, side_label_y, "Predicted risk\ntoo high", ha="left", va="center",
            fontsize=6.7, color=theme["colors"][-1],
        )
        bottom = 0.120

    figure.subplots_adjust(left=0.030, right=0.990, top=0.965, bottom=bottom)
    return figure


def save(figure, theme: dict) -> None:
    stem = OUTPUT / f"figure_10_option_{theme['name']}"
    figure.savefig(stem.with_suffix(".png"), dpi=600, bbox_inches="tight", pad_inches=0.04)
    figure.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.04)
    figure.savefig(stem.with_suffix(".svg"), bbox_inches="tight", pad_inches=0.04)
    plt.close(figure)
    print(f"Wrote {stem}.png/.pdf/.svg", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--design", choices=["all", "1", "2", "3", "4", "5"], default="all",
        help="design number to generate; default creates all five",
    )
    args = parser.parse_args()
    mapped = load_data()
    selected = THEMES if args.design == "all" else {int(args.design): THEMES[int(args.design)]}
    for theme in selected.values():
        apply_style(theme)
        save(build(mapped, theme), theme)


if __name__ == "__main__":
    main()
