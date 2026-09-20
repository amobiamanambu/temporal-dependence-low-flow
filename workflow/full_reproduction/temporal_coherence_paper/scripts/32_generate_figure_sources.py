#!/usr/bin/env python3
"""Create figure-source tables and the base figure suite.

The script reads only the archived continental results and the versioned public
forecast release.  It writes editable SVG, journal-ready PDF, high-resolution
PNG, and a source CSV for every plotted result.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

PAPER_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = PAPER_ROOT.parent
FIGURE_ROOT = PROJECT_ROOT / "figures"
OUTPUT = FIGURE_ROOT / "output"
SOURCES = FIGURE_ROOT / "source_tables"
CACHE = FIGURE_ROOT / ".cache"
for folder in (OUTPUT, SOURCES, CACHE):
    folder.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(CACHE / "matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

BENCHMARK_SCRIPTS = PROJECT_ROOT / "lowflow_forecast_benchmark" / "scripts"
sys.path.insert(0, str(BENCHMARK_SCRIPTS))
from lib.common import all_accepted_inventory, load_benchmark_config, load_daily  # noqa: E402
from lib.extended_120 import HORIZON, analog_paths_120, extension_cases  # noqa: E402
from lib.extended_trajectory import HYDROGRAPH_FEATURES, first_event_time  # noqa: E402


OLD_ROOT = PROJECT_ROOT / "lowflow_forecast_benchmark" / "results" / "09_extended_forecast"
EXT_ROOT = PROJECT_ROOT / "lowflow_forecast_benchmark" / "results" / "10_extension_120"
RELEASE_ROOTS = (PAPER_ROOT / "zenodo_deposit" / "v1.0.0_upload",)
TEMPORAL_ROOT = PAPER_ROOT / "temporal_validation"
RECONSTRUCTION_ROOT = PAPER_ROOT / "dependence_reconstruction" / "full"
ATTRIBUTES_FILE = PROJECT_ROOT / "continental_run" / "06_basin_index" / "basin_index.csv"
SHAPEFILE = PROJECT_ROOT / "gagesII_9322_sept30_2011.shp"

INK = "#172B3A"
TEXT = "#2C3E4B"
MUTED = "#687983"
GRID = "#DCE4E7"
LAND = "#F3F2ED"
WATER = "#EAF3F6"
BLUE = "#176B9B"
TEAL = "#00827C"
AMBER = "#D48616"
CORAL = "#C94D59"
PURPLE = "#6F58A5"
GRAY = "#8A969D"
PALE = "#EDF1F2"
CORE = ("event_brier", "timing_crps", "duration_crps", "deficit_crps")
OUTCOME_LABELS = {
    "event_brier": "Occurrence",
    "timing_crps": "First-onset time",
    "duration_crps": "Low-flow-day count",
    "deficit_crps": "Cumulative deficit",
    "endpoint_brier": "Endpoint state",
    "minimum_flow_crps": "Minimum flow",
}
OUTCOME_COLORS = dict(zip(CORE, (BLUE, TEAL, AMBER, CORAL)))
ECO_LABELS = {
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
ECO_COLORS = [
    "#0B5D7A", "#2A7F9E", "#62A4B8", "#007D73", "#55A868",
    "#9A9E45", "#D18A23", "#C45C3F", "#7C5AA6", "#A35B82",
]
REPRESENTATIVE_GAGES = ("01013500", "13313000", "09404115", "02479300")
TEMPORAL_LEADS = (30, 60, 90, 120)
LEAD_COLORS = {30: BLUE, 60: TEAL, 90: AMBER, 120: CORAL}

# Restrained palette for the classical journal treatment used in every figure
# except Figures 2 and 5, whose existing design was retained at the author's
# request.  Line type and marker fill always duplicate the color encoding.
CLASSIC_NAVY = "#17365D"
CLASSIC_RED = "#8B2E2E"
CLASSIC_GREEN = "#355E3B"
CLASSIC_GOLD = "#8A6A1F"
CLASSIC_BLACK = "#202020"
CLASSIC_GRAY = "#707070"
CLASSIC_LIGHT = "#D8D8D8"
CLASSIC_PALE = "#F1F1F1"


def style() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "Liberation Serif", "DejaVu Serif"],
        "font.size": 8.5,
        "axes.titlesize": 9.5,
        "axes.labelsize": 8.7,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "legend.fontsize": 7.2,
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
        "mathtext.default": "regular",
        "mathtext.fontset": "stix",
    })


def panel(
    axis, letter: str, x: float = -0.10, y: float = 1.02,
    fontsize: float = 11,
) -> None:
    axis.text(x, y, letter, transform=axis.transAxes, ha="left", va="bottom",
              fontsize=fontsize, fontweight="bold", color=INK, clip_on=False)


def grid(axis, which: str = "y") -> None:
    axis.grid(axis=which, color=GRID, lw=0.55, alpha=0.85)


def traditional_axes(axis, which: str = "y") -> None:
    """Apply a quiet, print-safe hydrology-journal axis treatment."""
    axis.set_facecolor("white")
    for name in ("left", "bottom"):
        axis.spines[name].set_color(CLASSIC_BLACK)
        axis.spines[name].set_linewidth(0.65)
    axis.tick_params(colors=CLASSIC_BLACK, width=0.55)
    axis.xaxis.label.set_color(CLASSIC_BLACK)
    axis.yaxis.label.set_color(CLASSIC_BLACK)
    axis.title.set_color(CLASSIC_BLACK)
    axis.grid(False)
    axis.grid(axis=which, color=CLASSIC_LIGHT, lw=0.45, alpha=0.75)


def save(fig, stem: str) -> None:
    for suffix in ("png", "pdf", "svg"):
        kwargs = {"bbox_inches": "tight", "facecolor": "white", "pad_inches": 0.04}
        if suffix == "png":
            kwargs["dpi"] = 600
        fig.savefig(OUTPUT / f"{stem}.{suffix}", **kwargs)
    plt.close(fig)
    print(f"Wrote {stem}", flush=True)


def source(frame: pd.DataFrame, stem: str) -> None:
    frame.to_csv(SOURCES / f"{stem}.csv", index=False)


def add_conus(axis, classical: bool = True, coordinate_labels: bool = False) -> None:
    import cartopy
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    cache = Path.home() / ".local" / "share" / "cartopy"
    if cache.exists():
        cartopy.config["pre_existing_data_dir"] = str(cache)
    axis.set_extent([-125, -66.5, 24, 50], crs=ccrs.PlateCarree())
    if classical:
        land, water = "#F7F7F5", "white"
        lake_edge, state_edge, boundary = "#AFAFAF", "#BDBDBD", "#4A4A4A"
    else:
        land, water = LAND, WATER
        lake_edge, state_edge, boundary = "#B5CAD2", "#B9C2C6", "#607582"
    axis.set_facecolor(water)
    axis.add_feature(cfeature.LAND.with_scale("50m"), facecolor=land, edgecolor="none")
    axis.add_feature(cfeature.OCEAN.with_scale("50m"), facecolor=water, edgecolor="none")
    axis.add_feature(cfeature.LAKES.with_scale("50m"), facecolor=water,
                     edgecolor=lake_edge, linewidth=0.25)
    axis.add_feature(cfeature.STATES.with_scale("50m"), facecolor="none",
                     edgecolor=state_edge, linewidth=0.30)
    axis.add_feature(cfeature.COASTLINE.with_scale("50m"), edgecolor=boundary, linewidth=0.45)
    axis.add_feature(cfeature.BORDERS.with_scale("50m"), edgecolor=boundary, linewidth=0.45)
    if coordinate_labels:
        gridliner = axis.gridlines(
            crs=ccrs.PlateCarree(), draw_labels=True,
            xlocs=[-120, -105, -90, -75], ylocs=[25, 35, 45],
            linewidth=0.28, color="#9AA8AE", alpha=0.55, linestyle=":",
        )
        gridliner.top_labels = False
        gridliner.right_labels = False
        gridliner.xlabel_style = {"size": 6.0, "color": TEXT}
        gridliner.ylabel_style = {"size": 6.0, "color": TEXT}


def load_results() -> tuple[pd.DataFrame, pd.DataFrame]:
    old = pd.read_csv(OLD_ROOT / "extended_confirmation_metrics.csv.gz", dtype={"GAGE_ID": str})
    ext = pd.read_csv(EXT_ROOT / "extension_120_metrics.csv.gz", dtype={"GAGE_ID": str})
    old["GAGE_ID"] = old["GAGE_ID"].str.zfill(8)
    ext["GAGE_ID"] = ext["GAGE_ID"].str.zfill(8)
    return old, ext


def paired(metrics: pd.DataFrame, reference: str, score: str,
           threshold: str = "Q10", target: str = "onset") -> pd.DataFrame:
    keys = ["GAGE_ID", "spatial_group", "threshold_name", "lead_days", "target"]
    a = metrics[
        metrics["model"].eq("hydrograph_analog")
        & metrics["threshold_name"].eq(threshold)
        & metrics["target"].eq(target)
    ][keys + ["n", score]].dropna(subset=[score])
    b = metrics[
        metrics["model"].eq(reference)
        & metrics["threshold_name"].eq(threshold)
        & metrics["target"].eq(target)
    ][keys + [score]].dropna(subset=[score])
    return a.merge(b, on=keys, suffixes=("_candidate", "_reference"))


def summarize_paired(values: pd.DataFrame, score: str,
                     group: str | None = None) -> pd.DataFrame:
    groups = ["lead_days"] if group is None else [group, "lead_days"]
    rows = []
    for key, part in values.groupby(groups):
        if group is None:
            lead = int(key[0] if isinstance(key, tuple) else key)
            group_value = None
        else:
            group_value, lead = key
        weights = part["n"].to_numpy(float)
        candidate = float(np.average(part[f"{score}_candidate"], weights=weights))
        reference = float(np.average(part[f"{score}_reference"], weights=weights))
        record = {
            "lead_days": int(lead), "candidate_score": candidate,
            "reference_score": reference,
            "relative_skill_percent": 100 * (reference - candidate) / reference,
            "basins": int(len(part)), "cases": int(weights.sum()),
            "basin_fraction_improved": float(
                (part[f"{score}_candidate"] < part[f"{score}_reference"]).mean()
            ),
        }
        if group is not None:
            record[group] = group_value
        rows.append(record)
    return pd.DataFrame(rows)


def joined_curve(reference: str, score: str, old: pd.DataFrame,
                 ext: pd.DataFrame) -> pd.DataFrame:
    a = summarize_paired(paired(old, reference, score), score)
    a = a[a["lead_days"].le(90)].copy()
    b = summarize_paired(paired(ext, reference, score), score)
    b = b[b["lead_days"].isin([105, 120])].copy()
    result = pd.concat([a, b], ignore_index=True).sort_values("lead_days")

    interval_path = RECONSTRUCTION_ROOT / "archived_stratified_basin_intervals.csv"
    if not interval_path.exists():
        raise FileNotFoundError(
            "Run 36_run_dependence_reconstruction.py before generating intervals"
        )
    controls = pd.read_csv(interval_path)
    controls = controls[
        controls["candidate"].eq("hydrograph_analog")
        & controls["reference"].eq(reference)
        & controls["score"].eq(score)
    ][["lead_days", "stratified_ci_low", "stratified_ci_high"]].drop_duplicates("lead_days")
    result = result.merge(controls, on="lead_days", how="left")
    result["relative_ci_low"] = result["stratified_ci_low"]
    result["relative_ci_high"] = result["stratified_ci_high"]
    return result


def figure_1(old: pd.DataFrame, ext: pd.DataFrame) -> None:
    import cartopy.crs as ccrs
    import geopandas as gpd

    inventory = all_accepted_inventory().copy()
    inventory["GAGE_ID"] = inventory["GAGE_ID"].astype(str).str.zfill(8)
    development = pd.read_csv(
        OLD_ROOT.parent / "01_panel" / "candidate_screening_panel.csv", dtype={"GAGE_ID": str}
    )
    development_ids = set(development["GAGE_ID"].str.zfill(8))
    confirmation = inventory[~inventory["GAGE_ID"].isin(development_ids)].copy()
    valid = set(ext["GAGE_ID"])
    confirmation["status"] = np.where(confirmation["GAGE_ID"].isin(valid), "Scored", "Excluded")
    points = gpd.read_file(SHAPEFILE).to_crs("EPSG:4326")
    points["GAGE_ID"] = points["STAID"].astype(str).str.zfill(8)
    mapped = points[["GAGE_ID", "geometry"]].merge(
        confirmation[["GAGE_ID", "status", "spatial_group", "is_reference", "quality_tier"]],
        on="GAGE_ID", how="inner",
    )
    screened = points[points["GAGE_ID"].isin(development_ids)]

    failures = pd.read_csv(
        EXT_ROOT / "extension_120_failures.csv", dtype={"GAGE_ID": str}
    )
    reason_labels = {
        "ValueError('insufficient_training_initializations')": "Too few training initializations",
        "ValueError('insufficient_test_initializations')": "Too few evaluation initializations",
        "ValueError('no_scored_targets')": "No balanced target",
    }
    failures["label"] = failures["reason"].map(reason_labels).fillna("Other")
    reason_counts = failures["label"].value_counts().rename_axis("reason").reset_index(name="basins")
    region_counts = (
        mapped[mapped["status"].eq("Scored")]["spatial_group"]
        .value_counts().rename_axis("spatial_group").reset_index(name="basins")
    )
    region_counts["ecoregion"] = region_counts["spatial_group"].map(ECO_LABELS).fillna(
        region_counts["spatial_group"]
    )
    region_counts = region_counts.sort_values("basins")

    source(mapped.drop(columns="geometry"), "figure_01_analysis_sample")
    source(reason_counts, "figure_01_exclusion_reasons")
    source(region_counts, "figure_01_ecoregions")

    fig = plt.figure(figsize=(7.45, 4.85))
    gs = fig.add_gridspec(
        2, 2, width_ratios=[3.15, 1.42], hspace=0.52, wspace=0.34,
        left=0.025, right=0.99, bottom=0.105, top=0.965,
    )
    axm = fig.add_subplot(gs[:, 0], projection=ccrs.LambertConformal(
        central_longitude=-96, central_latitude=39, standard_parallels=(33, 45)
    ))
    axb = fig.add_subplot(gs[0, 1])
    axr = fig.add_subplot(gs[1, 1])
    add_conus(axm, classical=False, coordinate_labels=True)
    pc = ccrs.PlateCarree()
    bad = mapped[mapped["status"].eq("Excluded")]
    good = mapped[mapped["status"].eq("Scored")]
    axm.scatter(
        screened.geometry.x, screened.geometry.y, transform=pc, s=5.0,
        facecolors="none", edgecolors=PURPLE, linewidths=0.40, alpha=0.80,
        rasterized=True, zorder=4,
    )
    axm.scatter(
        bad.geometry.x, bad.geometry.y, transform=pc, s=4.6, color=AMBER,
        alpha=0.60, linewidth=0, rasterized=True, zorder=5,
    )
    axm.scatter(
        good.geometry.x, good.geometry.y, transform=pc, s=5.3, color=TEAL,
        alpha=0.74, linewidth=0, rasterized=True, zorder=6,
    )
    reference = good[good["is_reference"].astype(bool)]
    axm.scatter(
        reference.geometry.x, reference.geometry.y, transform=pc, s=12,
        facecolors="none", edgecolors=INK, linewidths=0.34,
        rasterized=True, zorder=7,
    )
    axm.legend(handles=[
        Line2D([], [], marker="o", ls="", ms=4.8, color=TEAL,
               label=f"Scored at 105/120 days (n = {len(good):,})"),
        Line2D([], [], marker="o", ls="", ms=4.8, color=AMBER,
               label=f"Rule-based exclusions (n = {len(bad):,})"),
        Line2D([], [], marker="o", ls="", ms=5.0, mfc="none", mec=PURPLE,
               label=f"Development panel, withheld (n = {len(screened):,})"),
        Line2D([], [], marker="o", ls="", ms=5.4, mfc="none", mec=INK,
               label=f"Reference among scored (n = {len(reference):,})"),
    ], frameon=True, fancybox=False, edgecolor="#CAD5D9", framealpha=0.96,
       borderpad=0.5, labelspacing=0.35, loc="lower left")
    panel(axm, "a", 0.005, 1.01)

    labels = ["Attempted", "Scored", "Excluded"]
    counts = [len(confirmation), len(good), len(bad)]
    bar_colors = [PALE, TEAL, AMBER]
    bar_edges = [GRAY, TEAL, AMBER]
    bars = axb.barh(labels[::-1], counts[::-1], color=bar_colors[::-1],
                    edgecolor=bar_edges[::-1], linewidth=0.60)
    for bar, value in zip(bars, counts[::-1]):
        axb.text(value + 75, bar.get_y() + bar.get_height() / 2,
                 f"{value:,}", va="center", color=TEXT, fontsize=7.8)
    axb.set_xlim(0, 5600)
    axb.set_xlabel("Confirmation catchments")
    grid(axb, "x")
    panel(axb, "b", -0.19, 1.02)

    colors = [TEAL if value >= region_counts["basins"].median() else BLUE
              for value in region_counts["basins"]]
    axr.barh(region_counts["ecoregion"], region_counts["basins"], color=colors, alpha=0.88)
    axr.set_xlabel("Scored catchments")
    axr.tick_params(axis="y", labelsize=6.6)
    grid(axr, "x")
    panel(axr, "c", -0.19, 1.02)
    save(fig, "figure_01_study_area_and_sample")


def forecast_case(gage: str) -> dict:
    config = load_benchmark_config()
    inventory = all_accepted_inventory().set_index("GAGE_ID")
    row = inventory.loc[gage]
    daily = load_daily(row["file"])
    scale = daily.loc[daily["date"].le(config["fit_end"]) & daily["q_mm_day"].gt(0), "q_mm_day"].median()
    threshold = float(
        (daily.loc[daily["date"].le(config["threshold_reference_end"]), "q_mm_day"] / scale).quantile(0.10)
    )
    cases = extension_cases(daily, scale, config)
    train = cases[cases["future_date"].le(config["calibration_end"])].copy()
    test = cases[
        cases["date"].ge(config["evaluation_start"])
        & cases["future_date"].le(config["evaluation_end"])
        & cases["q"].gt(threshold)
    ].copy().reset_index(drop=True)
    truth = test[[f"q_h{i}" for i in range(1, 121)]].to_numpy(float)
    observed_time = first_event_time(truth, threshold, "onset")
    eligible = np.flatnonzero((observed_time >= 20) & (observed_time <= 100))
    if len(eligible) == 0:
        eligible = np.flatnonzero(observed_time <= 120)
    choice = int(eligible[len(eligible) // 2]) if len(eligible) else int(len(test) // 2)
    paths = analog_paths_120(
        train, test.iloc[[choice]], HYDROGRAPH_FEATURES, True, 101
    )[0]
    issue = pd.Timestamp(test.iloc[choice]["date"])
    antecedent = daily[
        daily["date"].between(issue - pd.Timedelta(days=30), issue)
    ][["date", "q_mm_day"]].copy()
    antecedent["day"] = (antecedent["date"] - issue).dt.days
    antecedent["q"] = antecedent["q_mm_day"] / scale
    return {
        "GAGE_ID": gage, "issue": issue, "threshold": threshold,
        "truth": truth[choice], "paths": paths, "antecedent": antecedent,
    }


def figure_2(
    output_stem: str = "figure_02_representative_forecast_trajectories",
) -> None:
    cases = [forecast_case(gage) for gage in REPRESENTATIVE_GAGES]
    fig, axes = plt.subplots(2, 2, figsize=(7.45, 5.35), sharex=True,
                             gridspec_kw={"hspace": 0.25, "wspace": 0.24})
    records = []
    for index, (axis, case) in enumerate(zip(axes.flat, cases)):
        future_days = np.arange(1, 121)
        p05, p25, p50, p75, p95 = np.quantile(
            case["paths"], [0.05, 0.25, 0.50, 0.75, 0.95], axis=0
        )
        axis.fill_between(future_days, p05, p95, color=BLUE, alpha=0.14, linewidth=0,
                          label="5–95% forecast")
        axis.fill_between(future_days, p25, p75, color=BLUE, alpha=0.27, linewidth=0,
                          label="25–75% forecast")
        axis.plot(future_days, p50, color=BLUE, lw=1.4, label="Forecast median")
        axis.plot(future_days, case["truth"], color=INK, lw=1.05, label="Observed")
        axis.plot(case["antecedent"]["day"], case["antecedent"]["q"], color=INK, lw=1.05)
        axis.axvline(0, color=GRAY, lw=0.75)
        axis.axhline(case["threshold"], color=CORAL, lw=0.9, ls=(0, (4, 2)), label="Q10")
        axis.fill_between(
            future_days, np.maximum(case["truth"], 1e-5), case["threshold"],
            where=case["truth"] <= case["threshold"], color=CORAL, alpha=0.16, linewidth=0,
        )
        axis.set_yscale("log")
        axis.set_xlim(-30, 120)
        axis.set_title(f"USGS {case['GAGE_ID']} · forecast initialized {case['issue']:%d %b %Y}", loc="left")
        if index % 2 == 0:
            axis.set_ylabel(
                "Normalized daily flow", fontsize=9.7, fontweight="bold"
            )
        else:
            axis.set_ylabel("")
        axis.tick_params(axis="both", labelsize=9.5)
        for tick_label in [*axis.get_xticklabels(), *axis.get_yticklabels()]:
            tick_label.set_fontweight("bold")
        grid(axis)
        panel(axis, "abcd"[index])
        records.append(pd.DataFrame({
            "GAGE_ID": case["GAGE_ID"], "issue_date": case["issue"],
            "lead_days": future_days, "observed": case["truth"],
            "forecast_p05": p05, "forecast_p25": p25, "forecast_p50": p50,
            "forecast_p75": p75, "forecast_p95": p95, "Q10": case["threshold"],
        }))
    for axis in axes[-1, :]:
        axis.set_xlabel(
            "Days from forecast initialization", fontsize=9.7,
            fontweight="bold",
        )
    handles, labels = axes[0, 0].get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    legend = fig.legend(
        unique.values(), unique.keys(), frameon=False, ncol=5,
        loc="lower center", bbox_to_anchor=(0.5, 0.005),
    )
    for legend_text in legend.get_texts():
        legend_text.set_fontweight("bold")
    source(pd.concat(records, ignore_index=True), "figure_02_representative_forecasts")
    save(fig, output_stem)


def figure_3(
    old: pd.DataFrame,
    ext: pd.DataFrame,
    output_stem: str = "figure_03_coherence_changes_path_scores",
) -> None:
    comparison_path = RECONSTRUCTION_ROOT / "reconstruction_comparisons.csv"
    if not comparison_path.exists():
        raise FileNotFoundError(
            "Run 36_run_dependence_reconstruction.py before generating Figure 3"
        )
    review = pd.read_csv(comparison_path)
    fig, axes = plt.subplots(2, 2, figsize=(7.45, 5.15), sharex=True,
                             gridspec_kw={"hspace": 0.25, "wspace": 0.28})
    records = []
    for index, (axis, score) in enumerate(zip(axes.flat, CORE)):
        review_score = "joint_onset_rps" if score == "timing_crps" else score
        coherent = review[
            review["candidate"].eq("hydrograph_analog")
            & review["reference"].eq("independent_daily_shuffle")
            & review["score"].eq(review_score)
        ].sort_values("lead_days")
        reconstructed = review[
            review["candidate"].eq("seasonal_rank_reconstruction")
            & review["reference"].eq("independent_daily_shuffle")
            & review["score"].eq(review_score)
        ].sort_values("lead_days")
        if coherent.empty or reconstructed.empty:
            raise RuntimeError(f"Dependence-reconstruction results missing for {score}")
        data = coherent.copy()
        data["seasonal_reconstruction_score"] = reconstructed[
            "candidate_score"
        ].to_numpy(float)
        data["reported_score"] = score
        records.append(data)
        x = data["lead_days"].to_numpy(float)
        y_coherent = data["candidate_score"].to_numpy(float)
        y_reconstructed = data["seasonal_reconstruction_score"].to_numpy(float)
        y_shuffled = data["reference_score"].to_numpy(float)
        axis.fill_between(x, y_coherent, y_shuffled, where=y_shuffled >= y_coherent,
                          color=CLASSIC_LIGHT, alpha=0.55, linewidth=0)
        axis.plot(x, y_shuffled, color=CLASSIC_BLACK, lw=1.25, ls=(0, (4, 2)),
                  marker="s", ms=3.2, mfc="white", mec=CLASSIC_BLACK, mew=0.7,
                  label="Independent daily shuffle")
        axis.plot(x, y_reconstructed, color=CLASSIC_GREEN, lw=1.35,
                  ls=(0, (6, 2, 1, 2)), marker="D", ms=3.0,
                  mfc="white", mec=CLASSIC_GREEN, mew=0.75,
                  label="Seasonal rank reconstruction")
        axis.plot(x, y_coherent, color=CLASSIC_NAVY, lw=1.75, marker="o", ms=3.4,
                  mfc=CLASSIC_NAVY, mec=CLASSIC_NAVY,
                  label="State-conditioned coherent paths")
        axis.axvline(97.5, color=CLASSIC_GRAY, lw=0.65, ls=":", zorder=0)
        axis.set_title(OUTCOME_LABELS[score], loc="left")
        axis.set_ylabel(
            "Brier score" if "brier" in score else "CRPS",
            fontsize=9.7, fontweight="bold",
        )
        axis.set_xticks([30, 45, 60, 90, 105, 120])
        axis.tick_params(axis="both", labelsize=9.5)
        for tick_label in [*axis.get_xticklabels(), *axis.get_yticklabels()]:
            tick_label.set_fontweight("bold")
        traditional_axes(axis)
        panel(axis, "abcd"[index])
    for axis in axes[-1, :]:
        axis.set_xlabel(
            "Forecast window (days)", fontsize=9.7, fontweight="bold"
        )
    handles, labels = axes[0, 0].get_legend_handles_labels()
    legend = fig.legend(
        handles, labels, frameon=False, ncol=3, loc="lower center",
        bbox_to_anchor=(0.5, -0.005),
    )
    for legend_text in legend.get_texts():
        legend_text.set_fontweight("bold")
    source(pd.concat(records, ignore_index=True), "figure_03_same_marginal_scores")
    save(fig, output_stem)


def figure_s9() -> None:
    """Separate joint onset-or-no-event skill from conditional timing skill."""
    path = RECONSTRUCTION_ROOT / "reconstruction_comparisons.csv"
    if not path.exists():
        raise FileNotFoundError("Run Stage 36 before generating Figure S9")
    data = pd.read_csv(path)
    data = data[
        data["candidate"].eq("hydrograph_analog")
        & data["reference"].eq("independent_daily_shuffle")
        & data["score"].isin(["joint_onset_rps", "conditional_onset_rps"])
    ].copy()
    source(data, "figure_s9_onset_score_decomposition")
    fig, axis = plt.subplots(figsize=(6.2, 3.05))
    for score, label, color, linestyle, marker, face in [
        ("joint_onset_rps", "Joint onset or no event", CLASSIC_NAVY,
         "-", "o", CLASSIC_NAVY),
        ("conditional_onset_rps", "Onset day conditional on an observed event",
         CLASSIC_GREEN, (0, (5, 2)), "D", "white"),
    ]:
        part = data[data["score"].eq(score)].sort_values("lead_days")
        axis.fill_between(
            part["lead_days"], part["stratified_ci_low"],
            part["stratified_ci_high"], color=color, alpha=0.10, linewidth=0,
        )
        axis.plot(
            part["lead_days"], part["relative_skill_percent"], color=color,
            lw=1.55, ls=linestyle, marker=marker, ms=3.5, mfc=face, mec=color,
            mew=0.75, label=label,
        )
    axis.axhline(0, color=CLASSIC_BLACK, lw=0.7)
    axis.axvline(97.5, color=CLASSIC_GRAY, lw=0.65, ls=":")
    axis.set_xticks([30, 45, 60, 90, 105, 120])
    axis.set_xlabel("Forecast window (days)")
    axis.set_ylabel("Score reduction versus daily shuffle (%)")
    traditional_axes(axis)
    axis.legend(frameon=False, loc="upper left")
    fig.subplots_adjust(left=0.13, right=0.985, bottom=0.18, top=0.97)
    save(fig, "figure_s9_onset_score_decomposition")


def figure_s10() -> None:
    """Show sensitivity of coherence skill to each held-out water year."""
    comparison_path = RECONSTRUCTION_ROOT / "reconstruction_comparisons.csv"
    loo_path = RECONSTRUCTION_ROOT / "leave_one_water_year_out.csv"
    if not comparison_path.exists() or not loo_path.exists():
        raise FileNotFoundError("Run Stage 36 before generating Figure S10")
    full = pd.read_csv(comparison_path)
    loo = pd.read_csv(loo_path)
    full = full[
        full["candidate"].eq("hydrograph_analog")
        & full["reference"].eq("independent_daily_shuffle")
    ].copy()
    loo = loo[
        loo["candidate"].eq("hydrograph_analog")
        & loo["reference"].eq("independent_daily_shuffle")
    ].copy()
    score_map = {
        "event_brier": "event_brier",
        "timing_crps": "joint_onset_rps",
        "duration_crps": "duration_crps",
        "deficit_crps": "deficit_crps",
    }
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 4.9), sharex=True,
                             gridspec_kw={"hspace": 0.34, "wspace": 0.28})
    records = []
    for index, (axis, reported_score) in enumerate(zip(axes.flat, CORE)):
        score = score_map[reported_score]
        part = loo[loo["score"].eq(score)].copy()
        summary = part.groupby("lead_days")["relative_skill_percent"].agg(
            minimum="min", maximum="max", median="median"
        ).reset_index()
        estimate = full[full["score"].eq(score)][
            ["lead_days", "relative_skill_percent"]
        ].rename(columns={"relative_skill_percent": "full_estimate"})
        summary = summary.merge(estimate, on="lead_days", how="left")
        summary["score"] = score
        records.append(summary)
        axis.fill_between(
            summary["lead_days"], summary["minimum"], summary["maximum"],
            color=CLASSIC_LIGHT, alpha=0.75, linewidth=0,
            label="Range after omitting one water year",
        )
        axis.plot(
            summary["lead_days"], summary["full_estimate"],
            color=CLASSIC_NAVY, lw=1.55, marker="o", ms=3.2,
            label="All held-out years",
        )
        axis.plot(
            summary["lead_days"], summary["median"],
            color=CLASSIC_BLACK, lw=0.9, ls=(0, (4, 2)),
            label="Leave-one-year-out median",
        )
        axis.axhline(0, color=CLASSIC_BLACK, lw=0.65)
        axis.axvline(97.5, color=CLASSIC_GRAY, lw=0.65, ls=":")
        axis.set_title(OUTCOME_LABELS[reported_score], loc="left")
        axis.set_ylabel("Score reduction (%)")
        traditional_axes(axis)
        panel(axis, "abcd"[index])
    for axis in axes[-1, :]:
        axis.set_xlabel("Forecast window (days)")
        axis.set_xticks([30, 45, 60, 90, 105, 120])
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, ncol=3, loc="lower center",
               bbox_to_anchor=(0.5, -0.005))
    fig.subplots_adjust(bottom=0.13, top=0.97, left=0.09, right=0.99)
    source(pd.concat(records, ignore_index=True),
           "figure_s10_leave_one_water_year_out")
    save(fig, "figure_s10_leave_one_water_year_out")


def figure_4(
    old: pd.DataFrame,
    ext: pd.DataFrame,
    output_stem: str = "figure_04_skill_against_operational_benchmarks",
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(7.45, 5.15), sharex=True,
                             gridspec_kw={"hspace": 0.25, "wspace": 0.20})
    records = []
    for index, (axis, score) in enumerate(zip(axes.flat, CORE)):
        for reference, label, color, linestyle, marker, markerface in [
            ("seasonal_climatology_path", "Seasonal trajectory climatology",
             CLASSIC_NAVY, "-", "o", CLASSIC_NAVY),
            ("constant_persistence_path", "Constant persistence",
             CLASSIC_BLACK, (0, (4, 2)), "s", "white"),
        ]:
            data = joined_curve(reference, score, old, ext)
            data = data[data["lead_days"].ge(30)].copy()
            data["score"] = score
            data["reference"] = reference
            records.append(data)
            x = data["lead_days"].to_numpy(float)
            y = data["relative_skill_percent"].to_numpy(float)
            low = data["relative_ci_low"].to_numpy(float)
            high = data["relative_ci_high"].to_numpy(float)
            band_color = CLASSIC_LIGHT if reference == "constant_persistence_path" else CLASSIC_NAVY
            axis.fill_between(x, low, high, color=band_color,
                              alpha=0.10 if reference != "constant_persistence_path" else 0.22,
                              linewidth=0)
            axis.plot(x, y, color=color, lw=1.55, ls=linestyle, marker=marker,
                      ms=3.4, mfc=markerface, mec=color, mew=0.7, label=label)
        axis.axhline(0, color=CLASSIC_BLACK, lw=0.65)
        axis.set_title(
            OUTCOME_LABELS[score], loc="left", fontsize=10.5,
            fontweight="bold",
        )
        axis.set_ylabel(
            "Relative score reduction (%)", fontsize=9.7,
            fontweight="bold",
        )
        axis.set_xticks([30, 45, 60, 90, 105, 120])
        axis.tick_params(axis="both", labelsize=8.5)
        for tick_label in [*axis.get_xticklabels(), *axis.get_yticklabels()]:
            tick_label.set_fontweight("bold")
        traditional_axes(axis)
        panel(axis, "abcd"[index], fontsize=12)
    for axis in axes[-1, :]:
        axis.set_xlabel(
            "Forecast window (days)", fontsize=9.7, fontweight="bold"
        )
    handles, labels = axes[0, 0].get_legend_handles_labels()
    legend = fig.legend(
        handles, labels, frameon=False, ncol=2, loc="lower center",
        bbox_to_anchor=(0.5, -0.005),
    )
    for legend_text in legend.get_texts():
        legend_text.set_fontweight("bold")
    source(pd.concat(records, ignore_index=True), "figure_04_practical_benchmark_skill")
    save(fig, output_stem)


def station_temporal_series(gage: str) -> pd.DataFrame:
    """Recreate held-out Q10 occurrence forecasts for one representative gage."""
    config = load_benchmark_config()
    inventory = all_accepted_inventory().copy()
    inventory["GAGE_ID"] = inventory["GAGE_ID"].astype(str).str.zfill(8)
    row = inventory.set_index("GAGE_ID").loc[gage]
    daily = load_daily(row["file"])
    fit = daily.loc[
        daily["date"].le(config["fit_end"]) & daily["q_mm_day"].gt(0),
        "q_mm_day",
    ]
    scale = float(fit.median())
    threshold = float(
        (
            daily.loc[
                daily["date"].le(config["threshold_reference_end"]), "q_mm_day"
            ]
            / scale
        ).quantile(0.10)
    )
    cases = extension_cases(daily, scale, config)
    train = cases[cases["future_date"].le(config["calibration_end"])].copy()
    test = cases[
        cases["date"].ge(config["evaluation_start"])
        & cases["future_date"].le(config["evaluation_end"])
        & cases["q"].gt(threshold)
    ].copy().reset_index(drop=True)
    paths = analog_paths_120(train, test, HYDROGRAPH_FEATURES, True, 101)
    truth = test[[f"q_h{day}" for day in range(1, 121)]].to_numpy(float)
    records = []
    for lead in TEMPORAL_LEADS:
        probability = (paths[:, :, :lead] <= threshold).any(axis=2).mean(axis=1)
        observed = (truth[:, :lead] <= threshold).any(axis=1).astype(float)
        part = pd.DataFrame({
            "GAGE_ID": gage,
            "spatial_group": str(row["spatial_group"]),
            "issue_date": pd.to_datetime(test["date"]).to_numpy(),
            "lead_days": lead,
            "forecast_probability": probability,
            "observed_event": observed,
        })
        part["forecast_moving_mean"] = part["forecast_probability"].rolling(
            13, center=True, min_periods=5
        ).mean()
        part["observed_moving_mean"] = part["observed_event"].rolling(
            13, center=True, min_periods=5
        ).mean()
        part["brier_score"] = float(np.mean((probability - observed) ** 2))
        records.append(part)
    return pd.concat(records, ignore_index=True)


def figure_5() -> None:
    """Show when forecasts and observed low-flow occurrence changed together."""
    data = pd.concat(
        [station_temporal_series(gage) for gage in REPRESENTATIVE_GAGES],
        ignore_index=True,
    )
    source(data, "figure_05_station_temporal_validation")
    fig, axes = plt.subplots(
        4, 4, figsize=(7.45, 7.25), sharex=True, sharey=True,
        gridspec_kw={"hspace": 0.20, "wspace": 0.10},
    )
    years = mdates.YearLocator(2)
    for row_index, gage in enumerate(REPRESENTATIVE_GAGES):
        gage_data = data[data["GAGE_ID"].eq(gage)]
        region = ECO_LABELS.get(str(gage_data["spatial_group"].iloc[0]),
                                str(gage_data["spatial_group"].iloc[0]))
        for column_index, lead in enumerate(TEMPORAL_LEADS):
            axis = axes[row_index, column_index]
            part = gage_data[gage_data["lead_days"].eq(lead)].sort_values("issue_date")
            axis.plot(
                part["issue_date"], part["forecast_moving_mean"],
                color=LEAD_COLORS[lead], lw=1.45, solid_capstyle="round",
                label="Forecast probability",
            )
            axis.plot(
                part["issue_date"], part["observed_moving_mean"],
                color="#202020", lw=1.05, ls=(0, (4, 2)),
                label="Observed USGS frequency",
            )
            event_dates = part.loc[part["observed_event"].eq(1), "issue_date"]
            axis.vlines(event_dates, 0, 0.035, color="#202020", lw=0.35, alpha=0.28)
            axis.text(
                0.98, 0.94, f"BS = {part['brier_score'].iloc[0]:.3f}",
                transform=axis.transAxes, ha="right", va="top", fontsize=6.5,
                color=MUTED,
            )
            axis.set_ylim(0, 1)
            axis.set_yticks([0, 0.5, 1.0])
            axis.xaxis.set_major_locator(years)
            axis.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
            grid(axis)
            if row_index == 0:
                axis.set_title(f"{lead}-day window", loc="center", pad=6)
            if column_index == 0:
                axis.set_ylabel("Probability / frequency")
                axis.text(
                    0.02, 0.94, f"{chr(97 + row_index)}  USGS {gage}\n{region}",
                    transform=axis.transAxes, ha="left", va="top",
                    fontsize=7.7, fontweight="bold", color=INK,
                    bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82,
                          "pad": 0.8},
                )
            if row_index == len(REPRESENTATIVE_GAGES) - 1:
                axis.set_xlabel("Forecast initialization date")
                axis.tick_params(axis="x", rotation=0)
    handles = [
        Line2D([], [], color=MUTED, lw=1.6, label="Forecast probability (color denotes window)"),
        Line2D([], [], color="#202020", lw=1.1, ls=(0, (4, 2)),
               label="Observed USGS event frequency"),
        Line2D([], [], color="#202020", marker="|", ls="", alpha=0.45,
               label="Initialization with observed event"),
    ]
    fig.legend(handles=handles, frameon=False, ncol=3, loc="lower center",
               bbox_to_anchor=(0.5, 0.005), columnspacing=1.3)
    fig.subplots_adjust(bottom=0.095, top=0.95, left=0.10, right=0.99)
    save(fig, "figure_05_station_temporal_validation")


def figure_6(
    output_stem: str = "figure_06_ecoregion_forecast_observed",
) -> None:
    """Compare basin-first forecast and observed event frequencies by ecoregion."""
    path = TEMPORAL_ROOT / "ecoregion_horizon_calibration.csv"
    if not path.exists():
        print("Skipping Figure 6 until Stage 34 is complete", flush=True)
        return
    data = pd.read_csv(path)
    data["ecoregion"] = data["spatial_group"].map(ECO_LABELS).fillna(data["spatial_group"])
    source(data, "figure_06_ecoregion_forecast_observed")
    regions = sorted(data["spatial_group"].unique(), key=lambda x: ECO_LABELS.get(x, x))
    fig, axes = plt.subplots(
        3, 3, figsize=(7.45, 6.15), sharex=True, sharey=True,
        gridspec_kw={"hspace": 0.30, "wspace": 0.16},
    )
    for index, (axis, region) in enumerate(zip(axes.flat, regions)):
        part = data[data["spatial_group"].eq(region)].sort_values("lead_days")
        x = part["lead_days"].to_numpy(float)
        forecast = part["mean_forecast_probability"].to_numpy(float)
        observed = part["observed_event_frequency"].to_numpy(float)
        axis.fill_between(
            x, part["forecast_ci_low"].to_numpy(float),
            part["forecast_ci_high"].to_numpy(float), color=CLASSIC_NAVY,
            alpha=0.10, linewidth=0,
        )
        axis.fill_between(
            x, part["observed_ci_low"].to_numpy(float),
            part["observed_ci_high"].to_numpy(float), color="#4B4B4B",
            alpha=0.13, linewidth=0,
        )
        axis.plot(x, forecast, color=CLASSIC_NAVY, lw=1.50, marker="o", ms=2.8,
                  mfc=CLASSIC_NAVY, mec=CLASSIC_NAVY,
                  label="Forecast probability")
        axis.plot(x, observed, color=CLASSIC_BLACK, lw=1.15, ls=(0, (4, 2)),
                  marker="s", ms=2.8, mfc="white", mec=CLASSIC_BLACK, mew=0.65,
                  label="Observed USGS frequency")
        axis.set_title(
            ECO_LABELS.get(region, region), loc="left", pad=3,
            fontweight="bold",
        )
        axis.text(
            0.98, 0.05, f"{int(part['basins'].min()):,} basins",
            transform=axis.transAxes, ha="right", va="bottom", fontsize=6.5,
            color=MUTED, fontweight="bold",
        )
        axis.set_xlim(0, 123)
        axis.set_ylim(0, 0.60)
        axis.set_xticks([1, 30, 60, 90, 120])
        axis.set_yticks([0, 0.2, 0.4, 0.6])
        traditional_axes(axis)
        axis.tick_params(axis="both", labelsize=8.5)
        for tick_label in [*axis.get_xticklabels(), *axis.get_yticklabels()]:
            tick_label.set_fontweight("bold")
        panel(axis, chr(97 + index), -0.10, 1.02)
    for axis in axes[:, 0]:
        axis.set_ylabel(
            "Q10 event probability", fontsize=9.7, fontweight="bold"
        )
    for axis in axes[-1, :]:
        axis.set_xlabel(
            "Forecast window (days)", fontsize=9.7, fontweight="bold"
        )
    handles, labels = axes[0, 0].get_legend_handles_labels()
    legend = fig.legend(
        handles, labels, frameon=False, ncol=2, loc="lower center",
        bbox_to_anchor=(0.5, 0.003), fontsize=8.2,
    )
    for legend_text in legend.get_texts():
        legend_text.set_fontweight("bold")
    fig.subplots_adjust(bottom=0.105, top=0.97, left=0.09, right=0.99)
    save(fig, output_stem)


def figure_7(old: pd.DataFrame, ext: pd.DataFrame) -> None:
    """Put continental occurrence forecasts back on the held-out calendar."""
    leads = (30, 60, 90, 120)
    paths = []
    for lead in leads:
        candidates = [
            root / f"forecast_predictions_q10_lead_{lead:03d}.parquet"
            for root in RELEASE_ROOTS
        ]
        paths.append(next((path for path in candidates if path.exists()), candidates[0]))
    if not all(path.exists() for path in paths):
        print("Skipping Figure 7 until Stage 31 has assembled the public dataset", flush=True)
        return

    fig, axes = plt.subplots(
        2, 2, figsize=(7.45, 5.25), sharex=True, sharey=True,
        gridspec_kw={"hspace": 0.27, "wspace": 0.16},
    )
    records = []
    for index, (axis, lead, path) in enumerate(zip(axes.flat, leads, paths)):
        common_columns = [
            "GAGE_ID", "forecast_event_probability", "observed_event",
            "lead_is_scorable",
        ]
        try:
            frame = pd.read_parquet(
                path, columns=["initialization_date", *common_columns]
            ).rename(columns={"initialization_date": "issue_date"})
        except Exception:
            # The earlier local release used ``issue_date``; the public archive
            # uses the clearer ``initialization_date`` label.
            frame = pd.read_parquet(
                path, columns=["issue_date", *common_columns]
            )
        frame = frame[frame["lead_is_scorable"]].copy()
        frame["GAGE_ID"] = frame["GAGE_ID"].astype(str).str.zfill(8)
        frame["month"] = pd.to_datetime(frame["issue_date"]).dt.to_period("M").dt.to_timestamp()
        basin_month = frame.groupby(["GAGE_ID", "month"], as_index=False).agg(
            forecast_probability=("forecast_event_probability", "mean"),
            observed_frequency=("observed_event", "mean"),
            initializations=("observed_event", "size"),
        )
        monthly = basin_month.groupby("month", as_index=False).agg(
            forecast_probability=("forecast_probability", "mean"),
            observed_frequency=("observed_frequency", "mean"),
            basins=("GAGE_ID", "nunique"),
            initializations=("initializations", "sum"),
        )
        monthly["lead_days"] = lead
        monthly["forecast_minus_observed"] = (
            monthly["forecast_probability"] - monthly["observed_frequency"]
        )
        records.append(monthly)

        x = pd.to_datetime(monthly["month"])
        forecast = monthly["forecast_probability"].to_numpy(float)
        observed = monthly["observed_frequency"].to_numpy(float)
        axis.fill_between(
            x, forecast, observed, where=forecast < observed,
            color=CORAL, alpha=0.12, interpolate=True, linewidth=0,
        )
        axis.fill_between(
            x, forecast, observed, where=forecast >= observed,
            color=BLUE, alpha=0.08, interpolate=True, linewidth=0,
        )
        axis.plot(x, forecast, color=CLASSIC_NAVY, lw=1.35,
                  label="Forecast probability")
        axis.plot(x, observed, color=CLASSIC_BLACK, lw=1.02,
                  ls=(0, (4, 2)), label="Observed USGS frequency")
        correlation = float(np.corrcoef(forecast, observed)[0, 1])
        mean_bias = float(np.mean(forecast - observed))
        axis.text(
            0.025, 0.95,
            f"monthly r = {correlation:.2f}\nmean bias = {100 * mean_bias:+.1f} points",
            transform=axis.transAxes, ha="left", va="top", fontsize=7.1,
            color=CLASSIC_BLACK,
        )
        axis.text(
            0.975, 0.95,
            f"median basins/month = {int(monthly['basins'].median()):,}",
            transform=axis.transAxes, ha="right", va="top", fontsize=6.5,
            color=CLASSIC_GRAY,
        )
        axis.set_title(f"{lead}-day forecast window", loc="left")
        axis.set_ylim(0, 0.78)
        axis.set_yticks([0, 0.2, 0.4, 0.6])
        axis.xaxis.set_major_locator(mdates.YearLocator(2))
        axis.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        traditional_axes(axis)
        panel(axis, "abcd"[index])
    for axis in axes[:, 0]:
        axis.set_ylabel("Basin-first Q10 event frequency")
    for axis in axes[-1, :]:
        axis.set_xlabel("Forecast initialization month")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, ncol=2, loc="lower center",
               bbox_to_anchor=(0.5, 0.003), columnspacing=1.8)
    fig.subplots_adjust(bottom=0.105, left=0.09, right=0.99, top=0.97)
    source(pd.concat(records, ignore_index=True), "figure_07_continental_temporal_validation")
    save(fig, "figure_07_continental_temporal_validation")


def map_pairs(metrics: pd.DataFrame, lead: int) -> pd.DataFrame:
    values = paired(metrics, "hydrograph_marginal_shuffle", "event_brier")
    values = values[values["lead_days"].eq(lead)].copy()
    values["relative_skill_percent"] = 100 * (
        values["event_brier_reference"] - values["event_brier_candidate"]
    ) / values["event_brier_reference"].replace(0, np.nan)
    return values


def figure_8(old: pd.DataFrame, ext: pd.DataFrame) -> None:
    import cartopy.crs as ccrs
    import geopandas as gpd

    points = gpd.read_file(SHAPEFILE).to_crs("EPSG:4326")
    points["GAGE_ID"] = points["STAID"].astype(str).str.zfill(8)
    data90 = map_pairs(old, 90)
    data120 = map_pairs(ext, 120)
    mapped90 = points[["GAGE_ID", "geometry"]].merge(data90, on="GAGE_ID", how="inner")
    mapped120 = points[["GAGE_ID", "geometry"]].merge(data120, on="GAGE_ID", how="inner")
    source(mapped90.drop(columns="geometry"), "figure_08_basin_skill_090")
    source(mapped120.drop(columns="geometry"), "figure_08_basin_skill_120")

    regional = mapped120.groupby("spatial_group")["relative_skill_percent"].agg(
        basins="count",
        q25=lambda values: values.quantile(0.25),
        median="median",
        q75=lambda values: values.quantile(0.75),
        fraction_improved=lambda values: float((values > 0).mean()),
    ).reset_index()
    regional["ecoregion"] = regional["spatial_group"].map(ECO_LABELS).fillna(
        regional["spatial_group"]
    )
    regional = regional.sort_values("median")
    source(regional, "figure_08_ecoregion_summary_120")

    fig = plt.figure(figsize=(7.45, 3.65))
    projection = ccrs.LambertConformal(
        central_longitude=-96, central_latitude=39, standard_parallels=(33, 45)
    )
    gs = fig.add_gridspec(1, 2, width_ratios=[1.18, 1.0], wspace=0.34,
                          left=0.025, right=0.985, bottom=0.15, top=0.95)
    axm = fig.add_subplot(gs[0, 0], projection=projection)
    add_conus(axm, coordinate_labels=True)
    improved = mapped120[mapped120["relative_skill_percent"].gt(0)]
    not_improved = mapped120[~mapped120["relative_skill_percent"].gt(0)]
    pc = ccrs.PlateCarree()
    axm.scatter(
        improved.geometry.x, improved.geometry.y, transform=pc, s=4.4,
        color=CLASSIC_NAVY, alpha=0.58, linewidth=0, rasterized=True,
    )
    axm.scatter(
        not_improved.geometry.x, not_improved.geometry.y, transform=pc, s=8.5,
        facecolor="white", edgecolor=CLASSIC_RED, alpha=0.85,
        linewidth=0.55, rasterized=True,
    )
    axm.set_title("120-day Q10 occurrence", loc="left", x=0.07)
    axm.legend(handles=[
        Line2D([], [], marker="o", ls="", ms=4.6, color=CLASSIC_NAVY,
               label=f"Coherent paths better (n = {len(improved):,})"),
        Line2D([], [], marker="o", ls="", ms=4.6, mfc="white", mec=CLASSIC_RED,
               label=f"No improvement (n = {len(not_improved):,})"),
    ], frameon=False, loc="lower left", borderaxespad=0.2, handletextpad=0.4)
    panel(axm, "a", 0.00, 1.01)

    axr = fig.add_subplot(gs[0, 1])
    y = np.arange(len(regional))
    axr.hlines(y, regional["q25"], regional["q75"], color=CLASSIC_BLACK,
               lw=1.05, zorder=2)
    axr.scatter(regional["median"], y, s=28, color=CLASSIC_NAVY,
                edgecolor="white", linewidth=0.45, zorder=3)
    axr.axvline(0, color=CLASSIC_RED, lw=0.75, ls=(0, (3, 2)))
    display_labels = regional["ecoregion"].replace(
        {"Southeast Coastal Plain": "SE Coastal Plain"}
    )
    axr.set_yticks(y, display_labels)
    axr.set_xlim(-5, 93)
    axr.set_xticks([0, 20, 40, 60, 80])
    axr.set_xlabel("120-day Brier-score reduction (%)")
    axr.set_title("Median and interquartile range", loc="left")
    traditional_axes(axr, "x")
    for ypos, row in enumerate(regional.itertuples()):
        axr.text(92, ypos, f"{100 * row.fraction_improved:.0f}%",
                 ha="right", va="center", fontsize=6.8, color=CLASSIC_BLACK)
    axr.text(0.99, 1.015, "Basins improved", transform=axr.transAxes,
             ha="right", va="bottom", fontsize=6.7, color=CLASSIC_GRAY)
    panel(axr, "b", -0.14, 1.01)
    save(fig, "figure_08_spatial_distribution_of_skill")


def figure_9() -> None:
    """Map basin-level occurrence calibration bias at short and long windows."""
    import cartopy.crs as ccrs
    import geopandas as gpd
    from matplotlib.colors import TwoSlopeNorm

    path = TEMPORAL_ROOT / "basin_horizon_calibration.csv.gz"
    if not path.exists():
        print("Skipping Figure 9 until Stage 34 is complete", flush=True)
        return
    data = pd.read_csv(path, dtype={"GAGE_ID": str})
    data["GAGE_ID"] = data["GAGE_ID"].str.zfill(8)
    data = data[data["lead_days"].isin([30, 120])].copy()
    data["calibration_bias_clipped"] = data["forecast_minus_observed"].clip(-0.35, 0.35)
    points = gpd.read_file(SHAPEFILE).to_crs("EPSG:4326")
    points["GAGE_ID"] = points["STAID"].astype(str).str.zfill(8)
    mapped = points[["GAGE_ID", "geometry"]].merge(data, on="GAGE_ID", how="inner")
    source(mapped.drop(columns="geometry"), "figure_09_spatial_calibration_bias")

    projection = ccrs.LambertConformal(
        central_longitude=-96, central_latitude=39, standard_parallels=(33, 45)
    )
    fig, axes = plt.subplots(
        1, 2, figsize=(7.45, 3.45), subplot_kw={"projection": projection},
        gridspec_kw={"wspace": 0.04},
    )
    norm = TwoSlopeNorm(vmin=-0.35, vcenter=0, vmax=0.35)
    scatter = None
    pc = ccrs.PlateCarree()
    for index, (axis, lead) in enumerate(zip(axes, (30, 120))):
        part = mapped[mapped["lead_days"].eq(lead)]
        add_conus(axis, classical=False, coordinate_labels=True)
        scatter = axis.scatter(
            part.geometry.x, part.geometry.y,
            c=part["calibration_bias_clipped"], cmap="RdBu", norm=norm,
            transform=pc, s=7.0, alpha=0.86, linewidth=0, rasterized=True,
        )
        median = float(part["forecast_minus_observed"].median())
        if abs(median) < 0.0005:
            median = 0.0
        under = float((part["forecast_minus_observed"] < 0).mean())
        axis.set_title(f"{lead}-day forecast window", loc="left", x=0.05)
        axis.text(
            0.05, 0.04,
            f"median bias {100 * median:+.1f} points\n{100 * under:.0f}% of basins underforecast",
            transform=axis.transAxes, ha="left", va="bottom", fontsize=6.8,
            color=CLASSIC_BLACK,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.80, "pad": 1.8},
        )
        panel(axis, "ab"[index], 0.00, 1.01)
    cbar = fig.colorbar(
        scatter, ax=axes, orientation="horizontal", fraction=0.055, pad=0.035,
        aspect=38, ticks=[-0.35, -0.2, 0, 0.2, 0.35], extend="both",
    )
    cbar.set_label("Forecast probability − observed event frequency")
    cbar.ax.set_xticklabels(["≤−0.35", "−0.20", "0", "+0.20", "≥+0.35"])
    cbar.outline.set_linewidth(0.45)
    fig.subplots_adjust(left=0.02, right=0.99, top=0.94, bottom=0.16)
    save(fig, "figure_09_spatial_calibration_bias")


def figure_s7() -> None:
    paths = []
    for lead in (30, 45, 60, 90):
        candidates = [
            root / f"forecast_predictions_q10_lead_{lead:03d}.parquet"
            for root in RELEASE_ROOTS
        ]
        paths.append(next((path for path in candidates if path.exists()), candidates[0]))
    if not all(path.exists() for path in paths):
        print("Skipping Figure S7 until Stage 31 has assembled the public dataset", flush=True)
        return
    fig, axes = plt.subplots(2, 2, figsize=(7.45, 5.35), sharex=True, sharey=True,
                             gridspec_kw={"hspace": 0.30, "wspace": 0.18})
    records = []
    for index, (axis, lead, path) in enumerate(zip(axes.flat, (30, 45, 60, 90), paths)):
        frame = pd.read_parquet(
            path,
            columns=["forecast_event_probability", "observed_event", "lead_is_scorable"],
        )
        frame = frame[frame["lead_is_scorable"]].copy()
        frame["bin"] = pd.cut(frame["forecast_event_probability"],
                              bins=np.linspace(0, 1, 21), include_lowest=True)
        summary = frame.groupby("bin", observed=True).agg(
            probability=("forecast_event_probability", "mean"),
            observed_frequency=("observed_event", "mean"),
            n=("observed_event", "size"),
        ).reset_index(drop=True)
        se = np.sqrt(summary["observed_frequency"] * (1 - summary["observed_frequency"]) / summary["n"])
        summary["ci_low"] = (summary["observed_frequency"] - 1.96 * se).clip(0, 1)
        summary["ci_high"] = (summary["observed_frequency"] + 1.96 * se).clip(0, 1)
        summary["lead_days"] = lead
        records.append(summary)
        axis.plot([0, 1], [0, 1], color=CLASSIC_BLACK, lw=0.75, ls=(0, (4, 2)))
        axis.fill_between(summary["probability"], summary["ci_low"], summary["ci_high"],
                          color=CLASSIC_NAVY, alpha=0.10, linewidth=0)
        axis.plot(summary["probability"], summary["observed_frequency"],
                  color=CLASSIC_NAVY, lw=1.55, marker="o", ms=3.0,
                  mfc=CLASSIC_NAVY, mec=CLASSIC_NAVY)
        counts, edges = np.histogram(frame["forecast_event_probability"], bins=np.linspace(0, 1, 21))
        height = counts / counts.max() * 0.16
        axis.bar(edges[:-1], height, width=np.diff(edges), align="edge",
                 color=CLASSIC_LIGHT, alpha=0.70, edgecolor="none")
        brier = np.mean((frame["forecast_event_probability"] - frame["observed_event"].astype(float)) ** 2)
        axis.text(0.04, 0.94, f"{len(frame):,} forecasts\nBrier = {brier:.3f}",
                  transform=axis.transAxes, ha="left", va="top", color=TEXT)
        axis.set_title(f"{lead}-day occurrence", loc="left")
        axis.set_xlim(0, 1)
        axis.set_ylim(0, 1)
        traditional_axes(axis, "both")
        panel(axis, "abcd"[index])
    for axis in axes[:, 0]:
        axis.set_ylabel("Observed event frequency")
    for axis in axes[-1, :]:
        axis.set_xlabel("Forecast probability of ≥1 Q10 crossing")
    source(pd.concat(records, ignore_index=True), "figure_s7_occurrence_reliability")
    save(fig, "figure_s7_occurrence_reliability")


def figure_s8(old: pd.DataFrame, ext: pd.DataFrame) -> None:
    """Show the complete ecoregional coherence trajectories for all core outcomes."""
    records = []
    for score in CORE:
        a = summarize_paired(
            paired(old, "hydrograph_marginal_shuffle", score), score, "spatial_group"
        )
        b = summarize_paired(
            paired(ext, "hydrograph_marginal_shuffle", score), score, "spatial_group"
        )
        data = pd.concat(
            [a[a["lead_days"].isin([30, 45, 60, 90])],
             b[b["lead_days"].isin([105, 120])]],
            ignore_index=True,
        )
        data["score"] = score
        records.append(data)
    all_data = pd.concat(records, ignore_index=True)
    source(all_data, "figure_s8_full_ecoregional_horizon_curves")

    regions = sorted(all_data["spatial_group"].unique(), key=lambda x: ECO_LABELS.get(x, x))
    styles = {
        "event_brier": (BLUE, "-", "o"),
        "timing_crps": (TEAL, (0, (5, 2)), "s"),
        "duration_crps": (AMBER, (0, (2, 1.5)), "^"),
        "deficit_crps": (CORAL, (0, (6, 2, 1.5, 2)), "D"),
    }
    fig, axes = plt.subplots(
        3, 3, figsize=(7.45, 6.15), sharex=True, sharey=True,
        gridspec_kw={"hspace": 0.30, "wspace": 0.14},
    )
    for index, (axis, region) in enumerate(zip(axes.flat, regions)):
        part = all_data[all_data["spatial_group"].eq(region)]
        for score in CORE:
            curve = part[part["score"].eq(score)].sort_values("lead_days")
            color, linestyle, marker = styles[score]
            axis.plot(
                curve["lead_days"], curve["relative_skill_percent"],
                color=color, lw=1.35, ls=linestyle, marker=marker, ms=3.0,
                mfc="white" if score in ("timing_crps", "deficit_crps") else color,
                mec=color, mew=0.65, label=OUTCOME_LABELS[score],
            )
        axis.axhline(0, color=CLASSIC_BLACK, lw=0.60)
        axis.set_title(ECO_LABELS.get(region, region), loc="left", pad=3)
        axis.set_xlim(27, 123)
        axis.set_ylim(12, 82)
        axis.set_xticks([30, 60, 90, 120])
        axis.set_yticks([20, 40, 60, 80])
        traditional_axes(axis)
        panel(axis, chr(97 + index), -0.10, 1.02)
    for axis in axes[:, 0]:
        axis.set_ylabel("Score reduction from coherence (%)")
    for axis in axes[-1, :]:
        axis.set_xlabel("Forecast window (days)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, ncol=4, loc="lower center",
               bbox_to_anchor=(0.5, 0.002), columnspacing=1.15, handlelength=2.2)
    fig.subplots_adjust(bottom=0.105, left=0.09, right=0.99, top=0.97)
    save(fig, "figure_s8_full_ecoregional_horizon_curves")


def figure_s2(old: pd.DataFrame, ext: pd.DataFrame) -> None:
    records = []
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.35), sharey=True,
                             gridspec_kw={"wspace": 0.15})
    for axis, split, groups in [
        (axes[0], "is_reference", [
            (False, "Non-reference", CLASSIC_BLACK, (0, (4, 2)), "s", "white"),
            (True, "Reference", CLASSIC_NAVY, "-", "o", CLASSIC_NAVY),
        ]),
        (axes[1], "quality_tier", [
            ("TIER1_HIGH_QUALITY", "Tier 1", CLASSIC_NAVY, "-", "o", CLASSIC_NAVY),
            ("TIER2_ACCEPTABLE_QUALITY", "Tier 2", CLASSIC_BLACK,
             (0, (4, 2)), "s", "white"),
        ]),
    ]:
        attrs = pd.read_csv(ATTRIBUTES_FILE, usecols=["GAGE_ID", split], dtype={"GAGE_ID": str})
        attrs["GAGE_ID"] = attrs["GAGE_ID"].str.zfill(8)
        for value, label, color, linestyle, marker, markerface in groups:
            lines = []
            for metrics in (old, ext):
                joined = metrics.merge(attrs, on="GAGE_ID", how="left")
                sub = joined[joined[split].eq(value)]
                p = paired(sub, "hydrograph_marginal_shuffle", "event_brier")
                lines.append(summarize_paired(p, "event_brier"))
            data = pd.concat([lines[0][lines[0]["lead_days"].isin([30,45,60,90])],
                              lines[1][lines[1]["lead_days"].isin([105,120])]])
            data["group"] = label
            data["split"] = split
            records.append(data)
            axis.plot(data["lead_days"], data["relative_skill_percent"],
                      color=color, lw=1.55, ls=linestyle, marker=marker, ms=3.5,
                      mfc=markerface, mec=color, mew=0.7, label=label)
        axis.axvline(97.5, color=CLASSIC_GRAY, lw=0.65, ls=":")
        axis.set_xlabel("Forecast window (days)")
        axis.legend(frameon=False)
        traditional_axes(axis)
    axes[0].set_ylabel("Occurrence Brier-score reduction (%)")
    axes[0].set_title("Reference status", loc="left")
    axes[1].set_title("Record-quality tier", loc="left")
    panel(axes[0], "a")
    panel(axes[1], "b")
    source(pd.concat(records, ignore_index=True), "figure_s2_reference_tier")
    save(fig, "figure_s2_reference_and_tier_consistency")


def figure_s3(ext: pd.DataFrame) -> None:
    records = []
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.35), sharey=True)
    line_styles = {
        "event_brier": (CLASSIC_NAVY, "-", "o", CLASSIC_NAVY),
        "timing_crps": (CLASSIC_BLACK, (0, (4, 2)), "s", "white"),
        "duration_crps": (CLASSIC_GRAY, (0, (1, 1.7)), "^", "white"),
        "deficit_crps": (CLASSIC_RED, (0, (6, 2, 1, 2)), "D", "white"),
    }
    for axis, lead in zip(axes, (105, 120)):
        for score in CORE:
            values = []
            for threshold in ("Q5", "Q10", "Q20"):
                data = summarize_paired(
                    paired(ext, "hydrograph_marginal_shuffle", score, threshold), score
                )
                row = data[data["lead_days"].eq(lead)].copy()
                row["threshold"] = threshold
                row["score"] = score
                values.append(row)
            line = pd.concat(values)
            records.append(line)
            color, linestyle, marker, markerface = line_styles[score]
            axis.plot([5, 10, 20], line["relative_skill_percent"],
                      color=color, lw=1.45, ls=linestyle, marker=marker, ms=3.8,
                      mfc=markerface, mec=color, mew=0.7,
                      label=OUTCOME_LABELS[score])
        axis.set_title(f"{lead}-day window", loc="left")
        axis.set_xlabel("Low-flow threshold percentile")
        axis.set_xticks([5, 10, 20])
        traditional_axes(axis)
    axes[0].set_ylabel("Score reduction from coherence (%)")
    axes[1].legend(frameon=False, loc="best")
    panel(axes[0], "a")
    panel(axes[1], "b")
    source(pd.concat(records, ignore_index=True), "figure_s3_low_flow_severity")
    save(fig, "figure_s3_low_flow_severity")


def figure_s4() -> None:
    data = pd.read_csv(EXT_ROOT / "sensitivities" / "sensitivity_summary.csv")
    labels = {
        "base_age3_p0.1": "Primary",
        "dry_age1": "Dry age ≥1 d", "dry_age7": "Dry age ≥7 d",
        "precip_0.0": r"P = 0.0 mm d$^{-1}$", "precip_1.0": r"P ≤ 1.0 mm d$^{-1}$",
        "members_31": "31 members", "members_51": "51 members",
        "members_151": "151 members", "seasonal_q10": "Seasonal Q10",
        "shuffle_repeat_1": "Shuffle 1", "shuffle_repeat_2": "Shuffle 2",
        "shuffle_repeat_3": "Shuffle 3", "shuffle_repeat_4": "Shuffle 4",
        "shuffle_repeat_5": "Shuffle 5",
    }
    data = data[data["variant"].isin(labels)].copy()
    data["label"] = data["variant"].map(labels)
    data["order"] = data["variant"].map({k: i for i, k in enumerate(labels)})
    data = data.sort_values("order")
    source(data, "figure_s4_sensitivity_summary")
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 4.25), sharey=True,
                             gridspec_kw={"wspace": 0.10})
    y = np.arange(len(data))
    primary = data["variant"].eq("base_age3_p0.1")
    axes[0].hlines(y, 60, 100 * data["mean_basin_fraction_improved"],
                   color=CLASSIC_LIGHT, lw=1.0)
    axes[0].scatter(100 * data["mean_basin_fraction_improved"], y,
                    s=np.where(primary, 34, 24), color=np.where(primary, CLASSIC_RED, CLASSIC_NAVY),
                    marker="o", zorder=3)
    axes[1].hlines(y, 0, data["relative_skill_min"], color=CLASSIC_LIGHT, lw=1.0)
    axes[1].scatter(data["relative_skill_min"], y, s=np.where(primary, 34, 24),
                    color=np.where(primary, CLASSIC_RED, CLASSIC_NAVY), marker="s", zorder=3)
    axes[0].set_yticks(y, data["label"])
    axes[0].set_xlim(60, 100)
    axes[0].set_xlabel("Basins improved, mean across tests (%)")
    axes[1].set_xlim(0, max(35, float(data["relative_skill_min"].max()) + 2))
    axes[1].set_xlabel("Smallest score reduction in variant (%)")
    axes[0].set_title("Consistency across basins", loc="left")
    axes[1].set_title("Weakest comparison", loc="left")
    axes[0].invert_yaxis()
    for index, axis in enumerate(axes):
        traditional_axes(axis, "x")
        panel(axis, "ab"[index])
    fig.legend(handles=[
        Line2D([], [], marker="o", ls="", color=CLASSIC_RED, label="Primary configuration"),
        Line2D([], [], marker="o", ls="", color=CLASSIC_NAVY, label="Sensitivity variant"),
    ], frameon=False, ncol=2, loc="lower center", bbox_to_anchor=(0.5, 0.005))
    fig.subplots_adjust(bottom=0.14, left=0.22, right=0.99, top=0.93)
    save(fig, "figure_s4_extension_sensitivities")


def figure_s5() -> None:
    data = pd.read_csv(EXT_ROOT / "extension_120_severity_recovery.csv")
    data = data[
        data["reference"].eq("hydrograph_marginal_shuffle")
        & data["target"].eq("recovery")
        & data["score"].isin(CORE)
    ].copy()
    source(data, "figure_s5_recovery")
    fig, axis = plt.subplots(figsize=(6.25, 2.85))
    first = data[data["lead_days"].eq(105)].set_index("score").reindex(CORE)
    second = data[data["lead_days"].eq(120)].set_index("score").reindex(CORE)
    y = np.arange(len(CORE))
    x105 = 100 * first["basin_fraction_improved"].to_numpy(float)
    x120 = 100 * second["basin_fraction_improved"].to_numpy(float)
    for ypos, left, right in zip(y, x105, x120):
        axis.plot([left, right], [ypos, ypos], color=CLASSIC_GRAY, lw=0.9)
    axis.scatter(x105, y, s=30, facecolor="white", edgecolor=CLASSIC_BLACK,
                 marker="o", linewidth=0.75, zorder=3,
                 label=f"105 days (n = {int(first['basins'].min())})")
    axis.scatter(x120, y, s=30, facecolor=CLASSIC_NAVY, edgecolor=CLASSIC_NAVY,
                 marker="s", linewidth=0.75, zorder=3,
                 label=f"120 days (n = {int(second['basins'].min())})")
    axis.set_yticks(y, [OUTCOME_LABELS[x] for x in CORE])
    axis.set_xlabel("Basins improved by coherent paths (%)")
    axis.set_xlim(55, 102)
    axis.invert_yaxis()
    handles, labels = axis.get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, ncol=2, loc="upper center",
               bbox_to_anchor=(0.5, 0.98))
    traditional_axes(axis, "x")
    fig.subplots_adjust(top=0.78, bottom=0.17, left=0.24, right=0.98)
    save(fig, "figure_s5_low_flow_recovery")


def figure_s6() -> None:
    comparisons = pd.read_csv(EXT_ROOT / "extension_120_comparisons.csv")
    data = comparisons[
        comparisons["candidate"].eq("hydrograph_analog")
        & comparisons["reference"].eq("direct_event_logistic")
        & comparisons["score"].eq("event_brier")
        & comparisons["lead_days"].isin([105, 120])
    ].copy()
    source(data, "figure_s6_direct_event_benchmark")
    fig, ax = plt.subplots(figsize=(5.4, 2.9))
    x = data["lead_days"].to_numpy(float)
    y = data["relative_skill_percent"].to_numpy(float)
    low = 100 * data["bootstrap_ci_low"].to_numpy(float) / data["reference_score"].to_numpy(float)
    high = 100 * data["bootstrap_ci_high"].to_numpy(float) / data["reference_score"].to_numpy(float)
    yerr = np.vstack([y - low, high - y])
    ax.errorbar(x, y, yerr=yerr, color=CLASSIC_NAVY, lw=1.05,
                marker="o", ms=4.2, mfc=CLASSIC_NAVY, mec=CLASSIC_NAVY,
                capsize=3, label="Estimate and 95% interval")
    ax.axhline(0, color=CLASSIC_BLACK, lw=0.75)
    ax.set(xlabel="Forecast window (days)", ylabel="Analog Brier skill vs direct logistic (%)",
           xticks=[105, 120])
    traditional_axes(ax)
    save(fig, "figure_s6_direct_occurrence_benchmark")


def figure_s1() -> None:
    failures = pd.read_csv(EXT_ROOT / "extension_120_failures.csv")
    event = pd.read_csv(EXT_ROOT / "extension_120_event_rates.csv")
    reason = failures["reason"].value_counts().rename_axis("reason").reset_index(name="basins")
    labels = {
        "ValueError('insufficient_training_initializations')": "Too few training initializations",
        "ValueError('insufficient_test_initializations')": "Too few evaluation initializations",
        "ValueError('no_scored_targets')": "No balanced target",
    }
    reason["label"] = reason["reason"].map(labels).fillna("Other")
    source(reason, "figure_s1_exclusions")
    source(event, "figure_s1_event_rates")
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.25), gridspec_kw={"wspace": 0.35})
    axes[0].barh(reason["label"], reason["basins"], color=CLASSIC_LIGHT,
                 edgecolor=CLASSIC_BLACK, linewidth=0.55)
    axes[0].set_xlabel("Excluded basins")
    axes[0].invert_yaxis()
    traditional_axes(axes[0], "x")
    axes[1].plot(event["lead_days"], 100 * event["event_rate"],
                 color=CLASSIC_NAVY, lw=1.55, marker="o", ms=3.1,
                 mfc=CLASSIC_NAVY, label="Q10 event rate")
    axes[1].set_xlabel("Forecast window (days)")
    axes[1].set_ylabel("Held-out windows with onset (%)")
    axes[1].set_xticks([1, 7, 14, 30, 45, 60, 90, 105, 120])
    traditional_axes(axes[1])
    panel(axes[0], "a")
    panel(axes[1], "b")
    save(fig, "figure_s1_sample_accounting_and_event_rates")


def manifest() -> None:
    descriptions = {
        "figure_01_study_area_and_sample": "CONUS study sample, confirmation accounting, and ecoregional coverage",
        "figure_02_representative_forecast_trajectories": "Four held-out forecasts with full trajectory uncertainty",
        "figure_03_coherence_changes_path_scores": "Absolute path scores for coherent and identical-marginal shuffled forecasts",
        "figure_04_skill_against_operational_benchmarks": "Lead-dependent skill against seasonal trajectory climatology and persistence",
        "figure_05_station_temporal_validation": "Forecast and observed Q10 occurrence through time at four representative gages",
        "figure_06_ecoregion_forecast_observed": "Basin-first forecast and observed Q10 event frequencies by ecoregion",
        "figure_07_continental_temporal_validation": "Monthly basin-first forecast probabilities and observed Q10 event frequencies",
        "figure_08_spatial_distribution_of_skill": "Basin classification and ecoregional summary of 120-day occurrence skill",
        "figure_09_spatial_calibration_bias": "Basin-level spatial calibration bias at 30- and 120-day windows",
        "figure_s1_sample_accounting_and_event_rates": "Exclusion reasons and lead-dependent event rates",
        "figure_s2_reference_and_tier_consistency": "Reference-status and record-quality sensitivity",
        "figure_s3_low_flow_severity": "Q5, Q10, and Q20 sensitivity",
        "figure_s4_extension_sensitivities": "Dry screen, ensemble size, threshold, and shuffle sensitivity",
        "figure_s5_low_flow_recovery": "Secondary low-flow recovery analysis",
        "figure_s6_direct_occurrence_benchmark": "Comparison with direct logistic occurrence forecasts",
        "figure_s7_occurrence_reliability": "Pooled held-out reliability for Q10 occurrence forecasts",
        "figure_s8_full_ecoregional_horizon_curves": "Complete 30--120-day coherence-skill curves by ecoregion and outcome",
        "figure_s9_onset_score_decomposition": "Joint and occurrence-conditioned first-onset timing skill",
        "figure_s10_leave_one_water_year_out": "Sensitivity of coherence skill to each held-out water year",
    }
    rows = []
    for stem, description in descriptions.items():
        for suffix in ("png", "pdf", "svg"):
            path = OUTPUT / f"{stem}.{suffix}"
            rows.append({
                "figure": stem, "format": suffix, "path": str(path),
                "exists": path.exists(), "bytes": path.stat().st_size if path.exists() else 0,
                "description": description,
            })
    pd.DataFrame(rows).to_csv(FIGURE_ROOT / "FIGURE_MANIFEST.csv", index=False)


def main() -> None:
    style()
    old, ext = load_results()
    figure_1(old, ext)
    figure_2()
    figure_3(old, ext)
    figure_4(old, ext)
    figure_5()
    figure_6()
    figure_7(old, ext)
    figure_8(old, ext)
    figure_9()
    figure_s1()
    figure_s2(old, ext)
    figure_s3(ext)
    figure_s4()
    figure_s5()
    figure_s6()
    figure_s7()
    figure_s8(old, ext)
    figure_s9()
    figure_s10()
    manifest()
    print(f"Revised figures are in {OUTPUT}", flush=True)


if __name__ == "__main__":
    main()
