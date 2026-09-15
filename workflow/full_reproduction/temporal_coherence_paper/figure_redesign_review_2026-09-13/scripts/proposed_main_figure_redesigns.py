#!/usr/bin/env python3
"""Generate provisional data-forward redesigns for manuscript Figures 3, 4, 5, and 8.

These are review candidates only. The script does not modify either manuscript.

Run from the project root, for example:

    python temporal_coherence_paper/figure_redesign_review_2026-09-13/scripts/proposed_main_figure_redesigns.py --figure all
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve()
REVIEW_ROOT = SCRIPT.parents[1]
PAPER_ROOT = SCRIPT.parents[2]
PROJECT_ROOT = PAPER_ROOT.parent
OUTPUT = REVIEW_ROOT / "output"
SOURCES = REVIEW_ROOT / "source_tables"
CACHE = REVIEW_ROOT / ".cache"
for folder in (OUTPUT, SOURCES, CACHE):
    folder.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(CACHE / "matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

BENCHMARK_SCRIPTS = PROJECT_ROOT / "lowflow_forecast_benchmark" / "scripts"
sys.path.insert(0, str(BENCHMARK_SCRIPTS))
from lib.common import all_accepted_inventory, load_benchmark_config, load_daily  # noqa: E402
from lib.extended_120 import (  # noqa: E402
    SEASON_FEATURES,
    analog_paths_120,
    extension_cases,
)
from lib.extended_trajectory import (  # noqa: E402
    HYDROGRAPH_FEATURES,
    first_event_time,
    shuffle_members_by_day,
)


OLD_METRICS = (
    PROJECT_ROOT
    / "lowflow_forecast_benchmark"
    / "results"
    / "09_extended_forecast"
    / "extended_confirmation_metrics.csv.gz"
)
EXT_METRICS = (
    PROJECT_ROOT
    / "lowflow_forecast_benchmark"
    / "results"
    / "10_extension_120"
    / "extension_120_metrics.csv.gz"
)
REVIEW_METRICS = (
    PAPER_ROOT
    / "reviewer_strengthening"
    / "full"
    / "reviewer_strengthening_metrics.csv.gz"
)
BASIN_CALIBRATION = PAPER_ROOT / "temporal_validation" / "basin_horizon_calibration.csv.gz"
STRATIFIED_INTERVALS = (
    PAPER_ROOT
    / "reviewer_strengthening"
    / "full"
    / "archived_stratified_basin_intervals.csv"
)

LEADS = (30, 45, 60, 90, 105, 120)
SCORES = (
    ("event_brier", "Occurrence", "Brier-score difference"),
    ("timing_crps", "First-onset time", "RPS difference"),
    ("duration_crps", "Low-flow-day count", "CRPS difference, days"),
    ("deficit_crps", "Cumulative deficit", "CRPS difference"),
)
PATH_SCORES = (
    ("event_brier", "Occurrence", "Brier score"),
    ("joint_onset_rps", "First-onset time", "Ranked probability score"),
    ("duration_crps", "Low-flow-day count", "CRPS, days"),
    ("deficit_crps", "Cumulative deficit", "Normalized-deficit CRPS"),
)
ECOREGIONS = {
    "CntlPlains": "Central Plains",
    "EastHghlnds": "Eastern Highlands",
    "MxWdShld": "Mixed Wood Shield",
    "NorthEast": "Northeast",
    "SECstPlain": "Southeast Coastal Plain",
    "SEPlains": "Southeast Plains",
    "WestMnts": "Western Mountains",
    "WestPlains": "Western Plains",
    "WestXeric": "Western Xeric",
}

# Colorblind-safe, print-conscious palette.
INK = "#17232D"
NAVY = "#17365D"
BLUE = "#2B6F9E"
TEAL = "#16817A"
GREEN = "#3E7C59"
GOLD = "#B68124"
RUST = "#A24632"
CORAL = "#C85A4A"
PURPLE = "#66508F"
MIDGRAY = "#6F777C"
LIGHTGRAY = "#D9DEE1"
PALE = "#F3F5F6"
LEAD_COLORS = {
    30: "#3B7BA5",
    45: "#4F9A9A",
    60: "#5B8F59",
    90: "#B88B2A",
    105: "#C86B35",
    120: "#9D3E3E",
}


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "Liberation Serif", "DejaVu Serif"],
            "font.size": 9.0,
            "font.weight": "bold",
            "axes.titlesize": 10.0,
            "axes.titleweight": "bold",
            "axes.labelsize": 9.8,
            "axes.labelweight": "bold",
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "axes.linewidth": 0.72,
            "axes.edgecolor": INK,
            "xtick.color": INK,
            "ytick.color": INK,
            "text.color": INK,
            "axes.labelcolor": INK,
            "xtick.major.width": 0.62,
            "ytick.major.width": 0.62,
            "xtick.major.size": 3.0,
            "ytick.major.size": 3.0,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.dpi": 600,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "mathtext.fontset": "stix",
        }
    )


def finish_axis(axis: plt.Axes, grid_axis: str = "y") -> None:
    axis.set_facecolor("white")
    axis.grid(axis=grid_axis, color=LIGHTGRAY, lw=0.48, alpha=0.72, zorder=0)
    for tick in [*axis.get_xticklabels(), *axis.get_yticklabels()]:
        tick.set_fontweight("bold")


def panel(axis: plt.Axes, letter: str, x: float = -0.10, y: float = 1.025) -> None:
    axis.text(
        x,
        y,
        letter,
        transform=axis.transAxes,
        ha="left",
        va="bottom",
        fontsize=11.2,
        fontweight="bold",
        color=INK,
        clip_on=False,
    )


def save(fig: plt.Figure, stem: str) -> None:
    for suffix in ("png", "svg", "pdf"):
        kwargs = {"bbox_inches": "tight", "facecolor": "white", "pad_inches": 0.04}
        if suffix == "png":
            kwargs["dpi"] = 600
        fig.savefig(OUTPUT / f"{stem}.{suffix}", **kwargs)
    plt.close(fig)
    print(f"Wrote {OUTPUT / (stem + '.png')}", flush=True)


def normalize_gage(value: object) -> str:
    return str(value).replace(".0", "").zfill(8)


def seasonal_rank_reconstruction(
    marginal_paths: np.ndarray, seasonal_paths: np.ndarray
) -> np.ndarray:
    """Assign each day's sorted marginal values to season-only trajectory ranks."""
    sorted_values = np.sort(marginal_paths, axis=1)
    seasonal_order = np.argsort(seasonal_paths, axis=1, kind="stable")
    reconstructed = np.empty_like(marginal_paths)
    np.put_along_axis(reconstructed, seasonal_order, sorted_values, axis=1)
    return reconstructed


def anatomy_case(gage: str = "01013500") -> dict[str, object]:
    """Recreate one preselected held-out forecast and all three same-marginal paths."""
    config = load_benchmark_config()
    inventory = all_accepted_inventory().copy()
    inventory["GAGE_ID"] = inventory["GAGE_ID"].map(normalize_gage)
    row = inventory.set_index("GAGE_ID").loc[gage]
    daily = load_daily(row["file"])
    fit = daily.loc[
        daily["date"].le(pd.Timestamp(config["fit_end"])) & daily["q_mm_day"].gt(0),
        "q_mm_day",
    ]
    scale = float(fit.median())
    threshold = float(
        (
            daily.loc[
                daily["date"].le(pd.Timestamp(config["threshold_reference_end"])),
                "q_mm_day",
            ]
            / scale
        ).quantile(0.10)
    )
    cases = extension_cases(daily, scale, config)
    train = cases[
        cases["future_date"].le(pd.Timestamp(config["calibration_end"]))
    ].copy()
    test = cases[
        cases["date"].ge(pd.Timestamp(config["evaluation_start"]))
        & cases["future_date"].le(pd.Timestamp(config["evaluation_end"]))
        & cases["q"].gt(threshold)
    ].copy().reset_index(drop=True)
    truth_all = test[[f"q_h{day}" for day in range(1, 121)]].to_numpy(float)
    observed_onset = first_event_time(truth_all, threshold, "onset")
    eligible = np.flatnonzero((observed_onset >= 20) & (observed_onset <= 100))
    if len(eligible) == 0:
        eligible = np.flatnonzero(observed_onset <= 120)
    choice = int(eligible[len(eligible) // 2]) if len(eligible) else int(len(test) // 2)
    selected = test.iloc[[choice]]
    intact = analog_paths_120(
        train, selected, HYDROGRAPH_FEATURES, scale_by_initial_q=True, members=101
    )[0]
    seasonal_paths = analog_paths_120(
        train, selected, SEASON_FEATURES, scale_by_initial_q=False, members=101
    )[0]
    reconstructed = seasonal_rank_reconstruction(
        intact[None, :, :], seasonal_paths[None, :, :]
    )[0]
    seed = int(config["random_seed"] + int(gage[-5:]) + 24000)
    shuffled = shuffle_members_by_day(intact[None, :, :], seed)[0]
    issue_date = pd.Timestamp(selected.iloc[0]["date"])
    return {
        "gage": gage,
        "issue_date": issue_date,
        "threshold": threshold,
        "truth": truth_all[choice],
        "intact": intact,
        "reconstructed": reconstructed,
        "shuffled": shuffled,
    }


def cumulative_crossing_probability(paths: np.ndarray, threshold: float) -> np.ndarray:
    crossing = paths <= threshold
    return np.maximum.accumulate(crossing, axis=1).mean(axis=0)


def figure_3_anatomy() -> None:
    """New Figure 3: show what cross-day reordering does to actual member paths."""
    case = anatomy_case()
    days = np.arange(1, 121)
    threshold = float(case["threshold"])
    truth = np.asarray(case["truth"], dtype=float)
    intact = np.asarray(case["intact"], dtype=float)
    reconstructed = np.asarray(case["reconstructed"], dtype=float)
    shuffled = np.asarray(case["shuffled"], dtype=float)

    # Select the same member labels in every panel, spread across intact onset times.
    intact_onsets = first_event_time(intact[None, :, :], threshold, "onset")[0]
    ordered = np.argsort(intact_onsets, kind="stable")
    selected_members = ordered[np.linspace(0, len(ordered) - 1, 25).astype(int)]

    path_source = []
    for label, paths in (
        ("Intact state-conditioned", intact),
        ("Independent daily reordering", shuffled),
        ("Seasonal rank reconstruction", reconstructed),
    ):
        frame = pd.DataFrame(paths.T, columns=[f"member_{i + 1:03d}" for i in range(101)])
        frame.insert(0, "lead_days", days)
        frame.insert(0, "configuration", label)
        path_source.append(frame)
    pd.concat(path_source, ignore_index=True).to_csv(
        SOURCES / "figure_03_trajectory_anatomy_members.csv.gz", index=False
    )

    probability_source = pd.DataFrame(
        {
            "lead_days": days,
            "intact_probability": cumulative_crossing_probability(intact, threshold),
            "seasonal_reconstruction_probability": cumulative_crossing_probability(
                reconstructed, threshold
            ),
            "independent_reordering_probability": cumulative_crossing_probability(
                shuffled, threshold
            ),
            "observed_flow": truth,
            "Q10": threshold,
        }
    )
    probability_source.to_csv(
        SOURCES / "figure_03_trajectory_anatomy_probabilities.csv", index=False
    )
    observed_onset = int(first_event_time(truth[None, :], threshold, "onset")[0])
    metadata = {
        "GAGE_ID": case["gage"],
        "forecast_initialization_date": str(pd.Timestamp(case["issue_date"]).date()),
        "observed_onset_day": observed_onset,
        "Q10_normalized_flow": threshold,
        "selected_member_count": int(len(selected_members)),
        "selection_rule": "preselected Figure 2 basin; median eligible held-out Q10-onset case",
    }
    (SOURCES / "figure_03_trajectory_anatomy_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )

    fig = plt.figure(figsize=(7.55, 5.55))
    grid_spec = fig.add_gridspec(
        2,
        3,
        height_ratios=[1.32, 0.90],
        hspace=0.36,
        wspace=0.16,
        left=0.085,
        right=0.992,
        top=0.955,
        bottom=0.115,
    )
    axes = [fig.add_subplot(grid_spec[0, index]) for index in range(3)]
    probability_axis = fig.add_subplot(grid_spec[1, :])
    configurations = (
        (intact, "Intact state-conditioned paths", NAVY),
        (shuffled, "Independent daily reordering", CORAL),
        (reconstructed, "Seasonal rank reconstruction", GREEN),
    )
    positive = np.concatenate(
        [truth[truth > 0], intact[intact > 0], reconstructed[reconstructed > 0], shuffled[shuffled > 0]]
    )
    lower = max(float(np.quantile(positive, 0.002)) * 0.70, 1e-5)
    upper = float(np.quantile(positive, 0.998)) * 1.35
    for index, (axis, (paths, title, color)) in enumerate(zip(axes, configurations)):
        axis.plot(
            days,
            np.maximum(paths[selected_members].T, lower),
            color=color,
            lw=0.48,
            alpha=0.24,
            rasterized=True,
            zorder=1,
        )
        axis.plot(
            days,
            np.maximum(np.median(paths, axis=0), lower),
            color=color,
            lw=1.75,
            label="Ensemble median",
            zorder=4,
        )
        axis.plot(
            days,
            np.maximum(truth, lower),
            color=INK,
            lw=1.25,
            label="Observed USGS path",
            zorder=5,
        )
        axis.axhline(
            threshold,
            color=RUST,
            lw=1.0,
            ls=(0, (4, 2)),
            label="Basin Q10",
            zorder=3,
        )
        axis.set_yscale("log")
        axis.set_xlim(1, 120)
        axis.set_ylim(lower, upper)
        axis.set_xticks([1, 30, 60, 90, 120])
        axis.set_title(title, loc="left", pad=5, fontsize=9.4)
        axis.set_xlabel("Lead time (days)")
        if index == 0:
            axis.set_ylabel("Normalized daily flow")
        else:
            axis.tick_params(labelleft=False)
        finish_axis(axis)
        panel(axis, "abc"[index], -0.13 if index == 0 else -0.08)

    probabilities = {
        "Intact state-conditioned paths": probability_source["intact_probability"],
        "Seasonal rank reconstruction": probability_source[
            "seasonal_reconstruction_probability"
        ],
        "Independent daily reordering": probability_source[
            "independent_reordering_probability"
        ],
    }
    line_styles = {
        "Intact state-conditioned paths": (NAVY, "-", "o"),
        "Seasonal rank reconstruction": (GREEN, (0, (6, 2, 1, 2)), "D"),
        "Independent daily reordering": (CORAL, (0, (4, 2)), "s"),
    }
    for label, values in probabilities.items():
        color, line_style, marker = line_styles[label]
        probability_axis.plot(
            days,
            values,
            color=color,
            lw=1.75,
            ls=line_style,
            marker=marker,
            markevery=[29, 59, 89, 119],
            ms=3.5,
            mfc="white" if label != "Intact state-conditioned paths" else color,
            mec=color,
            mew=0.8,
            label=label,
            zorder=4,
        )
        probability_axis.text(
            121.0,
            float(values.iloc[-1]),
            f"{100 * float(values.iloc[-1]):.0f}%",
            color=color,
            ha="left",
            va="center",
            fontsize=8.0,
            fontweight="bold",
            clip_on=False,
        )
    if observed_onset <= 120:
        probability_axis.axvline(observed_onset, color=INK, lw=0.9, ls=":", zorder=2)
        probability_axis.text(
            observed_onset + 1.5,
            0.05,
            f"Observed onset: day {observed_onset}",
            rotation=90,
            ha="left",
            va="bottom",
            fontsize=7.5,
            color=INK,
        )
    probability_axis.set_xlim(1, 127)
    probability_axis.set_ylim(0, 1.02)
    probability_axis.set_xticks([1, 30, 60, 90, 120])
    probability_axis.set_yticks([0, 0.25, 0.50, 0.75, 1.0])
    probability_axis.set_xlabel("Forecast window (days)")
    probability_axis.set_ylabel("Probability of at least one Q10 crossing")
    finish_axis(probability_axis)
    panel(probability_axis, "d", -0.045)
    probability_axis.text(
        0.99,
        0.06,
        "Daily ensemble values are identical in all three configurations",
        transform=probability_axis.transAxes,
        ha="right",
        va="bottom",
        fontsize=7.6,
        color=MIDGRAY,
    )
    probability_axis.text(
        0.01,
        0.96,
        f"USGS {case['gage']}  |  forecast initialization {pd.Timestamp(case['issue_date']):%d %b %Y}",
        transform=probability_axis.transAxes,
        ha="left",
        va="top",
        fontsize=7.6,
        fontweight="bold",
        color=MIDGRAY,
    )
    handles, labels = probability_axis.get_legend_handles_labels()
    legend = fig.legend(
        handles,
        labels,
        frameon=False,
        ncol=3,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.005),
        fontsize=8.0,
    )
    for label in legend.get_texts():
        label.set_fontweight("bold")
    save(fig, "proposed_figure_03_trajectory_anatomy")


def same_marginal_basin_scores() -> pd.DataFrame:
    """Return matched 120-day basin scores for the three ordering experiments."""
    columns = [
        "aggregation",
        "GAGE_ID",
        "spatial_group",
        "lead_days",
        "model",
        "score",
        "n",
        "score_value",
    ]
    data = pd.read_csv(REVIEW_METRICS, usecols=columns, dtype={"GAGE_ID": str})
    data = data[
        data["aggregation"].eq("basin")
        & data["lead_days"].eq(120)
        & data["model"].isin(
            (
                "hydrograph_analog",
                "seasonal_rank_reconstruction",
                "independent_daily_shuffle",
            )
        )
        & data["score"].isin([item[0] for item in PATH_SCORES])
    ].copy()
    data["GAGE_ID"] = data["GAGE_ID"].map(normalize_gage)
    wide = data.pivot_table(
        index=["GAGE_ID", "spatial_group", "score"],
        columns="model",
        values="score_value",
        aggfunc="first",
    ).reset_index()
    required = [
        "hydrograph_analog",
        "seasonal_rank_reconstruction",
        "independent_daily_shuffle",
    ]
    return wide.dropna(subset=required).reset_index(drop=True)


def figure_4_same_marginal_basin_pairs() -> None:
    """Redesign old Figure 3 with the matched basin observations made visible."""
    data = same_marginal_basin_scores()
    data.to_csv(SOURCES / "figure_04_same_marginal_basin_pairs.csv.gz", index=False)

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(7.55, 5.85),
        gridspec_kw={"hspace": 0.24, "wspace": 0.16},
    )
    for index, (axis, (score, title, units)) in enumerate(zip(axes.flat, PATH_SCORES)):
        part = data[data["score"].eq(score)]
        shuffled = part["independent_daily_shuffle"].to_numpy(float)
        intact = part["hydrograph_analog"].to_numpy(float)
        reconstructed = part["seasonal_rank_reconstruction"].to_numpy(float)
        maximum = float(np.nanmax(np.concatenate([shuffled, intact, reconstructed])))
        limit = maximum * 1.035

        # Draw reconstruction first, then the state-conditioned points so their
        # close agreement remains visible without concealing individual basins.
        axis.scatter(
            shuffled,
            reconstructed,
            s=6.0,
            facecolor=GREEN,
            edgecolor="none",
            alpha=0.115,
            rasterized=True,
            zorder=2,
        )
        axis.scatter(
            shuffled,
            intact,
            s=3.8,
            facecolor=NAVY,
            edgecolor="none",
            alpha=0.235,
            rasterized=True,
            zorder=3,
        )
        axis.plot(
            [0, limit],
            [0, limit],
            color=INK,
            lw=1.0,
            ls=(0, (5, 2.5)),
            zorder=4,
        )
        axis.set_xlim(0, limit)
        axis.set_ylim(0, limit)
        axis.set_aspect("equal", adjustable="box")
        axis.set_title(f"{title} ({units})", loc="left", pad=4)
        finish_axis(axis, grid_axis="both")
        panel(axis, "abcd"[index], -0.13)
        axis.text(
            0.97,
            0.055,
            f"{len(part):,} matched basins",
            transform=axis.transAxes,
            ha="right",
            va="bottom",
            fontsize=7.3,
            color=MIDGRAY,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.80, "pad": 0.7},
        )

    handles = [
        Line2D([], [], marker="o", ls="", color=NAVY, ms=5.0,
               label="State-conditioned coherent paths"),
        Line2D([], [], marker="o", ls="", color=GREEN, ms=5.4,
               label="Seasonal rank reconstruction"),
        Line2D([], [], color=INK, lw=1.0, ls=(0, (5, 2.5)),
               label="Equal score to independent reordering"),
    ]
    legend = fig.legend(
        handles=handles,
        frameon=False,
        ncol=3,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.008),
        fontsize=7.7,
        columnspacing=1.25,
        handletextpad=0.4,
    )
    for label in legend.get_texts():
        label.set_fontweight("bold")
    fig.text(
        0.515,
        0.090,
        "Score after independent daily reordering",
        ha="center",
        va="center",
        fontsize=9.8,
        fontweight="bold",
    )
    fig.text(
        0.018,
        0.545,
        "Score after temporal order restoration",
        rotation=90,
        ha="center",
        va="center",
        fontsize=9.8,
        fontweight="bold",
    )
    fig.subplots_adjust(left=0.085, right=0.995, top=0.965, bottom=0.145)
    save(fig, "proposed_figure_04_same_marginal_basin_pairs")


def paired_benchmark_values() -> pd.DataFrame:
    old = pd.read_csv(OLD_METRICS, dtype={"GAGE_ID": str})
    extension = pd.read_csv(EXT_METRICS, dtype={"GAGE_ID": str})
    rows = []
    keys = ["GAGE_ID", "spatial_group", "threshold_name", "lead_days", "target"]
    for score, _, _ in SCORES:
        for reference, reference_label in (
            ("seasonal_climatology_path", "Seasonal trajectory climatology"),
            ("constant_persistence_path", "Constant persistence"),
        ):
            for metrics, accepted_leads in (
                (old, (30, 45, 60, 90)),
                (extension, (105, 120)),
            ):
                candidate = metrics[
                    metrics["model"].eq("hydrograph_analog")
                    & metrics["threshold_name"].eq("Q10")
                    & metrics["target"].eq("onset")
                    & metrics["lead_days"].isin(accepted_leads)
                ][keys + ["n", score]].dropna(subset=[score])
                benchmark = metrics[
                    metrics["model"].eq(reference)
                    & metrics["threshold_name"].eq("Q10")
                    & metrics["target"].eq("onset")
                    & metrics["lead_days"].isin(accepted_leads)
                ][keys + [score]].dropna(subset=[score])
                paired = candidate.merge(
                    benchmark, on=keys, suffixes=("_analog", "_benchmark")
                )
                paired["score"] = score
                paired["reference"] = reference_label
                paired["score_difference"] = (
                    paired[f"{score}_benchmark"] - paired[f"{score}_analog"]
                )
                paired["relative_skill_percent"] = np.where(
                    paired[f"{score}_benchmark"].gt(1e-12),
                    100.0
                    * paired["score_difference"]
                    / paired[f"{score}_benchmark"],
                    np.nan,
                )
                paired["GAGE_ID"] = paired["GAGE_ID"].map(normalize_gage)
                rows.append(
                    paired[
                        keys
                        + [
                            "n",
                            "score",
                            "reference",
                            f"{score}_analog",
                            f"{score}_benchmark",
                            "score_difference",
                            "relative_skill_percent",
                        ]
                    ].rename(
                        columns={
                            f"{score}_analog": "analog_score",
                            f"{score}_benchmark": "benchmark_score",
                        }
                    )
                )
    return pd.concat(rows, ignore_index=True)


def figure_5_empirical_benchmark_differences() -> None:
    """Hybrid redesign of current Figure 4: summary curves over raw basin skill."""
    data = paired_benchmark_values()
    data.to_csv(SOURCES / "figure_05_empirical_benchmark_differences.csv.gz", index=False)
    intervals = pd.read_csv(STRATIFIED_INTERVALS)
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(7.55, 5.55),
        sharex=True,
        sharey=True,
        gridspec_kw={"hspace": 0.25, "wspace": 0.15},
    )
    rng = np.random.default_rng(20260913)
    styles = {
        "Seasonal trajectory climatology": {
            "code": "seasonal_climatology_path",
            "color": NAVY,
            "offset": -1.25,
            "marker": "o",
            "linestyle": "-",
            "markerface": NAVY,
        },
        "Constant persistence": {
            "code": "constant_persistence_path",
            "color": RUST,
            "offset": 1.25,
            "marker": "s",
            "linestyle": (0, (4, 2)),
            "markerface": "white",
        },
    }
    lower, upper = -100.0, 100.0
    for index, (axis, (score, title, _)) in enumerate(zip(axes.flat, SCORES)):
        part = data[data["score"].eq(score)]
        for reference, style_values in styles.items():
            reference_data = part[part["reference"].eq(reference)]
            summary_rows = []
            for lead in LEADS:
                lead_data = reference_data[reference_data["lead_days"].eq(lead)]
                raw = lead_data["relative_skill_percent"].to_numpy(float)
                raw = raw[np.isfinite(raw)]
                visible = raw[(raw >= lower) & (raw <= upper)]
                jitter = rng.normal(0, 0.62, size=len(visible))
                axis.scatter(
                    lead + style_values["offset"] + jitter,
                    visible,
                    s=1.70,
                    color=style_values["color"],
                    alpha=0.055,
                    linewidths=0,
                    rasterized=True,
                    zorder=2,
                )
                weights = lead_data["n"].to_numpy(float)
                analog_score = float(np.average(lead_data["analog_score"], weights=weights))
                benchmark_score = float(
                    np.average(lead_data["benchmark_score"], weights=weights)
                )
                summary_rows.append(
                    {
                        "lead_days": lead,
                        "relative_skill_percent": 100.0
                        * (benchmark_score - analog_score)
                        / benchmark_score,
                    }
                )
            summary = pd.DataFrame(summary_rows)
            interval = intervals[
                intervals["candidate"].eq("hydrograph_analog")
                & intervals["reference"].eq(style_values["code"])
                & intervals["score"].eq(score)
            ][["lead_days", "stratified_ci_low", "stratified_ci_high"]]
            summary = summary.merge(interval, on="lead_days", how="left")
            x = summary["lead_days"].to_numpy(float)
            y = summary["relative_skill_percent"].to_numpy(float)
            axis.fill_between(
                x,
                summary["stratified_ci_low"].to_numpy(float),
                summary["stratified_ci_high"].to_numpy(float),
                color=style_values["color"],
                alpha=0.14 if reference == "Seasonal trajectory climatology" else 0.10,
                linewidth=0,
                zorder=4,
            )
            axis.plot(
                x,
                y,
                color=style_values["color"],
                lw=1.8,
                ls=style_values["linestyle"],
                marker=style_values["marker"],
                ms=4.0,
                mfc=style_values["markerface"],
                mec=style_values["color"],
                mew=0.85,
                zorder=7,
            )
        axis.axhline(0, color=INK, lw=0.9, zorder=5)
        axis.axvspan(97.5, 123, color=PALE, alpha=0.82, lw=0, zorder=0)
        axis.axvline(97.5, color=MIDGRAY, lw=0.7, ls=":", zorder=3)
        axis.set_title(title, loc="left", pad=4)
        axis.set_ylabel("Relative score reduction (%)")
        axis.set_xlim(26, 124)
        axis.set_ylim(lower, upper)
        axis.set_xticks(LEADS)
        axis.set_yticks([-100, -75, -50, -25, 0, 25, 50, 75, 100])
        finish_axis(axis)
        panel(axis, "abcd"[index])
        basin_counts = part.groupby("lead_days")["GAGE_ID"].nunique()
        axis.text(
            0.98,
            0.04,
            f"{int(basin_counts.min()):,}–{int(basin_counts.max()):,} basins/window",
            transform=axis.transAxes,
            ha="right",
            va="bottom",
            fontsize=7.2,
            color=MIDGRAY,
        )
    for axis in axes[-1, :]:
        axis.set_xlabel("Forecast window (days)")
    handles = [
        Line2D([], [], marker="o", ls="-", mfc=NAVY, mec=NAVY, color=NAVY,
               label="Seasonal trajectory climatology"),
        Line2D([], [], marker="s", ls=(0, (4, 2)), mfc="white", mec=RUST,
               color=RUST, label="Constant persistence"),
        Line2D([], [], marker=".", ls="", color=MIDGRAY, ms=6,
               label="Individual basin values"),
    ]
    legend = fig.legend(
        handles=handles,
        frameon=False,
        ncol=3,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.004),
        fontsize=7.8,
    )
    for label in legend.get_texts():
        label.set_fontweight("bold")
    fig.subplots_adjust(left=0.10, right=0.995, top=0.975, bottom=0.115)
    save(fig, "proposed_figure_05_empirical_benchmark_differences")


def figure_8_ecoregion_calibration_clouds() -> None:
    """Redesign of current Figure 7: plot basin-level forecast–observation pairs."""
    data = pd.read_csv(BASIN_CALIBRATION, dtype={"GAGE_ID": str})
    data["GAGE_ID"] = data["GAGE_ID"].map(normalize_gage)
    data = data[data["lead_days"].isin(LEADS)].copy()
    data["ecoregion"] = data["spatial_group"].map(ECOREGIONS)
    data.to_csv(SOURCES / "figure_08_ecoregion_calibration_clouds.csv.gz", index=False)

    regions = sorted(data["spatial_group"].unique(), key=lambda code: ECOREGIONS[code])
    fig, axes = plt.subplots(
        3,
        3,
        figsize=(7.55, 6.45),
        sharex=True,
        sharey=True,
        gridspec_kw={"hspace": 0.26, "wspace": 0.13},
    )
    for index, (axis, region) in enumerate(zip(axes.flat, regions)):
        part = data[data["spatial_group"].eq(region)]
        axis.plot([0, 1], [0, 1], color=INK, lw=0.85, ls=(0, (4, 2)), zorder=1)
        for lead in LEADS:
            lead_data = part[part["lead_days"].eq(lead)]
            axis.scatter(
                lead_data["observed_event_frequency"],
                lead_data["mean_forecast_probability"],
                s=2.55,
                color=LEAD_COLORS[lead],
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
                facecolor=LEAD_COLORS[lead],
                edgecolor=INK,
                linewidth=0.72,
                zorder=5,
            )
        axis.set_title(ECOREGIONS[region], loc="left", pad=3)
        axis.text(
            0.97,
            0.05,
            f"{part['GAGE_ID'].nunique():,} basins",
            transform=axis.transAxes,
            ha="right",
            va="bottom",
            fontsize=7.0,
            color=MIDGRAY,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 0.6},
        )
        axis.set_xlim(0, 1)
        axis.set_ylim(0, 1)
        axis.set_xticks([0, 0.25, 0.50, 0.75, 1.0])
        axis.set_yticks([0, 0.25, 0.50, 0.75, 1.0])
        axis.set_aspect("equal", adjustable="box")
        finish_axis(axis, grid_axis="both")
        panel(axis, chr(97 + index), -0.12)
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
        0.018,
        0.54,
        "Forecast Q10 event probability",
        rotation=90,
        ha="center",
        va="center",
        fontsize=9.8,
        fontweight="bold",
    )
    handles = [
        Line2D([], [], marker="o", ls="", color=LEAD_COLORS[lead], ms=4.8,
               label=f"{lead} days")
        for lead in LEADS
    ]
    handles.extend(
        [
            Line2D([], [], marker="D", ls="", mfc=MIDGRAY, mec=INK, color=MIDGRAY,
                   ms=5.2, label="Ecoregion mean"),
            Line2D([], [], color=INK, lw=0.85, ls=(0, (4, 2)), label="Perfect calibration"),
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
    save(fig, "proposed_figure_08_ecoregion_calibration_clouds")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--figure",
        choices=("3", "4", "5", "8", "all"),
        default="all",
        help="Proposed manuscript figure number to generate.",
    )
    return parser.parse_args()


def main() -> None:
    configure_style()
    requested = parse_args().figure
    if requested in ("3", "all"):
        figure_3_anatomy()
    if requested in ("4", "all"):
        figure_4_same_marginal_basin_pairs()
    if requested in ("5", "all"):
        figure_5_empirical_benchmark_differences()
    if requested in ("8", "all"):
        figure_8_ecoregion_calibration_clouds()


if __name__ == "__main__":
    main()
