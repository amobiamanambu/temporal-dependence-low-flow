"""Publication-style plots for the compact example workflow."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


INK = "#17232D"
NAVY = "#17365D"
BLUE = "#2B6F9E"
TEAL = "#16817A"
GOLD = "#B68124"
RUST = "#A24632"
GRAY = "#6F777C"
LIGHT = "#D9DEE1"


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "Liberation Serif", "DejaVu Serif"],
            "font.size": 9.0,
            "axes.labelsize": 10.0,
            "axes.titlesize": 10.5,
            "axes.titleweight": "bold",
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.dpi": 300,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )


def _finish(axis: plt.Axes, grid_axis: str = "y") -> None:
    axis.grid(axis=grid_axis, color=LIGHT, linewidth=0.55, alpha=0.75, zorder=0)
    axis.tick_params(direction="out", width=0.7, length=3)


def _panel(axis: plt.Axes, label: str) -> None:
    axis.text(
        -0.11,
        1.03,
        label,
        transform=axis.transAxes,
        fontsize=11.5,
        fontweight="bold",
        ha="left",
        va="bottom",
    )


def _save(fig: plt.Figure, output_directory: Path, stem: str) -> None:
    output_directory.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "svg"):
        kwargs = {"bbox_inches": "tight", "facecolor": "white", "pad_inches": 0.05}
        if extension == "png":
            kwargs["dpi"] = 300
        fig.savefig(output_directory / f"{stem}.{extension}", **kwargs)
    plt.close(fig)


def plot_sample_verification(
    horizon: pd.DataFrame,
    basin: pd.DataFrame,
    reliability: pd.DataFrame,
    output_directory: str | Path,
) -> None:
    """Plot a four-panel verification summary for the five-gage example."""
    configure_style()
    output_directory = Path(output_directory)
    leads = horizon["lead_days"].to_numpy()
    colors = {30: BLUE, 45: TEAL, 60: GOLD, 90: RUST}
    fig, axes = plt.subplots(2, 2, figsize=(9.1, 6.6))
    fig.subplots_adjust(left=0.095, right=0.98, bottom=0.10, top=0.95, wspace=0.25, hspace=0.35)

    axis = axes[0, 0]
    axis.plot(leads, horizon["observed_event_frequency"], color=INK, marker="o", lw=1.8,
              label="Observed frequency", zorder=3)
    axis.plot(leads, horizon["mean_forecast_probability"], color=RUST, marker="s", lw=1.8,
              label="Mean forecast probability", zorder=3)
    axis.set(xlabel="Forecast-window length (days)", ylabel="Q10-onset probability")
    axis.set_xticks(leads)
    axis.set_ylim(bottom=0)
    axis.legend(frameon=False, fontsize=8.2)
    _finish(axis)
    _panel(axis, "a")

    axis = axes[0, 1]
    axis.plot(leads, horizon["brier_score"], color=NAVY, marker="o", lw=1.9, zorder=3)
    axis.fill_between(leads, 0, horizon["brier_score"], color=BLUE, alpha=0.10)
    axis.set(xlabel="Forecast-window length (days)", ylabel="Brier score")
    axis.set_xticks(leads)
    axis.set_ylim(bottom=0)
    _finish(axis)
    _panel(axis, "b")

    axis = axes[1, 0]
    axis.plot([0, 1], [0, 1], color=GRAY, ls="--", lw=1.0, label="Perfect reliability")
    for lead, group in reliability.groupby("lead_days", sort=True):
        axis.plot(
            group["mean_forecast_probability"],
            group["observed_event_frequency"],
            marker="o",
            ms=4.0,
            lw=1.35,
            color=colors.get(int(lead), NAVY),
            label=f"{int(lead)} days",
        )
    axis.set(xlabel="Forecast probability", ylabel="Observed event frequency", xlim=(0, 1), ylim=(0, 1))
    axis.legend(frameon=False, fontsize=7.7, ncol=2, loc="upper left")
    _finish(axis, "both")
    _panel(axis, "c")

    axis = axes[1, 1]
    jitter = np.linspace(-1.2, 1.2, basin["GAGE_ID"].nunique())
    for index, (gage, group) in enumerate(basin.groupby("GAGE_ID", sort=True)):
        group = group.sort_values("lead_days")
        axis.plot(
            group["lead_days"] + jitter[index],
            100.0 * group["calibration_bias"],
            marker="o",
            ms=3.8,
            lw=0.9,
            alpha=0.78,
            label=gage,
        )
    axis.axhline(0, color=INK, lw=0.9)
    axis.set(xlabel="Forecast-window length (days)", ylabel="Forecast minus observed (percentage points)")
    axis.set_xticks(leads)
    axis.legend(frameon=False, fontsize=7.0, ncol=2, loc="best", title="USGS gage")
    _finish(axis)
    _panel(axis, "d")

    _save(fig, output_directory, "sample_forecast_verification")


def plot_trajectory_anatomy(sample_directory: str | Path, output_directory: str | Path) -> None:
    """Plot one 101-member forecast under three same-marginal orderings."""
    configure_style()
    sample_directory = Path(sample_directory)
    output_directory = Path(output_directory)
    members = pd.read_csv(sample_directory / "figure_03_trajectory_anatomy_members.csv.gz")
    probabilities = pd.read_csv(sample_directory / "figure_03_trajectory_anatomy_probabilities.csv")
    metadata = json.loads(
        (sample_directory / "figure_03_trajectory_anatomy_metadata.json").read_text(encoding="utf-8")
    )
    member_columns = [column for column in members.columns if column.startswith("member_")]
    selected = [member_columns[index] for index in np.linspace(0, len(member_columns) - 1, 25).astype(int)]
    configurations = [
        ("Intact state-conditioned", "Intact trajectories", NAVY),
        ("Independent daily reordering", "Independent daily reordering", RUST),
        ("Seasonal rank reconstruction", "Seasonal rank reconstruction", TEAL),
    ]
    threshold = float(probabilities["Q10"].iloc[0])
    observed = probabilities["observed_flow"].to_numpy(dtype=float)
    day = probabilities["lead_days"].to_numpy(dtype=int)

    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.8), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.075, right=0.99, bottom=0.22, top=0.84, wspace=0.12)
    for axis, (configuration, title, color), label in zip(axes, configurations, "abc"):
        subset = members.loc[members["configuration"].eq(configuration)].sort_values("lead_days")
        paths = subset[selected].to_numpy(dtype=float)
        all_paths = subset[member_columns].to_numpy(dtype=float)
        for column in range(paths.shape[1]):
            axis.plot(day, paths[:, column], color=color, alpha=0.18, lw=0.55, zorder=1)
        axis.plot(day, np.median(all_paths, axis=1), color=color, lw=1.8, label="Ensemble median", zorder=3)
        axis.plot(day, observed, color=INK, lw=1.6, label="Observed flow", zorder=4)
        axis.axhline(threshold, color=GOLD, ls="--", lw=1.2, label="Q10 threshold", zorder=2)
        occurrence = 100.0 * np.mean(np.any(all_paths <= threshold, axis=0))
        axis.set_title(title)
        axis.text(
            0.97,
            0.96,
            f"Occurrence probability = {occurrence:.1f}%",
            transform=axis.transAxes,
            ha="right",
            va="top",
            fontsize=8.0,
            color=INK,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.86, "pad": 2.0},
        )
        axis.text(-0.08, 1.04, label, transform=axis.transAxes, fontweight="bold", fontsize=11.5)
        axis.set_xlabel("Lead day")
        _finish(axis)
    axes[0].set_ylabel("Normalized daily streamflow")
    axes[0].set_ylim(bottom=0)
    legend = [
        Line2D([0], [0], color=INK, lw=1.6, label="Observed flow"),
        Line2D([0], [0], color=NAVY, lw=1.6, label="Ensemble median"),
        Line2D([0], [0], color=GOLD, lw=1.2, ls="--", label="Q10 threshold"),
        Line2D([0], [0], color=GRAY, lw=0.8, alpha=0.45, label="25 of 101 members"),
    ]
    fig.legend(handles=legend, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.02))
    fig.suptitle(
        f"USGS {metadata['GAGE_ID']} · forecast initialization {metadata['forecast_initialization_date']}",
        fontsize=10.5,
        fontweight="bold",
        y=0.97,
    )
    _save(fig, output_directory, "sample_trajectory_anatomy")
