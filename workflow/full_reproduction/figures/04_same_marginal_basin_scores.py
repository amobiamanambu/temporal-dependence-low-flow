#!/usr/bin/env python3
"""Create Figure 4: basin scores under identical daily marginals.

The scientific data, limits, marks, typography, colors, legend, and labels are
identical to the core figure. Only the horizontal placement of the two subplot
columns changes.

Run from the project root:

    python figures/04_same_marginal_basin_scores.py
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
        "figure4_accepted_implementation", IMPLEMENTATION
    )
    if specification is None or specification.loader is None:
        raise ImportError(f"Could not load plotting implementation: {IMPLEMENTATION}")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def make_option(
    figures,
    *,
    stem: str,
    wspace: float,
    anchor_fraction: float,
    y_label_x: float = 0.018,
    y_label_size: float = 9.8,
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    import numpy as np

    data = figures.same_marginal_basin_scores()
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(7.55, 5.85),
        gridspec_kw={"hspace": 0.24, "wspace": wspace},
    )

    for index, (axis, (score, title, units)) in enumerate(
        zip(axes.flat, figures.PATH_SCORES)
    ):
        part = data[data["score"].eq(score)]
        shuffled = part["independent_daily_shuffle"].to_numpy(float)
        intact = part["hydrograph_analog"].to_numpy(float)
        reconstructed = part["seasonal_rank_reconstruction"].to_numpy(float)
        maximum = float(np.nanmax(np.concatenate([shuffled, intact, reconstructed])))
        limit = maximum * 1.035

        axis.scatter(
            shuffled,
            reconstructed,
            s=6.0,
            facecolor=figures.GREEN,
            edgecolor="none",
            alpha=0.115,
            rasterized=True,
            zorder=2,
        )
        axis.scatter(
            shuffled,
            intact,
            s=3.8,
            facecolor=figures.NAVY,
            edgecolor="none",
            alpha=0.235,
            rasterized=True,
            zorder=3,
        )
        axis.plot(
            [0, limit],
            [0, limit],
            color=figures.INK,
            lw=1.0,
            ls=(0, (5, 2.5)),
            zorder=4,
        )
        axis.set_xlim(0, limit)
        axis.set_ylim(0, limit)
        axis.set_aspect("equal", adjustable="box")

        # Equal-aspect panels do not fill their full GridSpec cells. Move the
        # left column rightward and the right column leftward so the unused
        # interior space closes while the panels remain square.
        if index % 2 == 0:
            axis.set_anchor((anchor_fraction, 0.5))
        else:
            axis.set_anchor((1.0 - anchor_fraction, 0.5))

        axis.set_title(f"{title} ({units})", loc="left", pad=4)
        figures.finish_axis(axis, grid_axis="both")
        figures.panel(axis, "abcd"[index], -0.13)
        axis.text(
            0.97,
            0.055,
            f"{len(part):,} matched basins",
            transform=axis.transAxes,
            ha="right",
            va="bottom",
            fontsize=7.3,
            color=figures.MIDGRAY,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.80, "pad": 0.7},
        )

    handles = [
        Line2D(
            [],
            [],
            marker="o",
            ls="",
            color=figures.NAVY,
            ms=5.0,
            label="State-conditioned coherent paths",
        ),
        Line2D(
            [],
            [],
            marker="o",
            ls="",
            color=figures.GREEN,
            ms=5.4,
            label="Seasonal rank reconstruction",
        ),
        Line2D(
            [],
            [],
            color=figures.INK,
            lw=1.0,
            ls=(0, (5, 2.5)),
            label="Equal score to independent reordering",
        ),
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
        y_label_x,
        0.545,
        "Score after temporal order restoration",
        rotation=90,
        ha="center",
        va="center",
        fontsize=y_label_size,
        fontweight="bold",
    )
    fig.subplots_adjust(left=0.085, right=0.995, top=0.965, bottom=0.145)

    original_output = figures.OUTPUT
    try:
        figures.OUTPUT = OUTPUT
        figures.save(fig, stem)
    finally:
        figures.OUTPUT = original_output
        plt.close(fig)


def main() -> None:
    figures = load_implementation()
    figures.configure_style()
    make_option(
        figures,
        stem="figure_04_same_marginal_basin_scores",
        wspace=0.025,
        anchor_fraction=0.855,
        y_label_x=0.165,
        y_label_size=10.6,
    )
    print(f"Figure 4 written to {OUTPUT}")


if __name__ == "__main__":
    main()
