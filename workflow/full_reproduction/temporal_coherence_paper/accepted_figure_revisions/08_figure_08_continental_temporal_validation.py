#!/usr/bin/env python3
"""Regenerate the author-accepted revision of manuscript Figure 8.

The plotting implementation remains in its historical development location so
earlier outputs stay reproducible. This canonical entry point writes the
accepted Figure 8 PNG, PDF, and SVG without modifying the manuscript.

Run from the project root:

    python temporal_coherence_paper/accepted_figure_revisions/08_figure_08_continental_temporal_validation.py
"""

from __future__ import annotations

import importlib.util
from pathlib import Path


PAPER_ROOT = Path(__file__).resolve().parents[1]
IMPLEMENTATION = (
    PAPER_ROOT / "figure7_options" /
    "07_endgame_b_figure_07_nature_signal_error_strips.py"
)
OUTPUT = PAPER_ROOT / "figure7_options" / "output"
STEM = OUTPUT / "figure_08_annotation_spacing_review"


def load_implementation():
    specification = importlib.util.spec_from_file_location(
        "figure_08_temporal_validation_implementation", IMPLEMENTATION
    )
    if specification is None or specification.loader is None:
        raise ImportError(f"Could not load plotting implementation: {IMPLEMENTATION}")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def main() -> None:
    plot = load_implementation()
    plot.apply_style()
    data = plot.load_data()
    figure = plot.build(data)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    figure.savefig(STEM.with_suffix(".png"), dpi=600, bbox_inches="tight", pad_inches=0.04)
    figure.savefig(STEM.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.04)
    figure.savefig(STEM.with_suffix(".svg"), bbox_inches="tight", pad_inches=0.04)
    plot.plt.close(figure)
    print(f"Wrote {STEM}.png/.pdf/.svg")


if __name__ == "__main__":
    main()
