#!/usr/bin/env python3
"""Regenerate the author-accepted revision of manuscript Figure 2.

The plotting and forecast calculations remain in the main publication-figure
module. This canonical entry point writes the accepted Figure 2 PNG, PDF, and
SVG without rebuilding the remaining figures or modifying the manuscript.

Run from the project root:

    python temporal_coherence_paper/accepted_figure_revisions/02_figure_02_representative_forecast_trajectories.py
"""

from __future__ import annotations

import importlib.util
from pathlib import Path


PAPER_ROOT = Path(__file__).resolve().parents[1]
IMPLEMENTATION = PAPER_ROOT / "scripts" / "32_generate_publication_figures_v2.py"
OUTPUT_STEM = "figure_02_typography_spacing_review"


def load_implementation():
    specification = importlib.util.spec_from_file_location(
        "publication_figure_implementation", IMPLEMENTATION
    )
    if specification is None or specification.loader is None:
        raise ImportError(f"Could not load plotting implementation: {IMPLEMENTATION}")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def main() -> None:
    figures = load_implementation()
    figures.style()
    figures.figure_2(output_stem=OUTPUT_STEM)


if __name__ == "__main__":
    main()
