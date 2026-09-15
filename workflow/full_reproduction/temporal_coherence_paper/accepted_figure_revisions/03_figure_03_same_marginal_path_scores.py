#!/usr/bin/env python3
"""Regenerate the author-accepted revision of manuscript Figure 3.

The main publication-figure module supplies the archived-data reading and
plotting implementation. This entry point writes only the accepted Figure 3
PNG, PDF, and SVG and does not modify the manuscript.

Run from the project root:

    python temporal_coherence_paper/accepted_figure_revisions/03_figure_03_same_marginal_path_scores.py
"""

from __future__ import annotations

import importlib.util
from pathlib import Path


PAPER_ROOT = Path(__file__).resolve().parents[1]
IMPLEMENTATION = PAPER_ROOT / "scripts" / "32_generate_publication_figures_v2.py"
OUTPUT_STEM = "figure_03_typography_spacing_review"


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
    old_results, extension_results = figures.load_results()
    figures.figure_3(
        old_results,
        extension_results,
        output_stem=OUTPUT_STEM,
    )


if __name__ == "__main__":
    main()
