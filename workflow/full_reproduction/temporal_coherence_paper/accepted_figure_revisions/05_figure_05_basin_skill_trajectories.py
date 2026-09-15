#!/usr/bin/env python3
"""Regenerate the author-approved main-manuscript Figure 5.

Run from the project root:

    python temporal_coherence_paper/accepted_figure_revisions/05_figure_05_basin_skill_trajectories.py
"""

from __future__ import annotations

import importlib.util
from pathlib import Path


PAPER_ROOT = Path(__file__).resolve().parents[1]
IMPLEMENTATION = (
    PAPER_ROOT
    / "horizon_skill_candidate"
    / "10b_candidate_basin_skill_trajectories_classic.py"
)
OUTPUT_STEM = "figure_05_typography_spacing_review"


def load_implementation():
    specification = importlib.util.spec_from_file_location(
        "figure_05_approved_implementation", IMPLEMENTATION
    )
    if specification is None or specification.loader is None:
        raise ImportError(f"Could not load plotting implementation: {IMPLEMENTATION}")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def main() -> None:
    plot = load_implementation()
    values, summary = plot.load_sources()
    plot.draw(values, summary, output_stem=OUTPUT_STEM)
    print(f"Created approved Figure 5: {plot.OUTPUT / f'{OUTPUT_STEM}.svg'}")


if __name__ == "__main__":
    main()
