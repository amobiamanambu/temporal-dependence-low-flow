#!/usr/bin/env python3
"""Create Figure 2: representative observed and forecast trajectories.

The plotting and forecast calculations remain in the main publication-figure
module. This entry point writes Figure 2 as PNG, PDF, and SVG.

Run from the project root:

    python figures/02_representative_forecast_trajectories.py
"""

from __future__ import annotations

import importlib.util
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PAPER_ROOT = PROJECT_ROOT / "temporal_coherence_paper"
IMPLEMENTATION = PAPER_ROOT / "scripts" / "32_generate_figure_sources.py"
OUTPUT_STEM = "figure_02_representative_forecast_trajectories"


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
