#!/usr/bin/env python3
"""Create Figure 9: continental temporal validation.

This entry point writes the figure as PNG, PDF, and SVG.

Run from the project root:

    python figures/09_continental_temporal_validation.py
"""

from __future__ import annotations

import importlib.util
from pathlib import Path


FIGURE_ROOT = Path(__file__).resolve().parent
IMPLEMENTATION = FIGURE_ROOT / "09_continental_validation_base.py"
OUTPUT = FIGURE_ROOT / "output"
STEM = OUTPUT / "figure_09_continental_temporal_validation"


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
