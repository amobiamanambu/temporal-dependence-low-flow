#!/usr/bin/env python3
"""Rebuild the complete publication figure suite.

Run this script from the project root after Stages 36, 39, 41, and 42 have
completed. It first regenerates common source tables and base figures, then
runs the final presentation scripts in figure order.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PAPER_ROOT = PROJECT_ROOT / "temporal_coherence_paper"
FIGURE_ROOT = PROJECT_ROOT / "figures"


COMMANDS = [
    [PAPER_ROOT / "scripts" / "32_generate_figure_sources.py"],
    [FIGURE_ROOT / "03_04_05_08_core_figures.py"],
    [FIGURE_ROOT / "01_study_area.py", "--design", "hydrologic"],
    [FIGURE_ROOT / "02_representative_forecast_trajectories.py"],
    [FIGURE_ROOT / "04_same_marginal_basin_scores.py"],
    [FIGURE_ROOT / "06_basin_skill_trajectories.py"],
    [FIGURE_ROOT / "07_selected_basin_validation.py"],
    [FIGURE_ROOT / "08_ecoregional_calibration.py"],
    [FIGURE_ROOT / "09_continental_temporal_validation.py"],
    [FIGURE_ROOT / "10_conditional_skill_span.py"],
    [
        FIGURE_ROOT / "11_spatial_calibration_bias.py",
        "--design",
        "5",
    ],
]


def main() -> None:
    for number, command in enumerate(COMMANDS, start=1):
        script = Path(command[0])
        if not script.exists():
            raise FileNotFoundError(f"Figure script is missing: {script}")
        rendered = [sys.executable, *(str(item) for item in command)]
        print(f"[{number}/{len(COMMANDS)}] {script.relative_to(PROJECT_ROOT)}", flush=True)
        subprocess.run(rendered, cwd=PROJECT_ROOT, check=True)
    print("Figure suite rebuilt successfully.", flush=True)


if __name__ == "__main__":
    main()
