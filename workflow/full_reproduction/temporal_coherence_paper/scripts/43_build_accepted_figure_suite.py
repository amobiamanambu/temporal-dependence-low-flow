#!/usr/bin/env python3
"""Rebuild the author-approved main and Supporting Information figure suite.

Run this script from the project root after Stages 36, 39, 41, and 42 have
completed.  It first regenerates the common source tables and base figures,
then runs only the accepted presentation variants in manuscript order.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PAPER_ROOT = PROJECT_ROOT / "temporal_coherence_paper"


COMMANDS = [
    [PAPER_ROOT / "scripts" / "32_generate_publication_figures_v2.py"],
    [
        PAPER_ROOT
        / "figure_redesign_review_2026-09-13"
        / "scripts"
        / "proposed_main_figure_redesigns.py"
    ],
    [PAPER_ROOT / "figure1_options" / "01b_figure_01_nature_study_area.py"],
    [
        PAPER_ROOT
        / "accepted_figure_revisions"
        / "02_figure_02_representative_forecast_trajectories.py"
    ],
    [PAPER_ROOT / "figure4_spacing_options" / "04_figure_04_spacing_options.py"],
    [
        PAPER_ROOT
        / "accepted_figure_revisions"
        / "05_figure_05_basin_skill_trajectories.py"
    ],
    [
        PAPER_ROOT
        / "figure6_station_options"
        / "06a_current_figure_06_typography_review.py"
    ],
    [
        PAPER_ROOT
        / "figure8_spacing_options"
        / "08_figure_08_close_columns.py"
    ],
    [
        PAPER_ROOT
        / "accepted_figure_revisions"
        / "08_figure_08_continental_temporal_validation.py"
    ],
    [
        PAPER_ROOT
        / "figure9_replacement"
        / "09c_figure_09_original_gis_refinement.py"
    ],
    [
        PAPER_ROOT / "figure10_options" / "10_figure_10_design_options.py",
        "--design",
        "5",
    ],
]


def main() -> None:
    for number, command in enumerate(COMMANDS, start=1):
        script = Path(command[0])
        if not script.exists():
            raise FileNotFoundError(f"Accepted figure script is missing: {script}")
        rendered = [sys.executable, *(str(item) for item in command)]
        print(f"[{number}/{len(COMMANDS)}] {script.relative_to(PROJECT_ROOT)}", flush=True)
        subprocess.run(rendered, cwd=PROJECT_ROOT, check=True)
    print("Accepted figure suite rebuilt successfully.", flush=True)


if __name__ == "__main__":
    main()
