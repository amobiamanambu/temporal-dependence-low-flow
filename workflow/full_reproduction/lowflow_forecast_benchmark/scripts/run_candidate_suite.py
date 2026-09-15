#!/usr/bin/env python3
"""Run the complete candidate screen sequentially, stopping on a failed component."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--regional-training-basins", type=int, default=512)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    commands = [
        [sys.executable, str(root / "00_audit_inputs.py")],
        [sys.executable, str(root / "01_select_screening_panel.py")],
        [sys.executable, str(root / "run_endpoint_candidate_suite.py"),
         "--workers", str(args.workers)],
        [sys.executable, str(root / "07b_test_split_conformal_intervals.py"),
         "--workers", str(args.workers)],
        [sys.executable, str(root / "08_test_regional_innovations.py"),
         "--workers", str(args.workers), "--max-training-basins",
         str(args.regional_training_basins)],
        [sys.executable, str(root / "09_test_coherent_trajectories.py"),
         "--workers", str(args.workers)],
        [sys.executable, str(root / "10_test_operational_survival_mixture.py"),
         "--workers", str(args.workers)],
        [sys.executable, str(root / "11_evaluate_first_passage_and_deficit.py")],
        [sys.executable, str(root / "12_compare_all_candidates.py")],
    ]
    for number, command in enumerate(commands, start=1):
        print(f"\nCandidate suite {number}/{len(commands)}: {' '.join(command)}", flush=True)
        subprocess.run(command, check=True)
    print("\nCandidate suite complete. Inspect results/07_comparison before full confirmation.")


if __name__ == "__main__":
    main()
