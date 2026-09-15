#!/usr/bin/env python3
"""Run all marginal endpoint candidates in one efficient pass through each basin."""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from lib.common import RESULTS_ROOT, load_benchmark_config  # noqa: E402
from lib.endpoint import run_endpoint  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--all-leads", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--max-basins", type=int)
    args = parser.parse_args()
    panel = pd.read_csv(
        RESULTS_ROOT / "01_panel" / "candidate_screening_panel.csv",
        dtype={"GAGE_ID": str},
    )
    if args.max_basins:
        panel = panel.head(int(args.max_basins)).copy()
    config = load_benchmark_config()
    leads = config["lead_days"] if args.all_leads else config["sentinel_leads"]
    methods = [
        "binary_idr", "student_t_innovations", "gpd_tail_splice",
        "asymmetric_laplace_mixture", "quantile_forest",
        "gpd_isotonic_calibrated", "calibration_stack",
        "state_dependent_stack",
    ]
    receipt = run_endpoint(
        panel, methods, [int(value) for value in leads], int(args.workers),
        "02_endpoint_candidates", bool(args.force),
    )
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
