#!/usr/bin/env python3
"""Build training-only ecoregional innovation libraries and test partial pooling."""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from lib.common import RESULTS_ROOT, load_benchmark_config  # noqa: E402
from lib.regional import build_library, evaluate_regional  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--all-leads", action="store_true")
    parser.add_argument("--force-library", action="store_true")
    parser.add_argument("--max-training-basins", type=int)
    parser.add_argument("--max-evaluation-basins", type=int)
    args = parser.parse_args()
    panel = pd.read_csv(RESULTS_ROOT / "01_panel" / "candidate_screening_panel.csv",
                        dtype={"GAGE_ID": str})
    if args.max_evaluation_basins:
        panel = panel.head(int(args.max_evaluation_basins)).copy()
    config = load_benchmark_config()
    leads = [int(v) for v in (config["lead_days"] if args.all_leads else config["sentinel_leads"])]
    library = build_library(
        panel, leads, int(args.workers), bool(args.force_library),
        int(args.max_training_basins) if args.max_training_basins else None,
    )
    print(json.dumps(evaluate_regional(panel, library, leads, int(args.workers)), indent=2))


if __name__ == "__main__":
    main()
