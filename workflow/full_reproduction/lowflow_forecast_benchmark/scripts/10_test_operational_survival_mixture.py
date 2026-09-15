#!/usr/bin/env python3
"""Test an unconditional dry-spell-survival mixture from initialization-time data only."""

import argparse
import json
import sys
from pathlib import Path
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from lib.common import RESULTS_ROOT  # noqa: E402
from lib.operational import run_operational  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-basins", type=int)
    args = parser.parse_args()
    panel = pd.read_csv(RESULTS_ROOT / "01_panel" / "candidate_screening_panel.csv",
                        dtype={"GAGE_ID": str})
    if args.max_basins:
        panel = panel.head(int(args.max_basins)).copy()
    print(json.dumps(run_operational(panel, int(args.workers)), indent=2))


if __name__ == "__main__":
    main()
