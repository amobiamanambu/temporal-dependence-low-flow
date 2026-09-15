#!/usr/bin/env python3
"""Screen future-unrestricted 1--90-day trajectory methods on the frozen panel."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from lib.common import RESULTS_ROOT  # noqa: E402
from lib.extended_trajectory import run_extended  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-basins", type=int)
    args = parser.parse_args()
    panel = pd.read_csv(
        RESULTS_ROOT / "01_panel" / "candidate_screening_panel.csv",
        dtype={"GAGE_ID": str},
    )
    receipt = run_extended(
        panel, int(args.workers), "09_extended_forecast/screen", args.max_basins
    )
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
