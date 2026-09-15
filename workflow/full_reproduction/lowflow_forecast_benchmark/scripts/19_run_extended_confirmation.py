#!/usr/bin/env python3
"""Run one restartable shard of the frozen 1--90-day confirmation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from lib.common import RESULTS_ROOT, all_accepted_inventory  # noqa: E402
from lib.extended_confirmation import run_confirmation_shard  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--shard-count", type=int, default=4)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    decision_path = (
        RESULTS_ROOT / "09_extended_forecast" / "EXTENDED_SCREEN_DECISION.json"
    )
    if not decision_path.exists():
        raise FileNotFoundError("Run 18_compare_extended_trajectories.py first")
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    if decision.get("winning_path_architecture") != "hydrograph_analog":
        raise RuntimeError(
            "The frozen confirmation implementation expects hydrograph_analog; "
            "inspect the screen decision before continuing"
        )
    screen = pd.read_csv(
        RESULTS_ROOT / "01_panel" / "candidate_screening_panel.csv",
        dtype={"GAGE_ID": str},
    )
    accepted = all_accepted_inventory()
    confirmation = accepted[
        ~accepted["GAGE_ID"].isin(set(screen["GAGE_ID"]))
    ].copy()
    receipt = run_confirmation_shard(
        confirmation, int(args.shard_index), int(args.shard_count), bool(args.force)
    )
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
