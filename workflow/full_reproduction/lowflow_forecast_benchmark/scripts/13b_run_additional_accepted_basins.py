#!/usr/bin/env python3
"""Evaluate accepted basins absent from every Stage-18 endpoint result."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from lib.common import (  # noqa: E402
    RESULTS_ROOT, all_accepted_inventory, atomic_json, eligible_inventory,
    load_benchmark_config,
)
from lib.operational import run_operational  # noqa: E402
from lib.trajectory import run_trajectory  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    accepted = all_accepted_inventory()
    prior = set(eligible_inventory()["GAGE_ID"])
    additional = accepted[~accepted["GAGE_ID"].isin(prior)].copy()
    config = load_benchmark_config()
    result = {
        "basins_requested": int(len(additional)),
        "operational": run_operational(
            additional, int(args.workers), "08_full_confirmation/additional_operational",
            [int(v) for v in config["operational_lead_days"]],
        ),
        "trajectory": run_trajectory(
            additional, int(args.workers), "08_full_confirmation/additional_trajectories"
        ),
    }
    atomic_json(result, RESULTS_ROOT / "08_full_confirmation" / "additional_SUCCESS.json")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
