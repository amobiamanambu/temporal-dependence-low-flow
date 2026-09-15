#!/usr/bin/env python3
"""Run only predeclared winning methods on basins excluded from candidate screening."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from lib.common import RESULTS_ROOT, atomic_csv, atomic_json, eligible_inventory, load_benchmark_config  # noqa: E402
from lib.endpoint import run_endpoint  # noqa: E402
from lib.operational import run_operational  # noqa: E402
from lib.regional import build_library, evaluate_regional  # noqa: E402
from lib.trajectory import run_trajectory  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--component", choices=["endpoint", "operational", "trajectory", "all"],
                        default="all")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    decision_path = RESULTS_ROOT / "07_comparison" / "CANDIDATE_DECISION.json"
    if not decision_path.exists():
        raise FileNotFoundError("Run 12_compare_all_candidates.py first")
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    screen = pd.read_csv(RESULTS_ROOT / "01_panel" / "candidate_screening_panel.csv",
                         dtype={"GAGE_ID": str})
    # Stage 13 confirms candidates on the Stage-18-eligible population that was
    # not used for screening. Stage 13b separately evaluates accepted Tier-1/2
    # basins absent from Stage 18, so keeping the populations separate prevents
    # duplicate rows when Stage 14 combines their archived outputs.
    inventory = eligible_inventory()
    confirmation = inventory[~inventory["GAGE_ID"].isin(set(screen["GAGE_ID"]))].copy()
    atomic_csv(confirmation, RESULTS_ROOT / "08_full_confirmation" / "confirmation_basins.csv")
    config = load_benchmark_config()
    receipts = {}

    if args.component in {"endpoint", "all"}:
        endpoint = decision["endpoint"]
        winner = endpoint["winner"]
        if endpoint["new_endpoint_method_promoted"]:
            if winner == "regional_innovation_shrinkage":
                library = build_library(
                    screen, [int(v) for v in config["lead_days"]], int(args.workers),
                    bool(args.force), None,
                )
                receipts["endpoint"] = evaluate_regional(
                    confirmation, library, [int(v) for v in config["lead_days"]],
                    int(args.workers), "08_full_confirmation/regional_endpoint",
                )
            else:
                receipts["endpoint"] = run_endpoint(
                    confirmation, [winner], [int(v) for v in config["lead_days"]],
                    int(args.workers), "08_full_confirmation/endpoint", bool(args.force),
                )
        else:
            receipts["endpoint"] = {
                "status": "not_run", "reason": "no_new_method_passed_promotion_rule",
                "retained_method": winner,
            }

    if args.component in {"operational", "all"}:
        operational = decision.get("operational", {})
        if operational.get("promote_to_full_confirmation", False):
            receipts["operational"] = run_operational(
                confirmation, int(args.workers), "08_full_confirmation/operational",
                [int(v) for v in config["operational_lead_days"]],
            )
        else:
            receipts["operational"] = {
                "status": "not_run", "reason": "candidate_did_not_pass_operational_rule"
            }

    if args.component in {"trajectory", "all"}:
        trajectory = decision.get("trajectory", {})
        if trajectory.get("promote_to_full_confirmation", False):
            receipts["trajectory"] = run_trajectory(
                confirmation, int(args.workers), "08_full_confirmation/trajectories"
            )
        else:
            receipts["trajectory"] = {
                "status": "not_run", "reason": "coherent_paths_did_not_pass_promotion_rule"
            }

    result = {
        "status": "complete", "screening_basins_excluded": int(len(screen)),
        "confirmation_basins_available": int(len(confirmation)), "components": receipts,
        "manuscript_modified": False,
    }
    atomic_json(result, RESULTS_ROOT / "08_full_confirmation" / "_SUCCESS.json")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
