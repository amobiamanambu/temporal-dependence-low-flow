#!/usr/bin/env python3
"""Select a stratified candidate panel without consulting new forecast outcomes."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from lib.common import (  # noqa: E402
    RESULTS_ROOT, atomic_csv, atomic_json, eligible_inventory, ensure_dirs,
    load_benchmark_config,
)


def round_robin_sample(frame: pd.DataFrame, size: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    work = frame.copy()
    work["reference_group"] = np.where(
        work["is_reference"].fillna(False), "reference", "nonreference"
    )
    work["record_group"] = pd.qcut(
        work["record_years"].rank(method="first"), 3,
        labels=["short", "medium", "long"],
    ).astype(str)
    work["stratum"] = (
        work["spatial_group"].astype(str) + "|" + work["quality_tier"].astype(str)
        + "|" + work["reference_group"] + "|" + work["record_group"]
    )
    work["random_order"] = rng.random(len(work))
    groups = {
        key: group.sort_values("random_order").index.tolist()
        for key, group in work.groupby("stratum", dropna=False)
    }
    chosen = []
    while len(chosen) < min(size, len(work)):
        changed = False
        for key in sorted(groups):
            if groups[key] and len(chosen) < size:
                chosen.append(groups[key].pop(0))
                changed = True
        if not changed:
            break
    return work.loc[chosen].sort_values(["spatial_group", "GAGE_ID"]).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--basins", type=int, default=192)
    args = parser.parse_args()
    ensure_dirs()
    config = load_benchmark_config()
    inventory = eligible_inventory()
    total = int(args.basins)
    # Oversample basins with established long-lead data coverage, but never use a
    # forecast score or candidate prediction to select a gauge.
    long_target = min(total // 3, int(inventory["max_stage18_lead"].ge(30).sum() // 2))
    long = round_robin_sample(
        inventory[inventory["max_stage18_lead"].ge(30)], long_target,
        int(config["random_seed"]),
    )
    remaining = inventory[~inventory["GAGE_ID"].isin(long["GAGE_ID"])]
    medium_target = min(total // 3, int(remaining["max_stage18_lead"].ge(7).sum() // 3))
    medium = round_robin_sample(
        remaining[remaining["max_stage18_lead"].ge(7)], medium_target,
        int(config["random_seed"]) + 1,
    )
    remaining = remaining[~remaining["GAGE_ID"].isin(medium["GAGE_ID"])]
    broad = round_robin_sample(
        remaining, total - len(long) - len(medium), int(config["random_seed"]) + 2,
    )
    panel = pd.concat([long, medium, broad], ignore_index=True).drop_duplicates("GAGE_ID")
    panel["coverage_stratum"] = np.select(
        [panel["GAGE_ID"].isin(long["GAGE_ID"]), panel["GAGE_ID"].isin(medium["GAGE_ID"])],
        ["day30_coverage", "day7plus_coverage"], default="broad_coverage",
    )
    panel["role"] = "candidate_screening"
    output = RESULTS_ROOT / "01_panel" / "candidate_screening_panel.csv"
    atomic_csv(panel, output)
    receipt = {
        "status": "complete", "basins": int(len(panel)),
        "ecoregions": int(panel["spatial_group"].nunique()),
        "reference_basins": int(panel["is_reference"].fillna(False).sum()),
        "selection_uses_candidate_scores_or_predictions": False,
        "selection_uses_preexisting_data_sufficiency_only": True,
        "output": str(output),
    }
    atomic_json(receipt, RESULTS_ROOT / "01_panel" / "_SUCCESS.json")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
