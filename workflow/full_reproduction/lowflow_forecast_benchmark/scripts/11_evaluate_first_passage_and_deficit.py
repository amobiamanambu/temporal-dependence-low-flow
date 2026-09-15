#!/usr/bin/env python3
"""Summarize first-passage and deficit skill from the coherent-trajectory experiment."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from lib.common import RESULTS_ROOT, atomic_csv, atomic_json  # noqa: E402


def main() -> None:
    source = RESULTS_ROOT / "04_trajectories" / "trajectory_basin_metrics.csv.gz"
    if not source.exists():
        raise FileNotFoundError("Run 09_test_coherent_trajectories.py first")
    metrics = pd.read_csv(source, dtype={"GAGE_ID": str})
    keys = ["GAGE_ID", "spatial_group", "threshold_quantile", "threshold_name", "lead_days", "n"]
    reference = metrics[metrics["model"].eq("analog_marginal_shuffle")]
    candidate = metrics[metrics["model"].eq("coherent_block_analog")]
    paired = reference.merge(candidate, on=keys, suffixes=("_reference", "_candidate"))
    score_columns = [
        "first_passage_brier", "first_passage_discrete_crps", "deficit_volume_crps",
        "endpoint_brier",
    ]
    rows = []
    for (quantile, name, lead), group in paired.groupby(
        ["threshold_quantile", "threshold_name", "lead_days"]
    ):
        weights = group["n"].to_numpy(float)
        for score in score_columns:
            improvement = group[f"{score}_reference"] - group[f"{score}_candidate"]
            rng = np.random.default_rng(20260907 + int(lead) + int(100 * quantile) + sum(map(ord, score)))
            bootstrap = np.empty(2000)
            spatial_groups = sorted(group["spatial_group"].astype(str).unique())
            for replicate in range(len(bootstrap)):
                numerator = 0.0
                denominator = 0.0
                for spatial_group in rng.choice(spatial_groups, len(spatial_groups), replace=True):
                    candidates = np.flatnonzero(group["spatial_group"].astype(str).to_numpy() == spatial_group)
                    selected = rng.choice(candidates, len(candidates), replace=True)
                    numerator += float(np.sum(improvement.iloc[selected].to_numpy() * weights[selected]))
                    denominator += float(np.sum(weights[selected]))
                bootstrap[replicate] = numerator / denominator
            low, high = np.quantile(bootstrap, [0.025, 0.975])
            rows.append({
                "threshold_quantile": float(quantile), "threshold_name": name,
                "lead_days": int(lead), "score": score, "basins": int(len(group)),
                "cases": int(weights.sum()),
                "weighted_improvement_coherent_vs_shuffled": float(
                    np.average(improvement, weights=weights)
                ),
                "bootstrap_ci_low": float(low), "bootstrap_ci_high": float(high),
                "basin_fraction_improved": float((improvement > 0).mean()),
                "positive_means_coherent_is_better": True,
            })
    summary = pd.DataFrame(rows)
    output = RESULTS_ROOT / "06_first_passage"
    atomic_csv(summary, output / "first_passage_deficit_comparison.csv")
    primary = summary[
        summary["threshold_quantile"].eq(0.1)
        & summary["score"].isin(["first_passage_brier", "first_passage_discrete_crps",
                                  "deficit_volume_crps"])
    ]
    decision = {
        "status": "complete", "comparison": "coherent_block_vs_same_marginals_shuffled",
        "primary_comparisons": int(len(primary)),
        "positive_primary_comparisons": int(
            primary["weighted_improvement_coherent_vs_shuffled"].gt(0).sum()
        ),
        "mean_basin_fraction_improved": float(primary["basin_fraction_improved"].mean()),
        "interpretation": (
            "The control preserves each daily analog marginal exactly and shuffles only "
            "member identity across days, isolating temporal dependence."
        ),
    }
    atomic_json(decision, output / "_SUCCESS.json")
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
