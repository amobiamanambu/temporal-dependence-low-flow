#!/usr/bin/env python3
"""Assess robustness of the frozen temporal-coherence comparison."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from lib.common import RESULTS_ROOT, atomic_csv, atomic_json, load_benchmark_config  # noqa: E402


def hierarchical_ci(group: pd.DataFrame, replicates: int,
                    seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    regional = [values for _, values in group.groupby("spatial_group", dropna=False)]
    numerator = np.empty((replicates, len(regional)))
    denominator = np.empty_like(numerator)
    for column, values in enumerate(regional):
        x = values["improvement"].to_numpy(float)
        w = values["n"].to_numpy(float)
        indices = rng.integers(0, len(values), size=(replicates, len(values)))
        numerator[:, column] = np.sum(x[indices] * w[indices], axis=1)
        denominator[:, column] = np.sum(w[indices], axis=1)
    selected = rng.integers(0, len(regional), size=(replicates, len(regional)))
    rows = np.arange(replicates)[:, None]
    boot = numerator[rows, selected].sum(axis=1) / denominator[rows, selected].sum(axis=1)
    return tuple(np.quantile(boot, [0.025, 0.975]))


def main() -> None:
    config = load_benchmark_config()
    root = RESULTS_ROOT / "09_extended_forecast" / "sensitivities"
    metrics = pd.read_csv(root / "sensitivity_metrics.csv.gz", dtype={"GAGE_ID": str})
    keys = [
        "GAGE_ID", "spatial_group", "variant", "score", "lead_days", "n",
        "threshold_mode", "members", "shuffle_seed",
    ]
    coherent = metrics[metrics["model"].eq("coherent")][keys + ["value"]]
    shuffled = metrics[metrics["model"].eq("shuffled")][keys + ["value"]]
    paired = coherent.merge(
        shuffled, on=keys, suffixes=("_coherent", "_shuffled")
    )
    paired["improvement"] = paired["value_shuffled"] - paired["value_coherent"]
    rows: list[dict] = []
    for (variant, score, lead, mode, members, seed), group in paired.groupby(
        ["variant", "score", "lead_days", "threshold_mode", "members", "shuffle_seed"]
    ):
        low, high = hierarchical_ci(
            group, int(config["bootstrap_replicates"]),
            int(config["random_seed"] + 41000 + lead + members + seed
                + sum(map(ord, variant + score))),
        )
        rows.append({
            "variant": variant, "score": score, "lead_days": int(lead),
            "threshold_mode": mode, "members": int(members),
            "shuffle_seed": int(seed), "basins": int(len(group)),
            "cases": int(group["n"].sum()),
            "weighted_improvement": float(np.average(
                group["improvement"], weights=group["n"]
            )),
            "bootstrap_ci_low": float(low), "bootstrap_ci_high": float(high),
            "basin_fraction_improved": float(group["improvement"].gt(0).mean()),
        })
    comparison = pd.DataFrame(rows)
    atomic_csv(comparison, root / "sensitivity_comparisons.csv")
    summaries = comparison.groupby("variant", as_index=False).agg(
        comparisons=("weighted_improvement", "size"),
        positive=("weighted_improvement", lambda x: int((x > 0).sum())),
        significant_positive=("bootstrap_ci_low", lambda x: int((x > 0).sum())),
        mean_basin_fraction_improved=("basin_fraction_improved", "mean"),
        basins_min=("basins", "min"), basins_max=("basins", "max"),
    )
    summaries["passes"] = (
        summaries["comparisons"].eq(16)
        & summaries["positive"].ge(15)
        & summaries["significant_positive"].ge(12)
        & summaries["mean_basin_fraction_improved"].gt(0.75)
    )
    atomic_csv(summaries, root / "sensitivity_summary.csv")
    decision = {
        "status": "complete", "variants": int(len(summaries)),
        "variants_passing": int(summaries["passes"].sum()),
        "all_variants_pass": bool(summaries["passes"].all()),
        "summary": summaries.to_dict(orient="records"),
    }
    atomic_json(decision, root / "SENSITIVITY_DECISION.json")
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
