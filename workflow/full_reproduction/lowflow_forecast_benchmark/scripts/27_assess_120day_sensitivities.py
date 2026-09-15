#!/usr/bin/env python3
"""Summarize the independent 105/120-day sensitivity panel."""

from __future__ import annotations

import json
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from lib.common import atomic_csv, atomic_json, load_benchmark_config  # noqa: E402
from lib.extended_120 import OUTPUT_ROOT  # noqa: E402


ROOT = OUTPUT_ROOT / "sensitivities"


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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard-count", type=int, default=4)
    args = parser.parse_args()
    config = load_benchmark_config()
    metric_frames: list[pd.DataFrame] = []
    failure_frames: list[pd.DataFrame] = []
    for shard in range(int(args.shard_count)):
        folder = ROOT / f"sensitivity_shard_{shard:02d}"
        receipt = folder / "_SUCCESS.json"
        if not receipt.exists():
            raise FileNotFoundError(f"Missing completed sensitivity shard: {receipt}")
        metric_frames.append(pd.read_csv(
            folder / "sensitivity_metrics.csv.gz", dtype={"GAGE_ID": str}
        ))
        failure = folder / "sensitivity_failures.csv"
        if failure.exists() and failure.stat().st_size:
            try:
                frame = pd.read_csv(failure, dtype={"GAGE_ID": str})
            except pd.errors.EmptyDataError:
                frame = pd.DataFrame()
            if not frame.empty:
                failure_frames.append(frame)
    metrics = pd.concat(metric_frames, ignore_index=True)
    # The numerical RNG seed is basin-specific, but it is not an estimand or a
    # grouping variable. Collapse it to the five prespecified shuffle-repeat
    # labels (and zero for all other variants) before pairing basins.
    repeated = metrics["variant"].str.extract(r"shuffle_repeat_(\d+)", expand=False)
    metrics["shuffle_seed"] = repeated.fillna("0").astype(int)
    failures = pd.concat(failure_frames, ignore_index=True) if failure_frames else pd.DataFrame()
    atomic_csv(metrics, ROOT / "sensitivity_metrics.csv.gz", compression="gzip")
    atomic_csv(failures, ROOT / "sensitivity_failures.csv")
    keys = [
        "GAGE_ID", "spatial_group", "variant", "score", "lead_days", "n",
        "threshold_mode", "members", "shuffle_seed",
    ]
    coherent = metrics[metrics["model"].eq("coherent")][keys + ["value"]]
    shuffled = metrics[metrics["model"].eq("shuffled")][keys + ["value"]]
    paired = coherent.merge(shuffled, on=keys, suffixes=("_coherent", "_shuffled"))
    paired["improvement"] = paired["value_shuffled"] - paired["value_coherent"]
    paired["relative_skill_percent"] = np.where(
        paired["value_shuffled"].gt(0),
        100.0 * paired["improvement"] / paired["value_shuffled"],
        np.nan,
    )
    rows: list[dict] = []
    grouping = ["variant", "score", "lead_days", "threshold_mode", "members", "shuffle_seed"]
    for keys_group, group in paired.groupby(grouping):
        variant, score, lead, mode, members, shuffle_seed = keys_group
        low, high = hierarchical_ci(
            group, int(config["bootstrap_replicates"]),
            int(config["random_seed"] + 52000 + int(lead) + int(members)
                + int(shuffle_seed) + sum(map(ord, str(variant) + str(score)))),
        )
        coherent_score = float(np.average(group["value_coherent"], weights=group["n"]))
        shuffled_score = float(np.average(group["value_shuffled"], weights=group["n"]))
        rows.append({
            "variant": variant,
            "score": score,
            "lead_days": int(lead),
            "threshold_mode": mode,
            "members": int(members),
            "shuffle_seed": int(shuffle_seed),
            "basins": int(len(group)),
            "cases": int(group["n"].sum()),
            "coherent_score": coherent_score,
            "shuffled_score": shuffled_score,
            "weighted_improvement": float(np.average(group["improvement"], weights=group["n"])),
            "relative_skill_percent": float(
                100.0 * (shuffled_score - coherent_score) / shuffled_score
                if shuffled_score > 0 else np.nan
            ),
            "bootstrap_ci_low": float(low),
            "bootstrap_ci_high": float(high),
            "basin_fraction_improved": float(group["improvement"].gt(0).mean()),
        })
    comparison = pd.DataFrame(rows)
    atomic_csv(comparison, ROOT / "sensitivity_comparisons.csv")
    summaries = comparison.groupby("variant", as_index=False).agg(
        comparisons=("weighted_improvement", "size"),
        positive=("weighted_improvement", lambda x: int((x > 0).sum())),
        significant_positive=("bootstrap_ci_low", lambda x: int((x > 0).sum())),
        mean_basin_fraction_improved=("basin_fraction_improved", "mean"),
        basins_min=("basins", "min"),
        basins_max=("basins", "max"),
        relative_skill_min=("relative_skill_percent", "min"),
        relative_skill_max=("relative_skill_percent", "max"),
    )
    summaries["passes"] = (
        summaries["comparisons"].eq(8)
        & summaries["positive"].eq(8)
        & summaries["significant_positive"].ge(6)
        & summaries["mean_basin_fraction_improved"].gt(0.65)
    )
    atomic_csv(summaries, ROOT / "sensitivity_summary.csv")
    decision = {
        "status": "complete",
        "variants": int(len(summaries)),
        "variants_passing": int(summaries["passes"].sum()),
        "all_variants_pass": bool(summaries["passes"].all()),
        "test_scope": "Q10 onset; four path outcomes; 105 and 120 days",
        "summary": summaries.to_dict(orient="records"),
    }
    atomic_json(decision, ROOT / "SENSITIVITY_DECISION.json")
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
