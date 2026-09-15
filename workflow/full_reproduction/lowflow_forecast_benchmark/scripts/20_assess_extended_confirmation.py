#!/usr/bin/env python3
"""Aggregate all confirmation shards and assess the frozen extended-range claims."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from lib.common import (  # noqa: E402
    RESULTS_ROOT, all_accepted_inventory, atomic_csv, atomic_json,
    load_benchmark_config,
)


SCORES = [
    "endpoint_brier", "event_brier", "timing_crps", "deficit_crps",
    "duration_crps", "minimum_flow_crps",
]
CORE_PATH_SCORES = [
    "event_brier", "timing_crps", "deficit_crps", "duration_crps",
    "minimum_flow_crps",
]
PAIRS = [
    ("hydrograph_analog", "hydrograph_marginal_shuffle", "coherence"),
    ("hydrograph_analog", "seasonal_climatology_path", "path_baseline"),
    ("hydrograph_analog", "constant_persistence_path", "path_baseline"),
    ("hydrograph_analog", "flow_season_analog", "state_complexity"),
    ("hydrograph_analog", "hydroclimate_analog", "state_complexity"),
    ("hydrograph_analog", "direct_endpoint_logistic", "direct_target"),
    ("hydrograph_analog", "direct_event_logistic", "direct_target"),
    ("direct_endpoint_logistic", "seasonal_climatology_path", "direct_baseline"),
    ("direct_endpoint_logistic", "constant_persistence_path", "direct_baseline"),
    ("direct_event_logistic", "seasonal_climatology_path", "direct_baseline"),
    ("direct_event_logistic", "constant_persistence_path", "direct_baseline"),
]


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
    boot = (
        numerator[rows, selected].sum(axis=1)
        / denominator[rows, selected].sum(axis=1)
    )
    return tuple(np.quantile(boot, [0.025, 0.975]))


def paired_basin(metrics: pd.DataFrame, candidate: str, reference: str,
                 score: str) -> pd.DataFrame:
    keys = [
        "GAGE_ID", "spatial_group", "threshold_quantile", "threshold_name",
        "lead_days", "target",
    ]
    left = metrics[metrics["model"].eq(candidate)][keys + ["n", score]].dropna(
        subset=[score]
    )
    right = metrics[metrics["model"].eq(reference)][keys + [score]].dropna(
        subset=[score]
    )
    paired = left.merge(right, on=keys, suffixes=("_candidate", "_reference"))
    if paired.empty:
        return paired
    if "n_candidate" in paired:
        paired["n"] = paired["n_candidate"]
    paired["improvement"] = (
        paired[f"{score}_reference"] - paired[f"{score}_candidate"]
    )
    return paired


def primary_comparisons(metrics: pd.DataFrame, config: dict) -> pd.DataFrame:
    rows: list[dict] = []
    for candidate, reference, family in PAIRS:
        for score in SCORES:
            paired = paired_basin(metrics, candidate, reference, score)
            if paired.empty:
                continue
            paired = paired[
                paired["threshold_name"].eq("Q10")
                & paired["target"].eq("onset")
            ]
            for lead, group in paired.groupby("lead_days"):
                low, high = hierarchical_ci(
                    group, int(config["bootstrap_replicates"]),
                    int(config["random_seed"] + 30000 + lead
                        + sum(map(ord, candidate + reference + score))),
                )
                rows.append({
                    "family": family, "candidate": candidate,
                    "reference": reference, "score": score,
                    "threshold_name": "Q10", "target": "onset",
                    "lead_days": int(lead), "basins": int(len(group)),
                    "cases": int(group["n"].sum()),
                    "weighted_improvement": float(np.average(
                        group["improvement"], weights=group["n"]
                    )),
                    "bootstrap_ci_low": float(low),
                    "bootstrap_ci_high": float(high),
                    "basin_fraction_improved": float(
                        group["improvement"].gt(0).mean()
                    ),
                })
    return pd.DataFrame(rows)


def secondary_comparisons(metrics: pd.DataFrame, attributes: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows: list[dict] = []
    regional_rows: list[dict] = []
    subgroup_rows: list[dict] = []
    for score in CORE_PATH_SCORES:
        paired = paired_basin(
            metrics, "hydrograph_analog", "hydrograph_marginal_shuffle", score
        )
        if paired.empty:
            continue
        selected = paired[
            paired["lead_days"].isin([30, 45, 60, 90])
            & (
                (paired["target"].eq("onset")
                 & paired["threshold_name"].isin(["Q5", "Q20"]))
                | (paired["target"].eq("recovery")
                   & paired["threshold_name"].eq("Q10"))
            )
        ]
        for (name, target, lead), group in selected.groupby(
            ["threshold_name", "target", "lead_days"]
        ):
            rows.append({
                "score": score, "threshold_name": name, "target": target,
                "lead_days": int(lead), "basins": int(len(group)),
                "cases": int(group["n"].sum()),
                "weighted_improvement": float(np.average(
                    group["improvement"], weights=group["n"]
                )),
                "basin_fraction_improved": float(
                    group["improvement"].gt(0).mean()
                ),
            })

        primary = paired[
            paired["threshold_name"].eq("Q10")
            & paired["target"].eq("onset")
            & paired["lead_days"].isin([30, 45, 60, 90])
        ]
        for (region, lead), group in primary.groupby(["spatial_group", "lead_days"]):
            regional_rows.append({
                "spatial_group": region, "score": score, "lead_days": int(lead),
                "basins": int(len(group)), "cases": int(group["n"].sum()),
                "weighted_improvement": float(np.average(
                    group["improvement"], weights=group["n"]
                )),
                "basin_fraction_improved": float(group["improvement"].gt(0).mean()),
            })
        augmented = primary.merge(
            attributes[["GAGE_ID", "is_reference", "quality_tier"]],
            on="GAGE_ID", how="left",
        )
        augmented["reference_group"] = np.where(
            augmented["is_reference"].fillna(False), "reference", "non_reference"
        )
        for (reference_group, tier, lead), group in augmented.groupby(
            ["reference_group", "quality_tier", "lead_days"], dropna=False
        ):
            subgroup_rows.append({
                "reference_group": reference_group, "quality_tier": tier,
                "score": score, "lead_days": int(lead),
                "basins": int(len(group)), "cases": int(group["n"].sum()),
                "weighted_improvement": float(np.average(
                    group["improvement"], weights=group["n"]
                )),
                "basin_fraction_improved": float(group["improvement"].gt(0).mean()),
            })
    return pd.DataFrame(rows), pd.DataFrame(regional_rows), pd.DataFrame(subgroup_rows)


def summarize(frame: pd.DataFrame) -> dict:
    if frame.empty:
        return {"comparisons": 0, "positive": 0, "significant_positive": 0}
    return {
        "comparisons": int(len(frame)),
        "positive": int(frame["weighted_improvement"].gt(0).sum()),
        "significant_positive": int(frame["bootstrap_ci_low"].gt(0).sum()),
        "mean_basin_fraction_improved": float(
            frame["basin_fraction_improved"].mean()
        ),
        "basins_min": int(frame["basins"].min()),
        "basins_max": int(frame["basins"].max()),
    }


def main() -> None:
    config = load_benchmark_config()
    root = RESULTS_ROOT / "09_extended_forecast"
    receipts = []
    metrics_frames = []
    failure_frames = []
    for shard in range(4):
        folder = root / f"confirmation_shard_{shard:02d}"
        receipt_path = folder / "_SUCCESS.json"
        if not receipt_path.exists():
            raise FileNotFoundError(f"Missing completed shard: {receipt_path}")
        receipts.append(json.loads(receipt_path.read_text(encoding="utf-8")))
        metrics_frames.append(pd.read_csv(
            folder / "extended_confirmation_metrics.csv.gz", dtype={"GAGE_ID": str}
        ))
        failure_path = folder / "extended_confirmation_failures.csv"
        if failure_path.exists() and failure_path.stat().st_size:
            failure_frames.append(pd.read_csv(failure_path, dtype={"GAGE_ID": str}))
    metrics = pd.concat(metrics_frames, ignore_index=True)
    failures = pd.concat(failure_frames, ignore_index=True) if failure_frames else pd.DataFrame()
    attributes = all_accepted_inventory()
    atomic_csv(metrics, root / "extended_confirmation_metrics.csv.gz", compression="gzip")
    atomic_csv(failures, root / "extended_confirmation_failures.csv")

    comparison = primary_comparisons(metrics, config)
    secondary, regional, subgroups = secondary_comparisons(metrics, attributes)
    atomic_csv(comparison, root / "extended_confirmation_comparisons.csv")
    atomic_csv(secondary, root / "extended_secondary_sensitivity.csv")
    atomic_csv(regional, root / "extended_ecoregion_consistency.csv")
    atomic_csv(subgroups, root / "extended_reference_tier_consistency.csv")

    long = comparison[comparison["lead_days"].isin([30, 45, 60, 90])]
    coherence = long[
        long["candidate"].eq("hydrograph_analog")
        & long["reference"].eq("hydrograph_marginal_shuffle")
        & long["score"].isin(CORE_PATH_SCORES)
    ]
    baseline_summaries = {}
    for reference in ("seasonal_climatology_path", "constant_persistence_path"):
        selected = long[
            long["candidate"].eq("hydrograph_analog")
            & long["reference"].eq(reference)
            & long["score"].isin(CORE_PATH_SCORES)
        ]
        baseline_summaries[reference] = summarize(selected)
    coherence_summary = summarize(coherence)
    regional_primary = regional[
        regional["score"].isin([
            "event_brier", "timing_crps", "deficit_crps", "duration_crps"
        ])
    ]
    regional_positive_fraction = float(
        regional_primary["weighted_improvement"].gt(0).mean()
    )
    reference_primary = subgroups[
        subgroups["reference_group"].eq("reference")
        & subgroups["score"].isin([
            "event_brier", "timing_crps", "deficit_crps", "duration_crps"
        ])
    ]
    reference_positive_fraction = float(
        reference_primary["weighted_improvement"].gt(0).mean()
    )
    coherence_confirmed = bool(
        coherence_summary["comparisons"] == 20
        and coherence_summary["positive"] >= 18
        and coherence_summary["significant_positive"] >= 15
        and coherence_summary["mean_basin_fraction_improved"] > 0.65
        and coherence_summary["basins_min"] >= 1000
        and regional_positive_fraction >= 0.85
        and reference_positive_fraction >= 0.85
    )
    baseline_confirmed = all(
        value["comparisons"] == 20
        and value["positive"] >= 16
        and value["significant_positive"] >= 12
        and value["mean_basin_fraction_improved"] > 0.60
        for value in baseline_summaries.values()
    )
    decision = {
        "status": "complete",
        "basins_attempted": int(sum(r["basins_requested"] for r in receipts)),
        "basins_with_any_result": int(metrics["GAGE_ID"].nunique()),
        "basins_excluded": int(len(failures)),
        "primary_question": (
            "Does temporal coherence improve future-unrestricted Q10 onset, "
            "timing, duration, and deficit prediction at 30-90 days?"
        ),
        "coherence_summary": coherence_summary,
        "path_baseline_summaries": baseline_summaries,
        "ecoregion_score_lead_fraction_positive": regional_positive_fraction,
        "reference_basin_score_lead_fraction_positive": reference_positive_fraction,
        "coherence_independently_confirmed": coherence_confirmed,
        "useful_path_forecast_confirmed": bool(coherence_confirmed and baseline_confirmed),
        "future_forcing_selection": False,
        "future_observations_used_by_eligible_models": False,
        "title_supported": bool(coherence_confirmed and baseline_confirmed),
        "tested_title": (
            "Temporal Coherence Improves Extended-Range Low-Flow Onset and "
            "Deficit Prediction in U.S. Catchments"
        ),
        "claim_boundaries": [
            "Forecasts are initialized weekly on an observed dry day with dry age >= 3 days.",
            "No numerical weather-prediction archive is used.",
            "The experiment predicts Q5/Q10/Q20 threshold events, not water-use impacts.",
            "No manuscript file was modified by this analysis stage.",
        ],
    }
    atomic_json(decision, root / "EXTENDED_FINAL_DECISION.json")
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
