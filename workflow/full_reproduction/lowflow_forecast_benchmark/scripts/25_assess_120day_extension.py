#!/usr/bin/env python3
"""Aggregate and evaluate the frozen 105/120-day extension."""

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
    RESULTS_ROOT, all_accepted_inventory, atomic_csv, atomic_json,
    load_benchmark_config,
)
from lib.extended_120 import EXTENSION_LEADS, LEADS_120, OUTPUT_ROOT  # noqa: E402


SCORES = (
    "endpoint_brier", "event_brier", "timing_crps", "deficit_crps",
    "duration_crps", "minimum_flow_crps",
)
CORE_SCORES = (
    "event_brier", "timing_crps", "deficit_crps", "duration_crps",
)
PAIRS = (
    ("hydrograph_analog", "hydrograph_marginal_shuffle", "coherence"),
    ("hydrograph_analog", "seasonal_climatology_path", "path_baseline"),
    ("hydrograph_analog", "constant_persistence_path", "path_baseline"),
    ("hydrograph_analog", "direct_event_logistic", "direct_target"),
    ("direct_event_logistic", "seasonal_climatology_path", "direct_baseline"),
    ("direct_event_logistic", "constant_persistence_path", "direct_baseline"),
)


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


def comparison_table(metrics: pd.DataFrame, config: dict) -> pd.DataFrame:
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
                    group,
                    int(config["bootstrap_replicates"]),
                    int(config["random_seed"] + 41000 + lead
                        + sum(map(ord, candidate + reference + score))),
                )
                candidate_score = float(np.average(
                    group[f"{score}_candidate"], weights=group["n"]
                ))
                reference_score = float(np.average(
                    group[f"{score}_reference"], weights=group["n"]
                ))
                relative_skill = (
                    100.0 * (reference_score - candidate_score) / reference_score
                    if reference_score > 0 else np.nan
                )
                rows.append({
                    "family": family,
                    "candidate": candidate,
                    "reference": reference,
                    "score": score,
                    "threshold_name": "Q10",
                    "target": "onset",
                    "lead_days": int(lead),
                    "basins": int(len(group)),
                    "cases": int(group["n"].sum()),
                    "candidate_score": candidate_score,
                    "reference_score": reference_score,
                    "weighted_improvement": float(np.average(
                        group["improvement"], weights=group["n"]
                    )),
                    "relative_skill_percent": relative_skill,
                    "bootstrap_ci_low": float(low),
                    "bootstrap_ci_high": float(high),
                    "basin_fraction_improved": float(
                        group["improvement"].gt(0).mean()
                    ),
                })
    return pd.DataFrame(rows)


def secondary_tables(metrics: pd.DataFrame, attributes: pd.DataFrame):
    severity_rows: list[dict] = []
    regional_rows: list[dict] = []
    subgroup_rows: list[dict] = []
    for reference in (
        "hydrograph_marginal_shuffle",
        "seasonal_climatology_path",
        "constant_persistence_path",
    ):
        for score in CORE_SCORES:
            paired = paired_basin(metrics, "hydrograph_analog", reference, score)
            if paired.empty:
                continue
            extension = paired[paired["lead_days"].isin(EXTENSION_LEADS)]
            severity = extension[
                (extension["target"].eq("onset")
                 & extension["threshold_name"].isin(["Q5", "Q20"]))
                | (extension["target"].eq("recovery")
                   & extension["threshold_name"].eq("Q10"))
            ]
            for (threshold, target, lead), group in severity.groupby(
                ["threshold_name", "target", "lead_days"]
            ):
                severity_rows.append({
                    "reference": reference,
                    "score": score,
                    "threshold_name": threshold,
                    "target": target,
                    "lead_days": int(lead),
                    "basins": int(len(group)),
                    "cases": int(group["n"].sum()),
                    "weighted_improvement": float(np.average(
                        group["improvement"], weights=group["n"]
                    )),
                    "basin_fraction_improved": float(
                        group["improvement"].gt(0).mean()
                    ),
                })

            primary = extension[
                extension["threshold_name"].eq("Q10")
                & extension["target"].eq("onset")
            ]
            for (region, lead), group in primary.groupby(
                ["spatial_group", "lead_days"]
            ):
                regional_rows.append({
                    "reference": reference,
                    "spatial_group": region,
                    "score": score,
                    "lead_days": int(lead),
                    "basins": int(len(group)),
                    "cases": int(group["n"].sum()),
                    "weighted_improvement": float(np.average(
                        group["improvement"], weights=group["n"]
                    )),
                    "basin_fraction_improved": float(
                        group["improvement"].gt(0).mean()
                    ),
                })
            augmented = primary.merge(
                attributes[["GAGE_ID", "is_reference", "quality_tier"]],
                on="GAGE_ID", how="left",
            )
            augmented["reference_group"] = np.where(
                augmented["is_reference"].fillna(False),
                "reference", "non_reference",
            )
            for (status, tier, lead), group in augmented.groupby(
                ["reference_group", "quality_tier", "lead_days"], dropna=False
            ):
                subgroup_rows.append({
                    "comparison_reference": reference,
                    "reference_group": status,
                    "quality_tier": tier,
                    "score": score,
                    "lead_days": int(lead),
                    "basins": int(len(group)),
                    "cases": int(group["n"].sum()),
                    "weighted_improvement": float(np.average(
                        group["improvement"], weights=group["n"]
                    )),
                    "basin_fraction_improved": float(
                        group["improvement"].gt(0).mean()
                    ),
                })
    return (
        pd.DataFrame(severity_rows),
        pd.DataFrame(regional_rows),
        pd.DataFrame(subgroup_rows),
    )


def summarize(frame: pd.DataFrame) -> dict:
    if frame.empty:
        return {
            "comparisons": 0, "positive": 0, "significant_positive": 0,
            "mean_basin_fraction_improved": np.nan,
            "basins_min": 0, "basins_max": 0,
        }
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard-count", type=int, default=4)
    args = parser.parse_args()
    config = load_benchmark_config()
    metric_frames: list[pd.DataFrame] = []
    failure_frames: list[pd.DataFrame] = []
    receipts: list[dict] = []
    for shard in range(int(args.shard_count)):
        folder = OUTPUT_ROOT / f"extension_shard_{shard:02d}"
        receipt_path = folder / "_SUCCESS.json"
        if not receipt_path.exists():
            raise FileNotFoundError(f"Missing completed shard: {receipt_path}")
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if receipt.get("predeclared_extension_leads") != list(EXTENSION_LEADS):
            raise RuntimeError(f"Unexpected extension receipt: {receipt_path}")
        receipts.append(receipt)
        metric_frames.append(pd.read_csv(
            folder / "extension_metrics.csv.gz", dtype={"GAGE_ID": str}
        ))
        failure_path = folder / "extension_failures.csv"
        if failure_path.exists() and failure_path.stat().st_size:
            failure_frames.append(pd.read_csv(
                failure_path, dtype={"GAGE_ID": str}
            ))
    metrics = pd.concat(metric_frames, ignore_index=True)
    failures = (
        pd.concat(failure_frames, ignore_index=True)
        if failure_frames else pd.DataFrame()
    )
    atomic_csv(metrics, OUTPUT_ROOT / "extension_120_metrics.csv.gz", compression="gzip")
    atomic_csv(failures, OUTPUT_ROOT / "extension_120_failures.csv")

    comparisons = comparison_table(metrics, config)
    attributes = all_accepted_inventory()
    severity, regional, subgroups = secondary_tables(metrics, attributes)
    atomic_csv(comparisons, OUTPUT_ROOT / "extension_120_comparisons.csv")
    atomic_csv(severity, OUTPUT_ROOT / "extension_120_severity_recovery.csv")
    atomic_csv(regional, OUTPUT_ROOT / "extension_120_ecoregions.csv")
    atomic_csv(subgroups, OUTPUT_ROOT / "extension_120_reference_tiers.csv")

    event_rates = metrics[
        metrics["model"].eq("hydrograph_analog")
        & metrics["threshold_name"].eq("Q10")
        & metrics["target"].eq("onset")
    ].groupby("lead_days", as_index=False).agg(
        basins=("GAGE_ID", "nunique"), cases=("n", "sum"), events=("events", "sum")
    )
    event_rates["event_rate"] = event_rates["events"] / event_rates["cases"]
    atomic_csv(event_rates, OUTPUT_ROOT / "extension_120_event_rates.csv")

    extension = comparisons[
        comparisons["lead_days"].isin(EXTENSION_LEADS)
        & comparisons["score"].isin(CORE_SCORES)
    ]
    summaries = {}
    for reference in (
        "hydrograph_marginal_shuffle",
        "seasonal_climatology_path",
        "constant_persistence_path",
    ):
        selected = extension[
            extension["candidate"].eq("hydrograph_analog")
            & extension["reference"].eq(reference)
        ]
        summaries[reference] = summarize(selected)

    endpoint = paired_basin(
        metrics, "hydrograph_analog", "hydrograph_marginal_shuffle",
        "endpoint_brier",
    )
    endpoint = endpoint[
        endpoint["threshold_name"].eq("Q10")
        & endpoint["target"].eq("onset")
    ]
    max_endpoint_difference = float(
        np.max(np.abs(endpoint["improvement"])) if not endpoint.empty else np.nan
    )

    regional_primary = regional[
        regional["reference"].eq("hydrograph_marginal_shuffle")
    ]
    subgroup_primary = subgroups[
        subgroups["comparison_reference"].eq("hydrograph_marginal_shuffle")
    ]
    regional_positive = float(
        regional_primary["weighted_improvement"].gt(0).mean()
    )
    subgroup_positive = float(
        subgroup_primary["weighted_improvement"].gt(0).mean()
    )

    coherence = summaries["hydrograph_marginal_shuffle"]
    coherence_supported = bool(
        coherence["comparisons"] == 8
        and coherence["positive"] == 8
        and coherence["significant_positive"] >= 6
        and coherence["mean_basin_fraction_improved"] > 0.65
        and coherence["basins_min"] >= 1000
        and max_endpoint_difference <= 1e-12
        and regional_positive >= 0.85
        and subgroup_positive >= 0.85
    )
    practical_supported = all(
        summary["comparisons"] == 8
        and summary["positive"] == 8
        and summary["significant_positive"] >= 6
        and summary["mean_basin_fraction_improved"] > 0.55
        for key, summary in summaries.items()
        if key != "hydrograph_marginal_shuffle"
    )
    conclusion = (
        "useful_state_conditioned_prediction_extends_to_120_days"
        if coherence_supported and practical_supported
        else "coherence_persists_but_practical_skill_boundary_precedes_120_days"
        if coherence_supported
        else "120_day_temporal_coherence_not_supported"
    )
    decision = {
        "status": "complete",
        "analysis": "frozen one-shot 105/120-day extension",
        "basins_attempted": int(sum(r["basins_requested"] for r in receipts)),
        "basins_with_any_result": int(metrics["GAGE_ID"].nunique()),
        "basins_excluded": int(len(failures)),
        "full_leads_recomputed_for_internal_consistency": list(LEADS_120),
        "predeclared_extension_leads": list(EXTENSION_LEADS),
        "core_outcomes": list(CORE_SCORES),
        "summaries": summaries,
        "max_absolute_endpoint_brier_difference_same_marginals": max_endpoint_difference,
        "ecoregion_score_lead_fraction_positive": regional_positive,
        "reference_tier_score_lead_fraction_positive": subgroup_positive,
        "coherence_supported_at_105_120": coherence_supported,
        "practical_skill_supported_at_105_120": practical_supported,
        "conclusion": conclusion,
        "future_forcing_selection": False,
        "future_observations_used_by_eligible_models": False,
        "existing_1_90_outputs_overwritten": False,
    }
    atomic_json(decision, OUTPUT_ROOT / "EXTENSION_120_DECISION.json")
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()

