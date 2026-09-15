#!/usr/bin/env python3
"""Compare fixed-ecoregion and region-resampled uncertainty intervals.

The primary manuscript treats the nine aggregated GAGES-II ecoregions as
fixed strata in a CONUS population and resamples basins within every stratum.
This sensitivity repeats selected comparisons with the earlier two-stage
scheme that also resamples the nine ecoregions. The latter is deliberately
conservative and unstable with only nine top-level groups, but it reveals
which borderline conclusions depend on treating the regional population as
fixed.

Run from the project root:

    python temporal_coherence_paper/scripts/39_assess_uncertainty_sensitivity.py
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd


PAPER_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = PAPER_ROOT.parent
BENCHMARK_ROOT = PROJECT_ROOT / "lowflow_forecast_benchmark"
CONFIG = BENCHMARK_ROOT / "config.json"
EXTENSION_METRICS = (
    BENCHMARK_ROOT / "results" / "10_extension_120" /
    "extension_120_metrics.csv.gz"
)
REVIEW_ROOT = PAPER_ROOT / "reviewer_strengthening" / "full"
REVIEW_METRICS = REVIEW_ROOT / "reviewer_strengthening_metrics.csv.gz"
PRIMARY_INTERVALS = REVIEW_ROOT / "archived_stratified_basin_intervals.csv"
RECONSTRUCTION_INTERVALS = REVIEW_ROOT / "reconstruction_comparisons.csv"
OUTPUT = REVIEW_ROOT / "uncertainty_scheme_sensitivity.csv"
RECEIPT = REVIEW_ROOT / "STAGE39_SUCCESS.json"

PRACTICAL_SCORES = (
    "event_brier",
    "timing_crps",
    "duration_crps",
    "deficit_crps",
)
RESIDUAL_SCORES = (
    "event_brier",
    "joint_onset_rps",
    "conditional_onset_rps",
    "duration_crps",
    "deficit_crps",
)


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        prefix=f".{path.stem}.", suffix=".tmp", dir=path.parent, delete=False
    ) as handle:
        temporary = Path(handle.name)
    try:
        frame.to_csv(temporary, index=False)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_json(value: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        prefix=f".{path.stem}.", suffix=".tmp", dir=path.parent,
        delete=False, mode="w", encoding="utf-8"
    ) as handle:
        temporary = Path(handle.name)
        json.dump(value, handle, indent=2)
        handle.write("\n")
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def normalize_gage(values: pd.Series) -> pd.Series:
    return values.astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(8)


def two_stage_interval(
    paired: pd.DataFrame,
    reference_score: float,
    replicates: int,
    seed: int,
) -> tuple[float, float]:
    """Resample basins within regions and then resample the nine regions.

    The resampling statistic follows the earlier analysis: each replicate is
    the case-weighted mean absolute score improvement. Interval endpoints are
    divided by the original paired reference score to express them on the same
    relative-score scale as the manuscript point estimate.
    """
    rng = np.random.default_rng(seed)
    regional = [group for _, group in paired.groupby("spatial_group", dropna=False)]
    numerator = np.empty((replicates, len(regional)), dtype=float)
    denominator = np.empty_like(numerator)
    for column, group in enumerate(regional):
        improvement = (
            group["score_value_reference"] - group["score_value_candidate"]
        ).to_numpy(float)
        weights = group["n_candidate"].to_numpy(float)
        indices = rng.integers(0, len(group), size=(replicates, len(group)))
        numerator[:, column] = np.sum(
            improvement[indices] * weights[indices], axis=1
        )
        denominator[:, column] = np.sum(weights[indices], axis=1)
    selected = rng.integers(
        0, len(regional), size=(replicates, len(regional))
    )
    rows = np.arange(replicates)[:, None]
    absolute = (
        numerator[rows, selected].sum(axis=1)
        / denominator[rows, selected].sum(axis=1)
    )
    relative = 100.0 * absolute / reference_score
    return tuple(np.quantile(relative, [0.025, 0.975]))


def paired_scores(
    metrics: pd.DataFrame,
    candidate: str,
    reference: str,
    score: str,
    extra_keys: tuple[str, ...] = (),
) -> pd.DataFrame:
    keys = ["GAGE_ID", "spatial_group", "lead_days", *extra_keys]
    left = metrics[metrics["model"].eq(candidate)][keys + ["n", score]].dropna(
        subset=[score]
    )
    right = metrics[metrics["model"].eq(reference)][keys + [score]].dropna(
        subset=[score]
    )
    paired = left.merge(right, on=keys, suffixes=("_candidate", "_reference"))
    return paired.rename(columns={
        "n": "n_candidate",
        f"{score}_candidate": "score_value_candidate",
        f"{score}_reference": "score_value_reference",
    })


def point_summary(group: pd.DataFrame) -> tuple[float, float, float]:
    weights = group["n_candidate"].to_numpy(float)
    candidate = float(np.average(
        group["score_value_candidate"], weights=weights
    ))
    reference = float(np.average(
        group["score_value_reference"], weights=weights
    ))
    skill = 100.0 * (reference - candidate) / reference
    return candidate, reference, skill


def primary_lookup(
    table: pd.DataFrame,
    candidate: str,
    reference: str,
    score: str,
    lead: int,
) -> tuple[float, float]:
    selected = table[
        table["candidate"].eq(candidate)
        & table["reference"].eq(reference)
        & table["score"].eq(score)
        & table["lead_days"].eq(lead)
    ]
    if len(selected) != 1:
        raise RuntimeError(
            f"Expected one primary interval for {candidate}, {reference}, "
            f"{score}, {lead}; found {len(selected)}"
        )
    return (
        float(selected.iloc[0]["stratified_ci_low"]),
        float(selected.iloc[0]["stratified_ci_high"]),
    )


def build_table(replicates: int, seed: int) -> pd.DataFrame:
    primary = pd.read_csv(PRIMARY_INTERVALS)
    reconstruction = pd.read_csv(RECONSTRUCTION_INTERVALS)
    extension = pd.read_csv(EXTENSION_METRICS, dtype={"GAGE_ID": str})
    extension["GAGE_ID"] = normalize_gage(extension["GAGE_ID"])
    extension = extension[
        extension["threshold_name"].eq("Q10")
        & extension["target"].eq("onset")
        & extension["lead_days"].isin([105, 120])
    ]

    rows: list[dict[str, object]] = []
    for score in PRACTICAL_SCORES:
        paired = paired_scores(
            extension,
            "hydrograph_analog",
            "seasonal_climatology_path",
            score,
            ("threshold_name", "target"),
        )
        for lead, group in paired.groupby("lead_days", sort=True):
            candidate_score, reference_score, skill = point_summary(group)
            fixed_low, fixed_high = primary_lookup(
                primary,
                "hydrograph_analog",
                "seasonal_climatology_path",
                score,
                int(lead),
            )
            region_low, region_high = two_stage_interval(
                group,
                reference_score,
                replicates,
                seed + 41000 + int(lead) + sum(map(
                    ord,
                    "hydrograph_analogseasonal_climatology_path" + score,
                )),
            )
            rows.append({
                "comparison": "Analog vs seasonal trajectory climatology",
                "candidate": "hydrograph_analog",
                "reference": "seasonal_climatology_path",
                "lead_days": int(lead),
                "score": score,
                "basins": int(len(group)),
                "cases": int(group["n_candidate"].sum()),
                "candidate_score": candidate_score,
                "reference_score": reference_score,
                "relative_skill_percent": skill,
                "fixed_strata_ci_low": fixed_low,
                "fixed_strata_ci_high": fixed_high,
                "region_resampled_ci_low": float(region_low),
                "region_resampled_ci_high": float(region_high),
            })

    review = pd.read_csv(REVIEW_METRICS, dtype={"GAGE_ID": str})
    review["GAGE_ID"] = normalize_gage(review["GAGE_ID"])
    review = review[
        review["aggregation"].eq("basin") & review["lead_days"].eq(120)
    ]
    keys = ["GAGE_ID", "spatial_group", "lead_days", "score"]
    left = review[review["model"].eq("hydrograph_analog")][
        keys + ["n", "score_value"]
    ]
    right = review[review["model"].eq("seasonal_rank_reconstruction")][
        keys + ["n", "score_value"]
    ]
    residual = left.merge(right, on=keys, suffixes=("_candidate", "_reference"))
    for score, group in residual.groupby("score", sort=True):
        if score not in RESIDUAL_SCORES:
            continue
        candidate_score, reference_score, skill = point_summary(group)
        fixed_low, fixed_high = primary_lookup(
            reconstruction,
            "hydrograph_analog",
            "seasonal_rank_reconstruction",
            score,
            120,
        )
        region_low, region_high = two_stage_interval(
            group,
            reference_score,
            replicates,
            seed + 42000 + sum(map(ord, score)),
        )
        rows.append({
            "comparison": "Intact paths vs seasonal rank reconstruction",
            "candidate": "hydrograph_analog",
            "reference": "seasonal_rank_reconstruction",
            "lead_days": 120,
            "score": score,
            "basins": int(len(group)),
            "cases": int(group["n_candidate"].sum()),
            "candidate_score": candidate_score,
            "reference_score": reference_score,
            "relative_skill_percent": skill,
            "fixed_strata_ci_low": fixed_low,
            "fixed_strata_ci_high": fixed_high,
            "region_resampled_ci_low": float(region_low),
            "region_resampled_ci_high": float(region_high),
        })

    result = pd.DataFrame(rows)
    result["fixed_strata_resolved"] = result["fixed_strata_ci_low"].gt(0)
    result["region_resampled_resolved"] = result[
        "region_resampled_ci_low"
    ].gt(0)
    order = {
        "event_brier": 0,
        "timing_crps": 1,
        "joint_onset_rps": 1,
        "conditional_onset_rps": 2,
        "duration_crps": 3,
        "deficit_crps": 4,
    }
    result["_order"] = result["score"].map(order)
    return result.sort_values(
        ["comparison", "lead_days", "_order"]
    ).drop(columns="_order").reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replicates", type=int, default=None)
    args = parser.parse_args()
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    replicates = int(args.replicates or config["bootstrap_replicates"])
    seed = int(config["random_seed"])
    result = build_table(replicates, seed)
    atomic_csv(result, OUTPUT)

    practical_deficit = result[
        result["comparison"].str.startswith("Analog")
        & result["score"].eq("deficit_crps")
    ]
    residual = result[
        result["comparison"].str.startswith("Intact")
    ]
    receipt = {
        "stage": 39,
        "status": "complete",
        "primary_bootstrap_replicates": int(config["bootstrap_replicates"]),
        "sensitivity_bootstrap_replicates": replicates,
        "primary_scheme": "basins resampled within nine fixed ecoregion strata",
        "sensitivity_scheme": "basins and nine ecoregions resampled in two stages",
        "rows": int(len(result)),
        "practical_deficit_region_resampled_resolved": {
            str(int(row.lead_days)): bool(row.region_resampled_resolved)
            for row in practical_deficit.itertuples()
        },
        "residual_region_resampled_resolved": {
            row.score: bool(row.region_resampled_resolved)
            for row in residual.itertuples()
        },
    }
    atomic_json(receipt, RECEIPT)
    print(result.to_string(index=False))
    print(f"Wrote {OUTPUT}")
    print(f"Wrote {RECEIPT}")


if __name__ == "__main__":
    main()
