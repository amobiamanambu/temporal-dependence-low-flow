#!/usr/bin/env python3
"""Run non-ML analyses requested during review of the low-flow forecast paper.

The stage adds three checks without changing the frozen forecast:

1. A seasonal rank reconstruction that preserves every daily forecast value
   but restores cross-day member identity with a season-only historical
   template (a Schaake-shuffle-type middle rung).
2. A timing score conditional on an observed threshold crossing, which
   separates timing placement from the occurrence component embedded in the
   censored first-onset score.
3. Leave-one-water-year-out estimates and stratified basin bootstrap
   intervals, which avoid treating nine ecoregions as a large cluster sample.

Run from the project root:

    python temporal_coherence_paper/scripts/36_run_reviewer_strengthening.py --workers 4

The stage is restartable. Each completed basin is retained in a separate
compressed file before continental tables are assembled.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd


PAPER_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = PAPER_ROOT.parent
BENCHMARK_ROOT = PROJECT_ROOT / "lowflow_forecast_benchmark"
BENCHMARK_SCRIPTS = BENCHMARK_ROOT / "scripts"
if str(BENCHMARK_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_SCRIPTS))

from lib.common import (  # noqa: E402
    all_accepted_inventory,
    atomic_csv,
    atomic_json,
    ensemble_crps,
    load_benchmark_config,
    load_daily,
    water_year,
)
from lib.extended_120 import (  # noqa: E402
    MEMBERS,
    SEASON_FEATURES,
    analog_paths_120,
    extension_cases,
)
from lib.extended_trajectory import (  # noqa: E402
    HYDROGRAPH_FEATURES,
    discrete_crps,
    first_event_time,
    shuffle_members_by_day,
)


BASE_OUTPUT = PAPER_ROOT / "reviewer_strengthening"
EXT_ROOT = BENCHMARK_ROOT / "results" / "10_extension_120"
OLD_ROOT = BENCHMARK_ROOT / "results" / "09_extended_forecast"
LEADS = (30, 45, 60, 90, 105, 120)
MODELS = (
    "hydrograph_analog",
    "seasonal_rank_reconstruction",
    "independent_daily_shuffle",
)
SCORES = (
    "event_brier",
    "joint_onset_rps",
    "conditional_onset_rps",
    "duration_crps",
    "deficit_crps",
)
PRIMARY_ARCHIVED_SCORES = (
    "event_brier",
    "timing_crps",
    "duration_crps",
    "deficit_crps",
)


def normalize_gage(values: pd.Series) -> pd.Series:
    return values.astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(8)


def seasonal_rank_reconstruction(
    marginal_paths: np.ndarray,
    seasonal_templates: np.ndarray,
) -> np.ndarray:
    """Impose template ranks while preserving each daily marginal exactly."""
    if marginal_paths.shape != seasonal_templates.shape:
        raise ValueError("marginal and template path arrays must have equal shape")
    sorted_values = np.sort(marginal_paths, axis=1)
    template_order = np.argsort(seasonal_templates, axis=1, kind="stable")
    reconstructed = np.empty_like(marginal_paths)
    np.put_along_axis(reconstructed, template_order, sorted_values, axis=1)
    return reconstructed


def conditional_onset_rps(
    member_time: np.ndarray,
    observed_time: np.ndarray,
    horizon: int,
) -> np.ndarray:
    """Ranked probability score for onset day conditional on occurrence.

    Scores are returned only for cases with an observed event and at least one
    forecast member that crosses within the window. Forecast member times are
    renormalized over crossing members, removing forecast event probability
    from the conditional timing distribution.
    """
    days = np.arange(1, horizon + 1)
    member_event = member_time <= horizon
    event_members = member_event.sum(axis=1)
    numerator = np.sum(
        member_time[:, :, None] <= days[None, None, :], axis=1
    ).astype(float)
    conditional_cdf = np.divide(
        numerator,
        event_members[:, None],
        out=np.full_like(numerator, np.nan, dtype=float),
        where=event_members[:, None] > 0,
    )
    observed_cdf = observed_time[:, None] <= days[None, :]
    score = np.mean((conditional_cdf - observed_cdf) ** 2, axis=1)
    return np.where(
        (observed_time <= horizon) & (event_members > 0), score, np.nan
    )


def score_arrays(
    paths: np.ndarray,
    truth: np.ndarray,
    threshold: float,
    lead: int,
) -> dict[str, np.ndarray]:
    forecast = paths[:, :, :lead]
    observed = truth[:, :lead]
    observed_time = first_event_time(observed, threshold, "onset")
    member_time = first_event_time(forecast, threshold, "onset")
    observed_event = observed_time <= lead
    event_probability = np.mean(member_time <= lead, axis=1)
    observed_duration = np.sum(observed <= threshold, axis=1).astype(float)
    member_duration = np.sum(forecast <= threshold, axis=2).astype(float)
    observed_deficit = np.sum(np.maximum(threshold - observed, 0), axis=1)
    member_deficit = np.sum(np.maximum(threshold - forecast, 0), axis=2)
    return {
        "event_brier": (event_probability - observed_event.astype(float)) ** 2,
        "joint_onset_rps": discrete_crps(member_time, observed_time, lead),
        "conditional_onset_rps": conditional_onset_rps(
            member_time, observed_time, lead
        ),
        "duration_crps": ensemble_crps(member_duration, observed_duration),
        "deficit_crps": ensemble_crps(member_deficit, observed_deficit),
    }


def eligible_inventory(max_basins: int | None = None) -> pd.DataFrame:
    metrics = pd.read_csv(
        EXT_ROOT / "extension_120_metrics.csv.gz", dtype={"GAGE_ID": str}
    )
    valid = metrics[
        metrics["model"].eq("hydrograph_analog")
        & metrics["threshold_name"].eq("Q10")
        & metrics["target"].eq("onset")
    ].copy()
    valid_ids = set(normalize_gage(valid["GAGE_ID"]))
    inventory = all_accepted_inventory().copy()
    inventory["GAGE_ID"] = normalize_gage(inventory["GAGE_ID"])
    inventory = inventory[inventory["GAGE_ID"].isin(valid_ids)]
    inventory = inventory.sort_values("GAGE_ID").reset_index(drop=True)
    if max_basins is not None:
        inventory = inventory.head(int(max_basins)).copy()
    return inventory


def basin_worker(task: tuple[dict, dict, str]) -> dict:
    row, config, output_name = task
    output_path = Path(output_name)
    gage = str(row["GAGE_ID"]).zfill(8)
    try:
        frame = load_daily(row["file"])
        fit_end = pd.Timestamp(config["fit_end"])
        threshold_end = pd.Timestamp(config["threshold_reference_end"])
        qfit = frame.loc[
            frame["date"].le(fit_end) & frame["q_mm_day"].gt(0), "q_mm_day"
        ]
        if qfit.empty:
            raise ValueError("no_positive_training_scale")
        scale = float(qfit.median())
        history = frame.loc[
            frame["date"].le(threshold_end), "q_mm_day"
        ] / scale
        threshold = float(history.quantile(0.10))
        if not np.isfinite(threshold) or threshold <= 0:
            raise ValueError("invalid_Q10_threshold")

        cases = extension_cases(frame, scale, config)
        train = cases[
            cases["future_date"].le(pd.Timestamp(config["calibration_end"]))
        ].copy()
        test = cases[
            cases["date"].ge(pd.Timestamp(config["evaluation_start"]))
            & cases["future_date"].le(pd.Timestamp(config["evaluation_end"]))
        ].copy().reset_index(drop=True)
        if len(train) < int(config["minimum_trajectory_fit_cases"]):
            raise ValueError("insufficient_training_initializations")
        if len(test) < int(config["minimum_trajectory_evaluation_cases"]):
            raise ValueError("insufficient_test_initializations")

        coherent_all = analog_paths_120(
            train,
            test,
            HYDROGRAPH_FEATURES,
            scale_by_initial_q=True,
            members=MEMBERS,
        )
        seasonal_template_all = analog_paths_120(
            train,
            test,
            SEASON_FEATURES,
            scale_by_initial_q=False,
            members=MEMBERS,
        )
        reconstructed_all = seasonal_rank_reconstruction(
            coherent_all, seasonal_template_all
        )
        seed = int(config["random_seed"] + int(gage[-5:]) + 24000)
        shuffled_all = shuffle_members_by_day(coherent_all, seed)
        eligible = test["q"].gt(threshold).to_numpy(bool)
        coherent = coherent_all[eligible]
        reconstructed = reconstructed_all[eligible]
        shuffled = shuffled_all[eligible]
        test = test.loc[eligible].reset_index(drop=True)
        del coherent_all, reconstructed_all, shuffled_all, seasonal_template_all

        marginal_error_reconstructed = float(np.max(np.abs(
            np.sort(coherent, axis=1) - np.sort(reconstructed, axis=1)
        )))
        marginal_error_shuffled = float(np.max(np.abs(
            np.sort(coherent, axis=1) - np.sort(shuffled, axis=1)
        )))

        truth = test[[f"q_h{day}" for day in range(1, 121)]].to_numpy(
            np.float32
        )
        years = water_year(test["date"])
        rows: list[dict] = []
        endpoint_error = 0.0
        duration_mean_error = 0.0
        deficit_mean_error = 0.0
        model_paths = {
            "hydrograph_analog": coherent,
            "seasonal_rank_reconstruction": reconstructed,
            "independent_daily_shuffle": shuffled,
        }
        for lead in LEADS:
            observed_event = np.any(truth[:, :lead] <= threshold, axis=1)
            events = int(observed_event.sum())
            nonevents = int((~observed_event).sum())
            if events < int(config["minimum_events"]):
                continue
            if nonevents < int(config["minimum_nonevents"]):
                continue

            coherent_endpoint = np.mean(
                coherent[:, :, lead - 1] <= threshold, axis=1
            )
            for candidate in (reconstructed, shuffled):
                endpoint_error = max(
                    endpoint_error,
                    float(np.max(np.abs(
                        coherent_endpoint
                        - np.mean(candidate[:, :, lead - 1] <= threshold, axis=1)
                    ))),
                )
                duration_mean_error = max(
                    duration_mean_error,
                    float(np.max(np.abs(
                        np.mean(np.sum(coherent[:, :, :lead] <= threshold, axis=2), axis=1)
                        - np.mean(np.sum(candidate[:, :, :lead] <= threshold, axis=2), axis=1)
                    ))),
                )
                deficit_mean_error = max(
                    deficit_mean_error,
                    float(np.max(np.abs(
                        np.mean(np.sum(
                            np.maximum(threshold - coherent[:, :, :lead], 0), axis=2
                        ), axis=1)
                        - np.mean(np.sum(
                            np.maximum(threshold - candidate[:, :, :lead], 0), axis=2
                        ), axis=1)
                    ))),
                )

            model_arrays = {
                model: score_arrays(paths, truth, threshold, lead)
                for model, paths in model_paths.items()
            }
            common_conditional_support = np.logical_and.reduce([
                np.isfinite(arrays["conditional_onset_rps"])
                for arrays in model_arrays.values()
            ])
            for arrays in model_arrays.values():
                arrays["conditional_onset_rps"] = np.where(
                    common_conditional_support,
                    arrays["conditional_onset_rps"],
                    np.nan,
                )
            for model, arrays in model_arrays.items():
                for score, values in arrays.items():
                    finite = np.isfinite(values)
                    rows.append({
                        "aggregation": "basin",
                        "GAGE_ID": gage,
                        "spatial_group": str(row.get("spatial_group", "Unknown")),
                        "water_year": np.nan,
                        "lead_days": int(lead),
                        "model": model,
                        "score": score,
                        "n": int(finite.sum()),
                        "events": events,
                        "score_value": float(np.mean(values[finite])),
                    })
                    for year in np.unique(years):
                        selected = finite & (years == year)
                        if not selected.any():
                            continue
                        rows.append({
                            "aggregation": "water_year",
                            "GAGE_ID": gage,
                            "spatial_group": str(row.get("spatial_group", "Unknown")),
                            "water_year": int(year),
                            "lead_days": int(lead),
                            "model": model,
                            "score": score,
                            "n": int(selected.sum()),
                            "events": int(observed_event[years == year].sum()),
                            "score_value": float(np.mean(values[selected])),
                        })

        if not rows:
            raise ValueError("no_balanced_primary_lead")
        result = pd.DataFrame(rows)
        result["max_daily_marginal_error"] = max(
            marginal_error_reconstructed, marginal_error_shuffled
        )
        result["max_endpoint_probability_error"] = endpoint_error
        result["max_additive_duration_mean_error"] = duration_mean_error
        result["max_additive_deficit_mean_error"] = deficit_mean_error
        atomic_csv(result, output_path, compression="gzip")
        return {"GAGE_ID": gage, "status": "complete"}
    except Exception as error:
        failure_path = output_path.with_name(f"{gage}_failure.json")
        atomic_json({"GAGE_ID": gage, "reason": repr(error)}, failure_path)
        output_path.unlink(missing_ok=True)
        return {"GAGE_ID": gage, "status": "excluded", "reason": repr(error)}


def stratified_basin_interval(
    paired: pd.DataFrame,
    replicates: int,
    seed: int,
) -> tuple[float, float]:
    """Bootstrap basins within fixed ecoregion strata."""
    rng = np.random.default_rng(seed)
    regional = [group for _, group in paired.groupby("spatial_group", dropna=False)]
    numerator = np.zeros(replicates, dtype=float)
    denominator = np.zeros(replicates, dtype=float)
    for group in regional:
        candidate = group["score_value_candidate"].to_numpy(float)
        reference = group["score_value_reference"].to_numpy(float)
        weights = group["n_candidate"].to_numpy(float)
        indices = rng.integers(0, len(group), size=(replicates, len(group)))
        numerator += np.sum(
            (reference[indices] - candidate[indices]) * weights[indices], axis=1
        )
        denominator += np.sum(reference[indices] * weights[indices], axis=1)
    relative = 100.0 * numerator / denominator
    return tuple(np.quantile(relative, [0.025, 0.975]))


def paired_summary(
    metrics: pd.DataFrame,
    candidate: str,
    reference: str,
    replicates: int,
    seed: int,
) -> pd.DataFrame:
    keys = ["GAGE_ID", "spatial_group", "lead_days", "score"]
    left = metrics[metrics["model"].eq(candidate)][keys + ["n", "score_value"]]
    right = metrics[metrics["model"].eq(reference)][keys + ["n", "score_value"]]
    paired = left.merge(right, on=keys, suffixes=("_candidate", "_reference"))
    rows = []
    for (lead, score), group in paired.groupby(["lead_days", "score"], sort=True):
        weights = group["n_candidate"].to_numpy(float)
        candidate_score = float(np.average(
            group["score_value_candidate"], weights=weights
        ))
        reference_score = float(np.average(
            group["score_value_reference"], weights=weights
        ))
        low, high = stratified_basin_interval(
            group,
            replicates,
            seed + int(lead) + sum(map(ord, candidate + reference + score)),
        )
        rows.append({
            "candidate": candidate,
            "reference": reference,
            "lead_days": int(lead),
            "score": score,
            "basins": int(len(group)),
            "cases": int(weights.sum()),
            "candidate_score": candidate_score,
            "reference_score": reference_score,
            "relative_skill_percent": (
                100.0 * (reference_score - candidate_score) / reference_score
            ),
            "stratified_ci_low": float(low),
            "stratified_ci_high": float(high),
            "basin_fraction_improved": float(
                group["score_value_candidate"].lt(
                    group["score_value_reference"]
                ).mean()
            ),
        })
    return pd.DataFrame(rows)


def leave_one_year_out(yearly: pd.DataFrame) -> pd.DataFrame:
    keys = ["GAGE_ID", "spatial_group", "water_year", "lead_days", "score"]
    rows = []
    for candidate, reference in (
        ("hydrograph_analog", "independent_daily_shuffle"),
        ("seasonal_rank_reconstruction", "independent_daily_shuffle"),
        ("hydrograph_analog", "seasonal_rank_reconstruction"),
    ):
        left = yearly[yearly["model"].eq(candidate)][
            keys + ["n", "score_value"]
        ]
        right = yearly[yearly["model"].eq(reference)][
            keys + ["n", "score_value"]
        ]
        paired = left.merge(right, on=keys, suffixes=("_candidate", "_reference"))
        for (lead, score), group in paired.groupby(["lead_days", "score"]):
            for omitted in sorted(group["water_year"].dropna().unique()):
                kept = group[group["water_year"].ne(omitted)]
                weights = kept["n_candidate"].to_numpy(float)
                candidate_score = float(np.average(
                    kept["score_value_candidate"], weights=weights
                ))
                reference_score = float(np.average(
                    kept["score_value_reference"], weights=weights
                ))
                rows.append({
                    "candidate": candidate,
                    "reference": reference,
                    "lead_days": int(lead),
                    "score": score,
                    "omitted_water_year": int(omitted),
                    "basin_year_cells": int(len(kept)),
                    "cases": int(weights.sum()),
                    "relative_skill_percent": (
                        100.0 * (reference_score - candidate_score) / reference_score
                    ),
                })
    return pd.DataFrame(rows)


def archived_primary_intervals(
    replicates: int,
    seed: int,
) -> pd.DataFrame:
    old = pd.read_csv(
        OLD_ROOT / "extended_confirmation_metrics.csv.gz", dtype={"GAGE_ID": str}
    )
    ext = pd.read_csv(
        EXT_ROOT / "extension_120_metrics.csv.gz", dtype={"GAGE_ID": str}
    )
    frames = []
    for source_name, frame, leads in (
        ("1_to_90_confirmation", old, (30, 45, 60, 90)),
        ("105_120_extension", ext, (105, 120)),
    ):
        selected = frame[
            frame["threshold_name"].eq("Q10")
            & frame["target"].eq("onset")
            & frame["lead_days"].isin(leads)
        ]
        for reference in (
            "hydrograph_marginal_shuffle",
            "seasonal_climatology_path",
            "constant_persistence_path",
        ):
            for score in PRIMARY_ARCHIVED_SCORES:
                keys = ["GAGE_ID", "spatial_group", "lead_days"]
                left = selected[selected["model"].eq("hydrograph_analog")][
                    keys + ["n", score]
                ].dropna(subset=[score])
                right = selected[selected["model"].eq(reference)][
                    keys + [score]
                ].dropna(subset=[score])
                paired = left.merge(
                    right, on=keys, suffixes=("_candidate", "_reference")
                )
                paired = paired.rename(columns={
                    "n": "n_candidate",
                    f"{score}_candidate": "score_value_candidate",
                    f"{score}_reference": "score_value_reference",
                })
                for lead, group in paired.groupby("lead_days"):
                    weights = group["n_candidate"].to_numpy(float)
                    candidate_score = float(np.average(
                        group["score_value_candidate"], weights=weights
                    ))
                    reference_score = float(np.average(
                        group["score_value_reference"], weights=weights
                    ))
                    low, high = stratified_basin_interval(
                        group,
                        replicates,
                        seed + int(lead) + sum(map(ord, reference + score)),
                    )
                    frames.append({
                        "archive": source_name,
                        "candidate": "hydrograph_analog",
                        "reference": reference,
                        "score": score,
                        "lead_days": int(lead),
                        "basins": int(len(group)),
                        "cases": int(weights.sum()),
                        "candidate_score": candidate_score,
                        "reference_score": reference_score,
                        "relative_skill_percent": (
                            100.0 * (reference_score - candidate_score)
                            / reference_score
                        ),
                        "stratified_ci_low": float(low),
                        "stratified_ci_high": float(high),
                        "basin_fraction_improved": float(
                            group["score_value_candidate"].lt(
                                group["score_value_reference"]
                            ).mean()
                        ),
                    })
    return pd.DataFrame(frames)


def assemble(output_root: Path, inventory: pd.DataFrame, config: dict) -> dict:
    by_basin = output_root / "by_basin"
    files = sorted(by_basin.glob("*_review_metrics.csv.gz"))
    if not files:
        raise RuntimeError("No completed reviewer-strengthening basin files found")
    frames = [pd.read_csv(path, dtype={"GAGE_ID": str}) for path in files]
    metrics = pd.concat(frames, ignore_index=True)
    metrics["GAGE_ID"] = normalize_gage(metrics["GAGE_ID"])
    atomic_csv(metrics, output_root / "reviewer_strengthening_metrics.csv.gz",
               compression="gzip")

    basin = metrics[metrics["aggregation"].eq("basin")].copy()
    yearly = metrics[metrics["aggregation"].eq("water_year")].copy()
    comparison_frames = []
    for candidate, reference in (
        ("hydrograph_analog", "independent_daily_shuffle"),
        ("seasonal_rank_reconstruction", "independent_daily_shuffle"),
        ("hydrograph_analog", "seasonal_rank_reconstruction"),
    ):
        comparison_frames.append(paired_summary(
            basin,
            candidate,
            reference,
            int(config["bootstrap_replicates"]),
            int(config["random_seed"] + 36000),
        ))
    comparisons = pd.concat(comparison_frames, ignore_index=True)

    recovery = comparisons[
        comparisons["reference"].eq("independent_daily_shuffle")
    ].pivot_table(
        index=["lead_days", "score"],
        columns="candidate",
        values=["candidate_score", "reference_score"],
        aggfunc="first",
    )
    recovery.columns = ["__".join(column) for column in recovery.columns]
    recovery = recovery.reset_index()
    shuffle_score = recovery[
        "reference_score__hydrograph_analog"
    ]
    coherent_score = recovery["candidate_score__hydrograph_analog"]
    reconstructed_score = recovery[
        "candidate_score__seasonal_rank_reconstruction"
    ]
    recovery["fraction_of_coherent_advantage_recovered"] = np.divide(
        shuffle_score - reconstructed_score,
        shuffle_score - coherent_score,
        out=np.full(len(recovery), np.nan),
        where=np.abs(shuffle_score - coherent_score) > 0,
    )
    atomic_csv(comparisons, output_root / "reconstruction_comparisons.csv")
    atomic_csv(recovery, output_root / "reconstruction_recovery_fraction.csv")

    loo = leave_one_year_out(yearly)
    atomic_csv(loo, output_root / "leave_one_water_year_out.csv")
    archived = archived_primary_intervals(
        int(config["bootstrap_replicates"]),
        int(config["random_seed"] + 37000),
    )
    atomic_csv(archived, output_root / "archived_stratified_basin_intervals.csv")

    screen = pd.read_csv(OLD_ROOT / "screen_comparisons.csv")
    architecture = screen[
        screen["family"].eq("coherence")
        & screen["threshold_name"].eq("Q10")
        & screen["target"].eq("onset")
        & screen["lead_days"].isin([30, 45, 60, 90])
        & screen["score"].isin(PRIMARY_ARCHIVED_SCORES)
    ].copy()
    architecture_summary = architecture.groupby(
        ["candidate", "reference"], as_index=False
    ).agg(
        comparisons=("weighted_improvement", "size"),
        positive_comparisons=("weighted_improvement", lambda x: int((x > 0).sum())),
        intervals_above_zero=("bootstrap_ci_low", lambda x: int((x > 0).sum())),
        mean_basin_fraction_improved=("basin_fraction_improved", "mean"),
        minimum_basins=("basins", "min"),
        maximum_basins=("basins", "max"),
    )
    atomic_csv(
        architecture_summary,
        output_root / "development_architecture_replication.csv",
    )

    sample_rows = [
        ["Accepted Tier 1/Tier 2 inventory", 5227,
         "Parent quality-controlled inventory"],
        ["Development panel", 192,
         "Used for architecture and rule development; excluded from confirmation"],
        ["Confirmation attempted", 5035,
         "Accepted inventory minus development sample"],
        ["Valid through 90 days", 3835,
         "At least one valid archived trajectory score"],
        ["Successful 120-day-path extension processing", 3733,
         "Complete 120-day observation path and at least one scorable threshold-target combination at any evaluated window"],
        ["Other-target-only extension support", 331,
         "Supported another threshold or target but no estimable primary Q10-onset score"],
        ["Primary Q10 forecast cohort", 3402,
         "Estimable hydrograph-analog Q10-onset score at one or more forecast windows"],
        ["Primary paired 120-day inference", 3333,
         "At least 10 events and 10 non-events for all primary path scores"],
        ["Forecast-level 30–120-day Zenodo release", 3402,
         "434,198 common-support forecast initializations at each of six released windows"],
        ["Seasonal reconstruction sensitivity", int(
            basin["GAGE_ID"].nunique()
        ), "Completed reviewer-strengthening analysis"],
    ]
    accounting = pd.DataFrame(
        sample_rows, columns=["analysis_population", "basins", "reason"]
    )
    atomic_csv(accounting, output_root / "sample_accounting.csv")

    invariance_columns = [
        "max_daily_marginal_error",
        "max_endpoint_probability_error",
        "max_additive_duration_mean_error",
        "max_additive_deficit_mean_error",
    ]
    invariance = {
        column: float(pd.to_numeric(metrics[column], errors="coerce").max())
        for column in invariance_columns
    }
    receipt = {
        "stage": 36,
        "status": "complete",
        "eligible_basins": int(len(inventory)),
        "completed_basins": int(basin["GAGE_ID"].nunique()),
        "leads_days": list(LEADS),
        "models": list(MODELS),
        "scores": list(SCORES),
        "bootstrap": (
            "basins resampled within nine fixed GAGES-II ecoregion strata"
        ),
        "bootstrap_replicates": int(config["bootstrap_replicates"]),
        "temporal_check": "leave one held-out water year out",
        "invariance_checks": invariance,
    }
    atomic_json(receipt, output_root / "STAGE36_SUCCESS.json")
    return receipt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--max-basins", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--output-label",
        default="full",
        help="Subfolder under reviewer_strengthening (default: full)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = BASE_OUTPUT / str(args.output_label)
    by_basin = output_root / "by_basin"
    by_basin.mkdir(parents=True, exist_ok=True)
    inventory = eligible_inventory(args.max_basins)
    config = load_benchmark_config()
    tasks = []
    retained = 0
    for row in inventory.to_dict("records"):
        gage = str(row["GAGE_ID"]).zfill(8)
        output = by_basin / f"{gage}_review_metrics.csv.gz"
        failure = by_basin / f"{gage}_failure.json"
        if not args.force and (output.exists() or failure.exists()):
            retained += 1
        else:
            tasks.append((row, config, str(output)))
    print(
        f"Stage 36: {len(inventory):,} eligible basins; {retained:,} retained; "
        f"{len(tasks):,} to process",
        flush=True,
    )

    results = []
    if int(args.workers) <= 1:
        for number, task in enumerate(tasks, 1):
            results.append(basin_worker(task))
            if number % 20 == 0 or number == len(tasks):
                print(f"Completed {number:,}/{len(tasks):,}", flush=True)
    else:
        try:
            executor = ProcessPoolExecutor(max_workers=int(args.workers))
        except (PermissionError, OSError):
            print("Process workers unavailable; using thread workers", flush=True)
            executor = ThreadPoolExecutor(max_workers=int(args.workers))
        with executor:
            futures = [executor.submit(basin_worker, task) for task in tasks]
            for number, future in enumerate(as_completed(futures), 1):
                results.append(future.result())
                if number % 20 == 0 or number == len(futures):
                    print(f"Completed {number:,}/{len(futures):,}", flush=True)

    receipt = assemble(output_root, inventory, config)
    receipt["new_successes"] = int(sum(
        item.get("status") == "complete" for item in results
    ))
    receipt["new_exclusions"] = int(sum(
        item.get("status") == "excluded" for item in results
    ))
    atomic_json(receipt, output_root / "STAGE36_SUCCESS.json")
    print(json.dumps(receipt, indent=2), flush=True)


if __name__ == "__main__":
    main()
