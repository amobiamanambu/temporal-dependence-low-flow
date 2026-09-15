#!/usr/bin/env python3
"""Compare extended-range candidates and freeze the full-confirmation decision."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from lib.common import RESULTS_ROOT, atomic_csv, atomic_json, load_benchmark_config  # noqa: E402


SCORES = [
    "endpoint_brier", "event_brier", "timing_crps", "deficit_crps",
    "duration_crps", "minimum_flow_crps",
]
ARCHITECTURE_SHUFFLES = {
    "flow_season_analog": "flow_season_marginal_shuffle",
    "hydrograph_analog": "hydrograph_marginal_shuffle",
    "hydroclimate_analog": "hydroclimate_marginal_shuffle",
    "ridge_residual_block": "ridge_residual_shuffle",
    "extra_trees_path": "extra_trees_marginal_shuffle",
}
PATH_ARCHITECTURES = list(ARCHITECTURE_SHUFFLES)
PATH_BASELINES = ["seasonal_climatology_path", "constant_persistence_path"]
PAIR_SPECS = (
    [(candidate, shuffle, "coherence")
     for candidate, shuffle in ARCHITECTURE_SHUFFLES.items()]
    + [(candidate, reference, "path_baseline")
       for candidate in PATH_ARCHITECTURES for reference in PATH_BASELINES]
    + [(candidate, reference, "architecture")
       for candidate in PATH_ARCHITECTURES for reference in PATH_ARCHITECTURES
       if candidate != reference]
    + [("perfect_forcing_oracle", reference, "perfect_forcing_ceiling")
       for reference in PATH_ARCHITECTURES]
    + [(candidate, direct, "direct_target")
       for candidate in PATH_ARCHITECTURES
       for direct in ("direct_endpoint_logistic", "direct_event_logistic")]
    + [(direct, candidate, "direct_target")
       for direct in ("direct_endpoint_logistic", "direct_event_logistic")
       for candidate in PATH_ARCHITECTURES]
    + [(direct, baseline, "direct_baseline")
       for direct in ("direct_endpoint_logistic", "direct_event_logistic")
       for baseline in PATH_BASELINES]
)


def hierarchical_ci(group: pd.DataFrame, replicates: int, seed: int) -> tuple[float, float]:
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


def compare(metrics: pd.DataFrame, config: dict) -> pd.DataFrame:
    keys = [
        "GAGE_ID", "spatial_group", "threshold_quantile", "threshold_name",
        "lead_days", "target",
    ]
    rows = []
    for candidate, reference, family in PAIR_SPECS:
        left = metrics[metrics["model"].eq(candidate)]
        right = metrics[metrics["model"].eq(reference)]
        if left.empty or right.empty:
            continue
        for score in SCORES:
            a = left[keys + ["n", score]].dropna(subset=[score])
            b = right[keys + [score]].dropna(subset=[score])
            # Only the candidate table carries the common case count.  Because
            # ``n`` is absent from ``b``, pandas correctly keeps it as ``n``
            # rather than creating a misleading ``n_candidate`` suffix.
            paired = a.merge(b, on=keys, suffixes=("_candidate", "_reference"))
            if paired.empty:
                continue
            paired["improvement"] = paired[f"{score}_reference"] - paired[f"{score}_candidate"]
            if "n_candidate" in paired:
                paired["n"] = paired["n_candidate"]
            for (quantile, name, lead, target), group in paired.groupby(
                ["threshold_quantile", "threshold_name", "lead_days", "target"]
            ):
                low, high = hierarchical_ci(
                    group, int(config["bootstrap_replicates"]),
                    int(config["random_seed"] + lead + int(100 * quantile)
                        + sum(map(ord, candidate + reference + score + target))),
                )
                rows.append({
                    "family": family, "candidate": candidate, "reference": reference,
                    "score": score, "threshold_quantile": float(quantile),
                    "threshold_name": name, "lead_days": int(lead), "target": target,
                    "basins": int(len(group)), "cases": int(group["n"].sum()),
                    "weighted_improvement": float(np.average(
                        group["improvement"], weights=group["n"]
                    )),
                    "bootstrap_ci_low": float(low), "bootstrap_ci_high": float(high),
                    "basin_fraction_improved": float(group["improvement"].gt(0).mean()),
                })
    return pd.DataFrame(rows)


def summarize_pair(frame: pd.DataFrame) -> dict:
    if frame.empty:
        return {
            "comparisons": 0, "positive": 0, "significant_positive": 0,
            "mean_basin_fraction_improved": np.nan, "basins_min": 0, "basins_max": 0,
        }
    return {
        "comparisons": int(len(frame)),
        "positive": int(frame["weighted_improvement"].gt(0).sum()),
        "significant_positive": int(frame["bootstrap_ci_low"].gt(0).sum()),
        "mean_basin_fraction_improved": float(frame["basin_fraction_improved"].mean()),
        "basins_min": int(frame["basins"].min()),
        "basins_max": int(frame["basins"].max()),
    }


def main() -> None:
    config = load_benchmark_config()
    root = RESULTS_ROOT / "09_extended_forecast"
    path = root / "screen" / "extended_trajectory_metrics.csv.gz"
    if not path.exists():
        raise FileNotFoundError("Run 17_test_extended_trajectories.py first")
    metrics = pd.read_csv(path, dtype={"GAGE_ID": str})
    comparison = compare(metrics, config)
    atomic_csv(comparison, root / "screen_comparisons.csv")

    long_primary = comparison[
        comparison["threshold_name"].eq("Q10")
        & comparison["target"].eq("onset")
        & comparison["lead_days"].isin([30, 45, 60, 90])
    ]
    coherence_scores = [
        "event_brier", "timing_crps", "deficit_crps", "duration_crps",
        "minimum_flow_crps",
    ]
    architecture_rows = []
    for candidate, reference in ARCHITECTURE_SHUFFLES.items():
        selected = long_primary[
            long_primary["candidate"].eq(candidate)
            & long_primary["reference"].eq(reference)
            & long_primary["score"].isin(coherence_scores)
        ]
        coherence = summarize_pair(selected)
        baseline_summaries = {}
        for baseline in ("seasonal_climatology_path", "constant_persistence_path"):
            baseline_rows = long_primary[
                long_primary["candidate"].eq(candidate)
                & long_primary["reference"].eq(baseline)
                & long_primary["score"].isin(coherence_scores)
            ]
            baseline_summaries[baseline] = summarize_pair(baseline_rows)
        coherence_passes = bool(
            coherence["comparisons"] >= 15
            and coherence["positive"] / coherence["comparisons"] >= 0.75
            and coherence["significant_positive"] / coherence["comparisons"] >= 0.50
            and coherence["mean_basin_fraction_improved"] > 0.50
        )
        baselines_pass = all(
            summary["comparisons"] >= 15
            and summary["positive"] / summary["comparisons"] >= 0.70
            and summary["significant_positive"] / summary["comparisons"] >= 0.40
            and summary["mean_basin_fraction_improved"] > 0.50
            for summary in baseline_summaries.values()
        )
        head_to_head_rows = long_primary[
            long_primary["family"].eq("architecture")
            & long_primary["candidate"].eq(candidate)
            & long_primary["score"].isin(coherence_scores[:-1])
        ]
        head_to_head = summarize_pair(head_to_head_rows)
        summary = {
            "candidate": candidate,
            "shuffle_reference": reference,
            "coherence": coherence,
            "path_baselines": baseline_summaries,
            "head_to_head": head_to_head,
            "coherence_passes": coherence_passes,
            "path_baselines_pass": baselines_pass,
            "passes": bool(coherence_passes and baselines_pass),
        }
        architecture_rows.append(summary)
    ranked = sorted(
        architecture_rows,
        key=lambda row: (
            row["passes"], row["head_to_head"]["significant_positive"],
            row["head_to_head"]["positive"],
            row["coherence"]["significant_positive"],
            row["coherence"]["mean_basin_fraction_improved"]
            if np.isfinite(row["coherence"]["mean_basin_fraction_improved"]) else -1,
        ),
        reverse=True,
    )
    winner = ranked[0]["candidate"] if ranked and ranked[0]["passes"] else None
    direct_summaries = {}
    for candidate, score in (
        ("direct_endpoint_logistic", "endpoint_brier"),
        ("direct_event_logistic", "event_brier"),
    ):
        references = {}
        for reference in (*PATH_ARCHITECTURES, *PATH_BASELINES):
            selected = long_primary[
                long_primary["candidate"].eq(candidate)
                & long_primary["reference"].eq(reference)
                & long_primary["score"].eq(score)
            ]
            references[reference] = summarize_pair(selected)
        required = [
            references[reference] for reference in PATH_BASELINES
        ]
        passes = all(
            summary["comparisons"] >= 4
            and summary["positive"] >= 3
            and summary["significant_positive"] >= 2
            and summary["mean_basin_fraction_improved"] > 0.50
            for summary in required
        )
        direct_summaries[candidate] = {
            "score": score, "references": references, "passes": bool(passes)
        }
    oracle = long_primary[
        long_primary["candidate"].eq("perfect_forcing_oracle")
        & long_primary["reference"].eq(winner)
    ]
    oracle_summary = summarize_pair(oracle)
    oracle_summary["archive_acquisition_justified"] = bool(
        oracle_summary["comparisons"] >= 20
        and oracle_summary["positive"] / max(oracle_summary["comparisons"], 1) >= 0.60
        and oracle_summary["mean_basin_fraction_improved"] > 0.52
    )
    decision = {
        "status": "screen_complete",
        "primary_target": "Q10 onset from a weekly observed-dry initialization",
        "primary_extended_leads": [30, 45, 60, 90],
        "coherence_architectures": ranked,
        "winning_path_architecture": winner,
        "promote_to_independent_confirmation": bool(winner),
        "direct_target_models": direct_summaries,
        "perfect_forcing_oracle": oracle_summary,
        "weather_archive_acquisition_scientifically_justified": bool(
            oracle_summary["archive_acquisition_justified"]
        ),
        "future_observations_used_by_eligible_models": False,
        "manuscript_modified": False,
    }
    atomic_json(decision, root / "EXTENDED_SCREEN_DECISION.json")
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
