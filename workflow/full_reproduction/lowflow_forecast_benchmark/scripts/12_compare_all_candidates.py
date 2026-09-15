#!/usr/bin/env python3
"""Compare candidates with paired basin and water-year evidence and freeze decisions."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from lib.common import RESULTS_ROOT, atomic_csv, atomic_json, load_benchmark_config  # noqa: E402


def bootstrap_basin_improvement(group: pd.DataFrame, replicates: int, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    grouped = [values for _, values in group.groupby("spatial_group", dropna=False)]
    group_numerator = np.empty((replicates, len(grouped)), float)
    group_denominator = np.empty((replicates, len(grouped)), float)
    for column, values in enumerate(grouped):
        improvement = values["improvement"].to_numpy(float)
        weights = values["n"].to_numpy(float)
        indices = rng.integers(0, len(values), size=(replicates, len(values)))
        group_numerator[:, column] = np.sum(improvement[indices] * weights[indices], axis=1)
        group_denominator[:, column] = np.sum(weights[indices], axis=1)
    sampled_groups = rng.integers(0, len(grouped), size=(replicates, len(grouped)))
    row = np.arange(replicates)[:, None]
    output = (
        group_numerator[row, sampled_groups].sum(axis=1)
        / group_denominator[row, sampled_groups].sum(axis=1)
    )
    return tuple(np.quantile(output, [0.025, 0.975]))


def paired_table(metrics: pd.DataFrame, reference_name: str, config: dict,
                 score: str = "brier") -> pd.DataFrame:
    keys = ["GAGE_ID", "spatial_group", "threshold_quantile", "threshold_name", "lead_days"]
    reference = metrics[metrics["model"].eq(reference_name)][keys + ["n", score]]
    rows = []
    for method in sorted(set(metrics["model"]) - {reference_name}):
        candidate = metrics[metrics["model"].eq(method)][keys + [score]]
        paired = reference.merge(candidate, on=keys, suffixes=("_reference", "_candidate"))
        paired = paired[np.isfinite(paired[f"{score}_reference"]) & np.isfinite(paired[f"{score}_candidate"])]
        for (quantile, name, lead), group in paired.groupby(
            ["threshold_quantile", "threshold_name", "lead_days"]
        ):
            group = group.copy()
            group["improvement"] = group[f"{score}_reference"] - group[f"{score}_candidate"]
            low, high = bootstrap_basin_improvement(
                group, int(config["bootstrap_replicates"]),
                int(config["random_seed"] + lead + round(100 * quantile) + sum(map(ord, method))),
            )
            rows.append({
                "model": method, "reference_model": reference_name,
                "threshold_quantile": float(quantile), "threshold_name": name,
                "lead_days": int(lead), "score": score, "basins": int(len(group)),
                "cases": int(group["n"].sum()),
                "weighted_improvement": float(np.average(group["improvement"], weights=group["n"])),
                "bootstrap_ci_low": float(low), "bootstrap_ci_high": float(high),
                "basin_fraction_improved": float(group["improvement"].gt(0).mean()),
                "positive_means_candidate_is_better": True,
            })
    return pd.DataFrame(rows)


def endpoint_decision(comparison: pd.DataFrame, config: dict) -> dict:
    primary = comparison[
        comparison["threshold_quantile"].eq(config["primary_quantile"])
        & comparison["score"].eq("brier")
    ].copy()
    ranking = primary.groupby("model").agg(
        leads=("lead_days", "nunique"),
        positive_leads=("weighted_improvement", lambda x: int((x > 0).sum())),
        significant_positive_leads=("bootstrap_ci_low", lambda x: int((x > 0).sum())),
        mean_gain=("weighted_improvement", "mean"),
        mean_basin_fraction_improved=("basin_fraction_improved", "mean"),
    ).reset_index()
    day1 = primary[primary["lead_days"].eq(1)][["model", "weighted_improvement"]].rename(
        columns={"weighted_improvement": "day1_gain"}
    )
    ranking = ranking.merge(day1, on="model", how="left")
    rule = config["promotion"]
    ranking["passes_rule"] = (
        ranking["positive_leads"].ge(rule["minimum_positive_primary_leads"])
        & ranking["mean_gain"].ge(rule["minimum_mean_primary_brier_gain"])
        & ranking["mean_basin_fraction_improved"].ge(rule["minimum_mean_basin_fraction_improved"])
        & ranking["day1_gain"].fillna(-np.inf).ge(-rule["maximum_day1_brier_loss"])
    )
    ranking = ranking.sort_values(
        ["passes_rule", "significant_positive_leads", "mean_gain"], ascending=False
    )
    eligible = ranking[ranking["passes_rule"]]
    winner = str(eligible.iloc[0]["model"]) if len(eligible) else rule["primary_reference"]
    return {
        "winner": winner,
        "new_endpoint_method_promoted": bool(len(eligible)),
        "primary_reference": rule["primary_reference"],
        "ranking": ranking.to_dict(orient="records"),
        "rule_fixed_before_full_confirmation": True,
    }


def main() -> None:
    config = load_benchmark_config()
    endpoint_path = RESULTS_ROOT / "02_endpoint_candidates" / "endpoint_basin_metrics.csv.gz"
    if not endpoint_path.exists():
        raise FileNotFoundError("Run run_endpoint_candidate_suite.py first")
    endpoint = pd.read_csv(endpoint_path, dtype={"GAGE_ID": str})
    regional_path = RESULTS_ROOT / "03_regional" / "regional_basin_metrics.csv.gz"
    if regional_path.exists():
        endpoint = pd.concat([endpoint, pd.read_csv(regional_path, dtype={"GAGE_ID": str})],
                             ignore_index=True, sort=False)
    reference = config["promotion"]["primary_reference"]
    endpoint_comparison = paired_table(endpoint, reference, config, "brier")
    distribution_comparison = paired_table(endpoint, "empirical_heavytail", config,
                                           "twcrps_below_threshold")
    decision = {"status": "candidate_screen_complete",
                "endpoint": endpoint_decision(endpoint_comparison, config)}

    conformal_path = RESULTS_ROOT / "02_endpoint_candidates" / "conformal_interval_metrics.csv.gz"
    if conformal_path.exists():
        conformal = pd.read_csv(conformal_path, dtype={"GAGE_ID": str})
        conformal_summary = conformal.groupby(["model", "lead_days"], as_index=False).agg(
            basins=("GAGE_ID", "nunique"),
            mean_coverage=("coverage_90", "mean"),
            mean_coverage_abs_error=("coverage_abs_error", "mean"),
            mean_interval_width=("mean_interval_width", "mean"),
        )
        atomic_csv(conformal_summary, RESULTS_ROOT / "07_comparison" / "conformal_summary.csv")
        decision["conformal"] = {
            "purpose": "coverage correction only; not eligible as an endpoint-skill winner",
            "summary": conformal_summary.to_dict(orient="records"),
        }

    operational_path = RESULTS_ROOT / "05_operational" / "operational_basin_metrics.csv.gz"
    if operational_path.exists():
        operational = pd.read_csv(operational_path, dtype={"GAGE_ID": str})
        operational_comparison = pd.concat([
            paired_table(operational, reference_model, config, "brier")
            for reference_model in [
                "operational_qbin", "operational_persistence",
                "operational_training_climatology",
            ]
        ], ignore_index=True)
        atomic_csv(operational_comparison, RESULTS_ROOT / "07_comparison" / "operational_comparison.csv")
        allowable = operational_comparison[
            ~operational_comparison["model"].eq("oracle_future_forcing_branch")
            & operational_comparison["threshold_quantile"].eq(config["primary_quantile"])
        ]
        by_reference = allowable.groupby(["model", "reference_model"], as_index=False).agg(
            mean_gain=("weighted_improvement", "mean"),
            positive_leads=("weighted_improvement", lambda x: int((x > 0).sum())),
            significant_positive_leads=("bootstrap_ci_low", lambda x: int((x > 0).sum())),
        )
        required_references = 3
        operational_rank = by_reference.groupby("model", as_index=False).agg(
            references=("reference_model", "nunique"),
            minimum_mean_gain=("mean_gain", "min"),
            total_significant_positive_leads=("significant_positive_leads", "sum"),
            minimum_significant_positive_leads=("significant_positive_leads", "min"),
            mean_gain_across_references=("mean_gain", "mean"),
        )
        operational_rank["passes_rule"] = (
            operational_rank["references"].eq(required_references)
            & operational_rank["minimum_mean_gain"].gt(0)
            & operational_rank["minimum_significant_positive_leads"].ge(3)
        )
        operational_rank = operational_rank.sort_values(
            ["passes_rule", "minimum_significant_positive_leads", "minimum_mean_gain"],
            ascending=False,
        )
        decision["operational"] = {
            "required_references": [
                "operational_qbin", "operational_persistence",
                "operational_training_climatology",
            ],
            "by_reference": by_reference.to_dict(orient="records"),
            "ranking": operational_rank.to_dict(orient="records"),
            "best_method": str(operational_rank.iloc[0]["model"]) if len(operational_rank) else None,
            "future_weather_forecast_tested": False,
            "promote_to_full_confirmation": bool(
                len(operational_rank)
                and bool(operational_rank.iloc[0]["passes_rule"])
            ),
        }

    trajectory_path = RESULTS_ROOT / "06_first_passage" / "first_passage_deficit_comparison.csv"
    if trajectory_path.exists():
        trajectory = pd.read_csv(trajectory_path)
        primary_trajectory = trajectory[
            trajectory["threshold_quantile"].eq(config["primary_quantile"])
            & trajectory["score"].isin([
                "first_passage_brier", "first_passage_discrete_crps",
                "deficit_volume_crps",
            ])
        ]
        decision["trajectory"] = {
            "positive_comparisons": int(primary_trajectory[
                "weighted_improvement_coherent_vs_shuffled"
            ].gt(0).sum()),
            "comparisons": int(len(primary_trajectory)),
            "mean_basin_fraction_improved": float(primary_trajectory["basin_fraction_improved"].mean()),
            "significant_positive_comparisons": int(primary_trajectory[
                "bootstrap_ci_low"
            ].gt(0).sum()) if "bootstrap_ci_low" in primary_trajectory else 0,
            "promote_to_full_confirmation": bool(
                len(primary_trajectory)
                and primary_trajectory["weighted_improvement_coherent_vs_shuffled"].gt(0).mean() >= 0.75
                and primary_trajectory["basin_fraction_improved"].mean() >= 0.50
            ),
        }

    output = RESULTS_ROOT / "07_comparison"
    atomic_csv(endpoint_comparison, output / "conditional_endpoint_comparison.csv")
    atomic_csv(distribution_comparison, output / "conditional_distribution_comparison.csv")
    atomic_json(decision, output / "CANDIDATE_DECISION.json")
    lines = [
        "# Low-flow forecast candidate decision", "",
        "This experiment is isolated from manuscript production. No manuscript file was changed.", "",
        f"Conditional endpoint winner: **{decision['endpoint']['winner']}**", "",
        f"New endpoint method promoted: **{decision['endpoint']['new_endpoint_method_promoted']}**", "",
        "The decision uses temporally held-out 2016–2025 forecasts, paired basin comparisons,",
        "and a spatially hierarchical basin bootstrap. The oracle future-forcing branch is",
        "diagnostic only and is never eligible for promotion.", "",
    ]
    if "operational" in decision:
        lines.extend([
            f"Best initialization-only operational candidate: **{decision['operational']['best_method']}**", "",
            "No numerical-weather-prediction forcing was available, so this is a climatological",
            "dry-spell-survival experiment rather than a complete real-time forecast system.", "",
        ])
    (output / "CANDIDATE_DECISION.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
