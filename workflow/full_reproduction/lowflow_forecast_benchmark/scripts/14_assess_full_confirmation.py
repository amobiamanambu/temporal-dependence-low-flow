#!/usr/bin/env python3
"""Assess promoted methods on the non-screening basins and write the final decision."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from lib.common import RESULTS_ROOT, atomic_csv, atomic_json, load_benchmark_config  # noqa: E402


def hierarchical_ci(group: pd.DataFrame, value: str, weight: str, replicates: int,
                    seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    grouped = [g for _, g in group.groupby("spatial_group", dropna=False)]
    numerator = np.empty((replicates, len(grouped)))
    denominator = np.empty_like(numerator)
    for column, values in enumerate(grouped):
        x = values[value].to_numpy(float)
        w = values[weight].to_numpy(float)
        indices = rng.integers(0, len(values), size=(replicates, len(values)))
        numerator[:, column] = np.sum(x[indices] * w[indices], axis=1)
        denominator[:, column] = np.sum(w[indices], axis=1)
    selected = rng.integers(0, len(grouped), size=(replicates, len(grouped)))
    rows = np.arange(replicates)[:, None]
    boot = numerator[rows, selected].sum(axis=1) / denominator[rows, selected].sum(axis=1)
    return tuple(np.quantile(boot, [0.025, 0.975]))


def operational_comparison(metrics: pd.DataFrame, config: dict) -> pd.DataFrame:
    keys = ["GAGE_ID", "spatial_group", "threshold_quantile", "threshold_name", "lead_days"]
    rows = []
    candidates = ["survival_mixture", "operational_direct_logistic"]
    references = ["operational_qbin", "operational_persistence",
                  "operational_training_climatology"]
    for candidate in candidates:
        alternative = metrics[metrics["model"].eq(candidate)][keys + ["n", "brier"]]
        for reference in references:
            baseline = metrics[metrics["model"].eq(reference)][keys + ["brier"]]
            paired = alternative.merge(baseline, on=keys, suffixes=("_candidate", "_reference"))
            paired["improvement"] = paired["brier_reference"] - paired["brier_candidate"]
            for (quantile, name, lead), group in paired.groupby(
                ["threshold_quantile", "threshold_name", "lead_days"]
            ):
                low, high = hierarchical_ci(
                    group, "improvement", "n", config["bootstrap_replicates"],
                    config["random_seed"] + int(lead) + int(100 * quantile) + sum(map(ord, candidate + reference)),
                )
                rows.append({
                    "model": candidate, "reference_model": reference,
                    "threshold_quantile": float(quantile), "threshold_name": name,
                    "lead_days": int(lead), "basins": int(len(group)),
                    "cases": int(group["n"].sum()),
                    "weighted_brier_improvement": float(np.average(
                        group["improvement"], weights=group["n"]
                    )),
                    "bootstrap_ci_low": float(low), "bootstrap_ci_high": float(high),
                    "basin_fraction_improved": float(group["improvement"].gt(0).mean()),
                })
    return pd.DataFrame(rows)


def trajectory_comparison(metrics: pd.DataFrame, config: dict) -> pd.DataFrame:
    keys = ["GAGE_ID", "spatial_group", "threshold_quantile", "threshold_name", "lead_days", "n"]
    shuffled = metrics[metrics["model"].eq("analog_marginal_shuffle")]
    coherent = metrics[metrics["model"].eq("coherent_block_analog")]
    paired = shuffled.merge(coherent, on=keys, suffixes=("_shuffled", "_coherent"))
    rows = []
    for (quantile, name, lead), group in paired.groupby(
        ["threshold_quantile", "threshold_name", "lead_days"]
    ):
        for score in ["first_passage_brier", "first_passage_discrete_crps",
                      "deficit_volume_crps"]:
            work = group.copy()
            work["improvement"] = work[f"{score}_shuffled"] - work[f"{score}_coherent"]
            low, high = hierarchical_ci(
                work, "improvement", "n", config["bootstrap_replicates"],
                config["random_seed"] + 14000 + int(lead) + int(100 * quantile) + sum(map(ord, score)),
            )
            rows.append({
                "threshold_quantile": float(quantile), "threshold_name": name,
                "lead_days": int(lead), "score": score, "basins": int(len(work)),
                "cases": int(work["n"].sum()),
                "weighted_improvement": float(np.average(work["improvement"], weights=work["n"])),
                "bootstrap_ci_low": float(low), "bootstrap_ci_high": float(high),
                "basin_fraction_improved": float(work["improvement"].gt(0).mean()),
            })
    return pd.DataFrame(rows)


def main() -> None:
    config = load_benchmark_config()
    root = RESULTS_ROOT / "08_full_confirmation"
    operational_path = root / "operational" / "operational_basin_metrics.csv.gz"
    trajectory_path = root / "trajectories" / "trajectory_basin_metrics.csv.gz"
    if not operational_path.exists() or not trajectory_path.exists():
        raise FileNotFoundError("Full operational and trajectory confirmations must finish first")
    operational = pd.read_csv(operational_path, dtype={"GAGE_ID": str})
    trajectories = pd.read_csv(trajectory_path, dtype={"GAGE_ID": str})
    additional_operational = root / "additional_operational" / "operational_basin_metrics.csv.gz"
    additional_trajectory = root / "additional_trajectories" / "trajectory_basin_metrics.csv.gz"
    if additional_operational.exists():
        operational = pd.concat([
            operational, pd.read_csv(additional_operational, dtype={"GAGE_ID": str})
        ], ignore_index=True)
    if additional_trajectory.exists():
        trajectories = pd.concat([
            trajectories, pd.read_csv(additional_trajectory, dtype={"GAGE_ID": str})
        ], ignore_index=True)
    op = operational_comparison(operational, config)
    tr = trajectory_comparison(trajectories, config)
    atomic_csv(op, root / "operational_full_comparison.csv")
    atomic_csv(tr, root / "trajectory_full_comparison.csv")

    op_primary = op[op["threshold_quantile"].eq(config["primary_quantile"])]
    op_summary = op_primary.groupby(["model", "reference_model"], as_index=False).agg(
        mean_gain=("weighted_brier_improvement", "mean"),
        positive_leads=("weighted_brier_improvement", lambda x: int((x > 0).sum())),
        significant_positive_leads=("bootstrap_ci_low", lambda x: int((x > 0).sum())),
        basins_min=("basins", "min"), basins_max=("basins", "max"),
    )
    op_method = op_summary.groupby("model", as_index=False).agg(
        minimum_mean_gain=("mean_gain", "min"),
        minimum_significant_positive_leads=("significant_positive_leads", "min"),
        references=("reference_model", "nunique"),
    )
    op_method["confirmed"] = (
        op_method["references"].eq(3) & op_method["minimum_mean_gain"].gt(0)
        & op_method["minimum_significant_positive_leads"].ge(3)
    )
    tr_primary = tr[
        tr["threshold_quantile"].eq(config["primary_quantile"])
        & tr["lead_days"].ge(7)
    ]
    trajectory_confirmed = bool(
        len(tr_primary) >= 9
        and tr_primary["weighted_improvement"].gt(0).all()
        and tr_primary["bootstrap_ci_low"].gt(0).mean() >= 2 / 3
        and tr_primary["basin_fraction_improved"].mean() > 0.5
    )
    decision = {
        "status": "complete",
        "conditional_endpoint_new_method": "not_supported_in_screening",
        "operational_method_summary": op_method.to_dict(orient="records"),
        "operational_by_reference": op_summary.to_dict(orient="records"),
        "operational_skill_confirmed": bool(op_method["confirmed"].any()),
        "confirmed_operational_methods": op_method.loc[op_method["confirmed"], "model"].tolist(),
        "coherent_first_passage_and_deficit_confirmed": trajectory_confirmed,
        "trajectory_primary_comparisons": int(len(tr_primary)),
        "trajectory_significant_positive_comparisons": int(tr_primary["bootstrap_ci_low"].gt(0).sum()),
        "trajectory_mean_basin_fraction_improved": float(tr_primary["basin_fraction_improved"].mean()),
        "boundaries": [
            "Operational models use initialization-time hydrograph and antecedent GridMET data but no NWP forecast forcing.",
            "Trajectory results are conditional on observed dry-spell continuation and are not unconditional real-time forecasts.",
            "No manuscript file was modified by this benchmark.",
        ],
    }
    atomic_json(decision, root / "FINAL_SCIENTIFIC_DECISION.json")
    lines = [
        "# Final low-flow forecasting decision", "",
        f"Operational skill confirmed: **{decision['operational_skill_confirmed']}**", "",
        "Confirmed operational methods: " + ", ".join(decision["confirmed_operational_methods"]), "",
        ("Coherent first-passage/deficit skill confirmed: **"
         f"{decision['coherent_first_passage_and_deficit_confirmed']}**"), "",
        "The established adaptive heavy-tail/recession model remains the conditional endpoint winner;",
        "none of the new marginal distribution families passed the screening promotion rule.", "",
        "These results do not alter manuscript files. The operational experiment has no future NWP",
        "forcing, and the trajectory experiment is conditional on an observed continuing dry spell.",
    ]
    (root / "FINAL_SCIENTIFIC_DECISION.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
