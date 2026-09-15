"""Frozen 1--120-day low-flow trajectory extension.

This module deliberately writes outside the confirmed 1--90-day result tree.
The hydrograph-state analog, features, thresholds, ensemble size, issue-day
rule, and chronological split are inherited unchanged.  Only the maximum path
length and the two predeclared extension leads (105 and 120 days) are new.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from .common import (
    RESULTS_ROOT, atomic_csv, atomic_json, dry_spell_age, ensemble_crps,
    fit_logistic, initialization_features, load_benchmark_config, load_daily,
    water_year,
)
from .extended_trajectory import (
    HYDROCLIMATE_FEATURES, HYDROGRAPH_FEATURES, SEASON_FEATURES,
    discrete_crps, first_event_time, shuffle_members_by_day,
)


HORIZON = 120
LEADS_120 = (1, 3, 7, 14, 30, 45, 60, 90, 105, 120)
EXTENSION_LEADS = (105, 120)
MEMBERS = 101
ISSUE_WEEKDAY = 2  # Wednesday, unchanged from the 1--90-day experiment.
OUTPUT_ROOT = RESULTS_ROOT / "10_extension_120"
FROZEN_MODELS = (
    "constant_persistence_path",
    "seasonal_climatology_path",
    "hydrograph_analog",
    "hydrograph_marginal_shuffle",
    "direct_event_logistic",
)


def _future_matrix(values: pd.Series, horizon: int = HORIZON) -> np.ndarray:
    return np.column_stack([
        values.shift(-lead).to_numpy(float) for lead in range(1, horizon + 1)
    ])


def extension_cases(frame: pd.DataFrame, scale: float,
                    config: dict) -> pd.DataFrame:
    """Build weekly dry-start cases with a complete 120-day discharge path."""
    dates = frame["date"]
    q = frame["q_mm_day"] / scale
    q_path = _future_matrix(q)
    consecutive = (dates.shift(-HORIZON) - dates).dt.days.eq(HORIZON).to_numpy(bool)
    valid_path = np.isfinite(q_path).all(axis=1) & (q_path >= 0).all(axis=1)
    current_dry = (
        frame["pr_mm"].le(config["precipitation_threshold_mm_day"])
        & frame["snowmelt_risk"].eq(False)
    )
    age = dry_spell_age(current_dry)
    selected = (
        consecutive
        & valid_path
        & q.gt(0).to_numpy(bool)
        & current_dry.fillna(False).to_numpy(bool)
        & (age >= int(config["antecedent_days"]))
        & dates.dt.dayofweek.eq(ISSUE_WEEKDAY).to_numpy(bool)
    )
    features, _ = initialization_features(frame, scale, config)
    rows = np.flatnonzero(selected)
    additions: dict[str, np.ndarray] = {
        "future_date": dates.shift(-HORIZON).iloc[rows].to_numpy(),
        "q": q.iloc[rows].to_numpy(float),
    }
    additions.update({
        f"q_h{day}": q_path[rows, day - 1]
        for day in range(1, HORIZON + 1)
    })
    return pd.concat(
        [features.iloc[rows].reset_index(drop=True), pd.DataFrame(additions)],
        axis=1,
    )


def analog_paths_120(train: pd.DataFrame, test: pd.DataFrame,
                     features: list[str], scale_by_initial_q: bool,
                     members: int = MEMBERS) -> np.ndarray:
    """Apply the frozen analog rule to complete 120-day historical paths."""
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    x_train = scaler.fit_transform(imputer.fit_transform(train[features]))
    x_test = scaler.transform(imputer.transform(test[features]))
    neighbors = min(int(members), len(train))
    search = NearestNeighbors(n_neighbors=neighbors)
    search.fit(x_train)
    indices = search.kneighbors(x_test, return_distance=False)
    path_columns = [f"q_h{day}" for day in range(1, HORIZON + 1)]
    training_paths = train[path_columns].to_numpy(np.float32)
    if scale_by_initial_q:
        training_paths = training_paths / train["q"].to_numpy(np.float32)[:, None]
        paths = (
            test["q"].to_numpy(np.float32)[:, None, None]
            * training_paths[indices]
        )
    else:
        paths = training_paths[indices]
    if neighbors < members:
        paths = np.tile(
            paths, (1, int(np.ceil(members / neighbors)), 1)
        )[:, :members]
    return np.maximum(paths, 0).astype(np.float32, copy=False)


def score_paths_120(gage: str, spatial_group: str, model: str,
                    paths: np.ndarray, test: pd.DataFrame,
                    thresholds: dict[float, float], config: dict) -> list[dict]:
    truth_all = test[
        [f"q_h{day}" for day in range(1, HORIZON + 1)]
    ].to_numpy(float)
    q0 = test["q"].to_numpy(float)
    years = water_year(test["date"])
    rows: list[dict] = []
    for quantile, threshold in thresholds.items():
        for target in ("onset", "recovery"):
            eligible = q0 > threshold if target == "onset" else q0 <= threshold
            if eligible.sum() < config["minimum_trajectory_evaluation_cases"]:
                continue
            for lead in LEADS_120:
                truth = truth_all[eligible, :lead]
                forecast = paths[eligible, :, :lead]
                observed_endpoint = truth[:, -1] <= threshold
                endpoint_probability = np.mean(
                    forecast[:, :, -1] <= threshold, axis=1
                )
                observed_time = first_event_time(truth, threshold, target)
                member_time = first_event_time(forecast, threshold, target)
                observed_event = observed_time <= lead
                if min(observed_event.sum(), (~observed_event).sum()) < config["minimum_events"]:
                    continue
                event_probability = np.mean(member_time <= lead, axis=1)
                observed_deficit = np.sum(np.maximum(threshold - truth, 0), axis=1)
                member_deficit = np.sum(
                    np.maximum(threshold - forecast, 0), axis=2
                )
                observed_duration = np.sum(truth <= threshold, axis=1).astype(float)
                member_duration = np.sum(forecast <= threshold, axis=2).astype(float)
                observed_minimum = np.min(truth, axis=1)
                member_minimum = np.min(forecast, axis=2)
                rows.append({
                    "GAGE_ID": str(gage).zfill(8),
                    "spatial_group": spatial_group,
                    "threshold_quantile": float(quantile),
                    "threshold_name": f"Q{int(round(100 * quantile))}",
                    "lead_days": int(lead),
                    "target": target,
                    "model": model,
                    "n": int(eligible.sum()),
                    "events": int(observed_event.sum()),
                    "endpoint_brier": float(np.mean(
                        (endpoint_probability - observed_endpoint) ** 2
                    )),
                    "event_brier": float(np.mean(
                        (event_probability - observed_event) ** 2
                    )),
                    "timing_crps": float(np.mean(discrete_crps(
                        member_time, observed_time, lead
                    ))),
                    "deficit_crps": float(np.mean(ensemble_crps(
                        member_deficit, observed_deficit
                    ))),
                    "duration_crps": float(np.mean(ensemble_crps(
                        member_duration, observed_duration
                    ))),
                    "minimum_flow_crps": float(np.mean(ensemble_crps(
                        member_minimum, observed_minimum
                    ))),
                    "test_water_years": int(len(np.unique(years[eligible]))),
                    "future_forcing_used": False,
                    "analysis_horizon_days": HORIZON,
                })
    return rows


def direct_q10_onset_rows(gage: str, spatial_group: str, train: pd.DataFrame,
                          test: pd.DataFrame, threshold: float,
                          config: dict) -> list[dict]:
    """Frozen initialization-time binary benchmark for 105 and 120 days."""
    rows: list[dict] = []
    train_selected = train["q"].to_numpy(float) > threshold
    test_selected = test["q"].to_numpy(float) > threshold
    if train_selected.sum() < config["minimum_trajectory_fit_cases"]:
        return rows
    if test_selected.sum() < config["minimum_trajectory_evaluation_cases"]:
        return rows
    fit = train.loc[train_selected]
    evaluate = test.loc[test_selected]
    for lead in EXTENSION_LEADS:
        fit_path = fit[
            [f"q_h{day}" for day in range(1, lead + 1)]
        ].to_numpy(float)
        test_path = evaluate[
            [f"q_h{day}" for day in range(1, lead + 1)]
        ].to_numpy(float)
        fit_event = (fit_path <= threshold).any(axis=1)
        test_event = (test_path <= threshold).any(axis=1)
        if min(fit_event.sum(), (~fit_event).sum()) < config["minimum_events"]:
            continue
        if min(test_event.sum(), (~test_event).sum()) < config["minimum_events"]:
            continue
        model = fit_logistic(fit, fit_event, HYDROCLIMATE_FEATURES)
        probability = model.predict_proba(evaluate[HYDROCLIMATE_FEATURES])[:, 1]
        rows.append({
            "GAGE_ID": str(gage).zfill(8),
            "spatial_group": spatial_group,
            "threshold_quantile": 0.10,
            "threshold_name": "Q10",
            "lead_days": int(lead),
            "target": "onset",
            "model": "direct_event_logistic",
            "n": int(len(evaluate)),
            "events": int(test_event.sum()),
            "endpoint_brier": np.nan,
            "event_brier": float(np.mean((probability - test_event) ** 2)),
            "timing_crps": np.nan,
            "deficit_crps": np.nan,
            "duration_crps": np.nan,
            "minimum_flow_crps": np.nan,
            "test_water_years": int(len(np.unique(water_year(evaluate["date"])))),
            "future_forcing_used": False,
            "analysis_horizon_days": HORIZON,
        })
    return rows


def extension_worker(task) -> tuple[pd.DataFrame, list[dict]]:
    gage, source, spatial_group, config = task
    try:
        frame = load_daily(source)
        qfit = frame.loc[
            frame["date"].le(pd.Timestamp(config["fit_end"])), "q_mm_day"
        ]
        qfit = qfit[qfit.gt(0)]
        if qfit.empty:
            raise ValueError("no_positive_training_scale")
        scale = float(qfit.median())
        history = frame.loc[
            frame["date"].le(pd.Timestamp(config["threshold_reference_end"])),
            "q_mm_day",
        ] / scale
        thresholds = {
            float(q): float(history.quantile(float(q)))
            for q in config["low_flow_quantiles"]
        }
        thresholds = {
            q: value for q, value in thresholds.items()
            if np.isfinite(value) and value > 0
        }
        cases = extension_cases(frame, scale, config)
        train = cases[
            cases["future_date"].le(pd.Timestamp(config["calibration_end"]))
        ].copy()
        test = cases[
            cases["date"].ge(pd.Timestamp(config["evaluation_start"]))
            & cases["future_date"].le(pd.Timestamp(config["evaluation_end"]))
        ].copy()
        if len(train) < config["minimum_trajectory_fit_cases"]:
            raise ValueError("insufficient_training_initializations")
        if len(test) < config["minimum_trajectory_evaluation_cases"]:
            raise ValueError("insufficient_test_initializations")

        rows: list[dict] = []
        persistence = np.repeat(
            test["q"].to_numpy(np.float32)[:, None, None], HORIZON, axis=2
        )
        rows.extend(score_paths_120(
            gage, spatial_group, "constant_persistence_path", persistence,
            test, thresholds, config,
        ))
        del persistence

        seasonal = analog_paths_120(
            train, test, SEASON_FEATURES, scale_by_initial_q=False
        )
        rows.extend(score_paths_120(
            gage, spatial_group, "seasonal_climatology_path", seasonal,
            test, thresholds, config,
        ))
        del seasonal

        coherent = analog_paths_120(
            train, test, HYDROGRAPH_FEATURES, scale_by_initial_q=True
        )
        rows.extend(score_paths_120(
            gage, spatial_group, "hydrograph_analog", coherent,
            test, thresholds, config,
        ))
        seed = int(config["random_seed"] + int(str(gage)[-5:]) + 24000)
        shuffled = shuffle_members_by_day(coherent, seed)
        rows.extend(score_paths_120(
            gage, spatial_group, "hydrograph_marginal_shuffle", shuffled,
            test, thresholds, config,
        ))
        del coherent, shuffled

        if 0.10 in thresholds:
            try:
                rows.extend(direct_q10_onset_rows(
                    gage, spatial_group, train, test, thresholds[0.10], config
                ))
            except Exception:
                # The path experiment remains valid if a basin-specific direct
                # comparator cannot be fitted because of separation or missingness.
                pass
        if not rows:
            raise ValueError("no_scored_targets")
        return pd.DataFrame(rows), []
    except Exception as error:
        return pd.DataFrame(), [{
            "GAGE_ID": str(gage).zfill(8), "reason": repr(error)
        }]


def _read_metric_files(folder: Path) -> pd.DataFrame:
    paths = sorted(folder.glob("*_metrics.csv.gz"))
    frames = [pd.read_csv(path, dtype={"GAGE_ID": str}) for path in paths]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def run_extension_shard(panel: pd.DataFrame, shard_index: int, shard_count: int,
                        force: bool = False, output_label: str | None = None) -> dict:
    if shard_count < 1 or not 0 <= shard_index < shard_count:
        raise ValueError("shard_index must be in [0, shard_count)")
    config = load_benchmark_config()
    ordered = panel.sort_values("GAGE_ID").reset_index(drop=True)
    selected = ordered.iloc[
        np.arange(len(ordered)) % shard_count == shard_index
    ].copy()
    label = output_label or f"extension_shard_{shard_index:02d}"
    output = OUTPUT_ROOT / label
    by_basin = output / "by_basin"
    by_basin.mkdir(parents=True, exist_ok=True)
    completed = 0
    for number, row in enumerate(selected.itertuples(index=False), start=1):
        gage = str(row.GAGE_ID).zfill(8)
        metric_path = by_basin / f"{gage}_metrics.csv.gz"
        failure_path = by_basin / f"{gage}_failure.json"
        if not force and (metric_path.exists() or failure_path.exists()):
            completed += 1
        else:
            metrics, failed = extension_worker(
                (gage, row.file, row.spatial_group, config)
            )
            if not metrics.empty:
                atomic_csv(metrics, metric_path, compression="gzip")
                failure_path.unlink(missing_ok=True)
            else:
                record = failed[0] if failed else {
                    "GAGE_ID": gage, "reason": "no_scored_targets"
                }
                atomic_json(record, failure_path)
            completed += 1
        if number % 10 == 0 or number == len(selected):
            print(
                f"120-day shard {shard_index + 1}/{shard_count}: "
                f"{number}/{len(selected)} basins",
                flush=True,
            )

    metrics = _read_metric_files(by_basin)
    failure_files = sorted(by_basin.glob("*_failure.json"))
    failures = [
        json.loads(path.read_text(encoding="utf-8")) for path in failure_files
    ]
    atomic_csv(metrics, output / "extension_metrics.csv.gz", compression="gzip")
    atomic_csv(pd.DataFrame(failures), output / "extension_failures.csv")
    receipt = {
        "status": "complete",
        "shard_index": int(shard_index),
        "shard_count": int(shard_count),
        "basins_requested": int(len(selected)),
        "basins_completed": int(completed),
        "basins_with_results": int(
            metrics["GAGE_ID"].nunique() if not metrics.empty else 0
        ),
        "failures": int(len(failures)),
        "models": list(FROZEN_MODELS),
        "leads": list(LEADS_120),
        "predeclared_extension_leads": list(EXTENSION_LEADS),
        "future_forcing_selection": False,
        "future_observations_used_by_eligible_models": False,
        "separate_from_1_90_archive": True,
    }
    atomic_json(receipt, output / "_SUCCESS.json")
    return receipt
