"""Conditional endpoint-forecast candidate engine."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from .common import (
    RESULTS_ROOT, asymmetric_laplace_mixture_members, atomic_csv, atomic_json,
    conditional_cases, empirical_members, ensemble_probability, fit_location_scale,
    fit_logistic, gpd_members, idr_probability, load_benchmark_config, load_daily,
    lower_tail_twcrps, qbin_probability, quantile_forest_members, score_ensemble,
    score_probability, split_cases, student_members, water_year,
)


ALL_CANDIDATES = [
    "training_climatology", "persistence", "qbin_probability",
    "empirical_heavytail", "adaptive_heavytail_recession_logistic",
    "binary_idr", "student_t_innovations", "gpd_tail_splice",
    "asymmetric_laplace_mixture", "quantile_forest",
    "gpd_isotonic_calibrated", "calibration_stack", "state_dependent_stack",
]
ENDPOINT_VERSION = 3


def _valid_binary(target: np.ndarray, minimum: int) -> bool:
    target = np.asarray(target, bool)
    return int(target.sum()) >= minimum and int((~target).sum()) >= minimum


def _safe_logit(probability: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(probability, float), 1e-5, 1 - 1e-5)
    return np.log(p / (1 - p))


def _score_rows(gage: str, spatial_group: str, lead: int, quantile: float,
                threshold: float, model: str, probability: np.ndarray,
                target: np.ndarray, test: pd.DataFrame, ensemble: np.ndarray | None,
                diagnostics: dict, config: dict) -> tuple[dict, list[dict]]:
    row = {
        "benchmark_version": ENDPOINT_VERSION,
        "GAGE_ID": str(gage).zfill(8), "spatial_group": spatial_group,
        "lead_days": int(lead), "threshold_quantile": float(quantile),
        "threshold_name": f"Q{int(round(100 * quantile))}",
        "threshold_qnorm": float(threshold), "model": model,
        **score_probability(probability, target, float(config["probability_clip"])),
        "crps": np.nan, "twcrps_below_threshold": np.nan,
        "interval_90_coverage": np.nan, "interval_90_width": np.nan,
        **diagnostics,
    }
    if ensemble is not None:
        row.update(score_ensemble(ensemble, test["q_next"].to_numpy(float), threshold))
    years = water_year(test["date"])
    p = np.clip(
        probability,
        float(config["probability_clip"]),
        1 - float(config["probability_clip"]),
    )
    y = target.astype(float)
    brier = (p - y) ** 2
    log_score = -(y * np.log(p) + (1 - y) * np.log(1 - p))
    yearly = []
    for year in np.unique(years):
        selected = years == year
        yearly.append({
            "GAGE_ID": str(gage).zfill(8), "spatial_group": spatial_group,
            "lead_days": int(lead), "threshold_quantile": float(quantile),
            "threshold_name": f"Q{int(round(100 * quantile))}",
            "model": model, "water_year": int(year), "n": int(selected.sum()),
            "events": int(target[selected].sum()),
            "brier_sum": float(brier[selected].sum()),
            "log_loss_sum": float(log_score[selected].sum()),
        })
    return row, yearly


def basin_endpoint_worker(task: tuple) -> tuple[pd.DataFrame, pd.DataFrame, list[dict]]:
    gage, source, spatial_group, methods, leads, config = task
    failures: list[dict] = []
    rows: list[dict] = []
    yearly_rows: list[dict] = []
    frame = load_daily(source)
    scale_source = frame.loc[frame["date"].le(pd.Timestamp(config["fit_end"])), "q_mm_day"]
    scale_source = scale_source[scale_source.gt(0)]
    if scale_source.empty:
        return pd.DataFrame(), pd.DataFrame(), [{"GAGE_ID": gage, "reason": "no_fit_scale"}]
    scale = float(scale_source.median())
    historical = frame.loc[
        frame["date"].le(pd.Timestamp(config["threshold_reference_end"])), "q_mm_day"
    ] / scale
    historical = historical[np.isfinite(historical) & historical.ge(0)]
    thresholds = {float(q): float(historical.quantile(float(q)))
                  for q in config["low_flow_quantiles"]}

    for lead in leads:
        try:
            cases = conditional_cases(frame, int(lead), scale, config)
            fit, calibration, test = split_cases(cases, config)
            full_train = cases[
                cases["future_date"].le(pd.Timestamp(config["calibration_end"]))
            ].copy()
        except Exception as error:
            failures.append({"GAGE_ID": gage, "lead_days": lead, "reason": f"cases:{error!r}"})
            continue
        if len(fit) < int(config["minimum_fit_cases"]):
            failures.append({"GAGE_ID": gage, "lead_days": lead, "reason": "insufficient_fit_cases"})
            continue
        if len(calibration) < int(config["minimum_calibration_cases"]):
            failures.append({"GAGE_ID": gage, "lead_days": lead, "reason": "insufficient_calibration_cases"})
            continue
        if len(test) < int(config["minimum_evaluation_cases"]):
            failures.append({"GAGE_ID": gage, "lead_days": lead, "reason": "insufficient_evaluation_cases"})
            continue
        try:
            fitted_calibration = fit_location_scale(fit)
            fitted_test = fit_location_scale(full_train)
        except Exception as error:
            failures.append({"GAGE_ID": gage, "lead_days": lead, "reason": f"location_scale:{error!r}"})
            continue
        seed = int(config["random_seed"] + int(str(gage)[-5:]) + 100 * int(lead))
        ensembles: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        ensemble_diagnostics: dict[str, dict] = {}
        try:
            cal_values = empirical_members(
                calibration["q"].to_numpy(float), fitted_calibration,
                int(config["ensemble_members"]),
            )
            test_values = empirical_members(
                test["q"].to_numpy(float), fitted_test, int(config["ensemble_members"])
            )
            ensembles["empirical_heavytail"] = (cal_values, test_values)
        except Exception as error:
            failures.append({"GAGE_ID": gage, "lead_days": lead, "reason": f"empirical:{error!r}"})
        if "student_t_innovations" in methods:
            try:
                cal_values = student_members(
                    calibration["q"].to_numpy(float), fitted_calibration,
                    int(config["ensemble_members"]),
                )
                test_values = student_members(
                    test["q"].to_numpy(float), fitted_test, int(config["ensemble_members"])
                )
                ensembles["student_t_innovations"] = (cal_values, test_values)
            except Exception as error:
                failures.append({"GAGE_ID": gage, "lead_days": lead, "reason": f"student_t:{error!r}"})
        if any(name in methods for name in [
            "gpd_tail_splice", "gpd_isotonic_calibrated", "calibration_stack",
            "state_dependent_stack",
        ]):
            try:
                cal_values, _ = gpd_members(
                    calibration["q"].to_numpy(float), fitted_calibration,
                    int(config["ensemble_members"]), config,
                )
                test_values, diagnostic = gpd_members(
                    test["q"].to_numpy(float), fitted_test,
                    int(config["ensemble_members"]), config,
                )
                ensembles["gpd_tail_splice"] = (cal_values, test_values)
                ensemble_diagnostics["gpd_tail_splice"] = diagnostic
            except Exception as error:
                failures.append({"GAGE_ID": gage, "lead_days": lead, "reason": f"gpd:{error!r}"})
        if "asymmetric_laplace_mixture" in methods:
            try:
                cal_values, _ = asymmetric_laplace_mixture_members(
                    calibration["q"].to_numpy(float), fitted_calibration,
                    int(config["ensemble_members"]), seed,
                )
                test_values, diagnostic = asymmetric_laplace_mixture_members(
                    test["q"].to_numpy(float), fitted_test,
                    int(config["ensemble_members"]), seed + 1,
                )
                ensembles["asymmetric_laplace_mixture"] = (cal_values, test_values)
                ensemble_diagnostics["asymmetric_laplace_mixture"] = diagnostic
            except Exception as error:
                failures.append({"GAGE_ID": gage, "lead_days": lead, "reason": f"al_mixture:{error!r}"})
        feature_columns = [column for column in fit.columns if column not in {
            "date", "future_date", "q", "q_next", "dq"
        }]
        if "quantile_forest" in methods:
            try:
                cal_values = quantile_forest_members(
                    fit, calibration, feature_columns, config, seed
                )
                test_values = quantile_forest_members(
                    full_train, test, feature_columns, config, seed + 1
                )
                ensembles["quantile_forest"] = (cal_values, test_values)
            except Exception as error:
                failures.append({"GAGE_ID": gage, "lead_days": lead, "reason": f"quantile_forest:{error!r}"})

        for quantile, threshold in thresholds.items():
            if not np.isfinite(threshold) or threshold <= 0:
                continue
            fit_target = fit["q_next"].to_numpy(float) <= threshold
            full_target = full_train["q_next"].to_numpy(float) <= threshold
            cal_target = calibration["q_next"].to_numpy(float) <= threshold
            test_target = test["q_next"].to_numpy(float) <= threshold
            minimum = int(config["minimum_events"])
            if not _valid_binary(fit_target, minimum) or not _valid_binary(test_target, minimum):
                continue
            probability_cal: dict[str, np.ndarray] = {
                "training_climatology": np.full(len(calibration), (fit_target.sum() + 0.5) / (len(fit_target) + 1)),
                "persistence": (calibration["q"].to_numpy(float) <= threshold).astype(float),
                "qbin_probability": qbin_probability(
                    fit["q"].to_numpy(float), fit_target,
                    calibration["q"].to_numpy(float),
                ),
            }
            probability_test: dict[str, np.ndarray] = {
                "training_climatology": np.full(
                    len(test), (full_target.sum() + 0.5) / (len(full_target) + 1)
                ),
                "persistence": (test["q"].to_numpy(float) <= threshold).astype(float),
                "qbin_probability": qbin_probability(
                    full_train["q"].to_numpy(float), full_target,
                    test["q"].to_numpy(float),
                ),
            }
            for name, (cal_ensemble, test_ensemble) in ensembles.items():
                probability_cal[name] = ensemble_probability(cal_ensemble, threshold)
                probability_test[name] = ensemble_probability(test_ensemble, threshold)
            try:
                calibration_logistic = fit_logistic(fit, fit_target, feature_columns)
                test_logistic = fit_logistic(full_train, full_target, feature_columns)
                probability_cal["recession_logistic"] = calibration_logistic.predict_proba(
                    calibration[feature_columns]
                )[:, 1]
                probability_test["recession_logistic"] = test_logistic.predict_proba(
                    test[feature_columns]
                )[:, 1]
                empirical_cal = probability_cal["empirical_heavytail"]
                empirical_test = probability_test["empirical_heavytail"]
                weight = 0.5 if int(lead) >= 7 else 0.0
                probability_cal["adaptive_heavytail_recession_logistic"] = (
                    (1 - weight) * empirical_cal + weight * probability_cal["recession_logistic"]
                )
                probability_test["adaptive_heavytail_recession_logistic"] = (
                    (1 - weight) * empirical_test + weight * probability_test["recession_logistic"]
                )
            except Exception as error:
                failures.append({"GAGE_ID": gage, "lead_days": lead,
                                 "threshold_name": f"Q{int(100 * quantile)}",
                                 "reason": f"logistic:{error!r}"})
            if any(name in methods for name in [
                "binary_idr", "calibration_stack", "state_dependent_stack",
            ]):
                try:
                    probability_cal["binary_idr"] = idr_probability(
                        fit["q"].to_numpy(float), fit_target,
                        calibration["q"].to_numpy(float),
                    )
                    probability_test["binary_idr"] = idr_probability(
                        full_train["q"].to_numpy(float), full_target,
                        test["q"].to_numpy(float),
                    )
                except Exception as error:
                    failures.append({"GAGE_ID": gage, "lead_days": lead,
                                     "threshold_name": f"Q{int(100 * quantile)}",
                                     "reason": f"idr:{error!r}"})
            if "gpd_isotonic_calibrated" in methods and _valid_binary(cal_target, minimum):
                if "gpd_tail_splice" in probability_cal:
                    try:
                        calibrator = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip")
                        calibrator.fit(probability_cal["gpd_tail_splice"], cal_target.astype(float))
                        probability_test["gpd_isotonic_calibrated"] = calibrator.predict(
                            probability_test["gpd_tail_splice"]
                        )
                    except Exception as error:
                        failures.append({"GAGE_ID": gage, "lead_days": lead,
                                         "threshold_name": f"Q{int(100 * quantile)}",
                                         "reason": f"gpd_calibration:{error!r}"})
            if any(name in methods for name in [
                "calibration_stack", "state_dependent_stack",
            ]) and _valid_binary(cal_target, minimum):
                base = [name for name in [
                    "empirical_heavytail", "gpd_tail_splice", "binary_idr",
                    "recession_logistic", "qbin_probability",
                ] if name in probability_cal and name in probability_test]
                if len(base) >= 3:
                    try:
                        x_cal = np.column_stack([_safe_logit(probability_cal[name]) for name in base])
                        x_test = np.column_stack([_safe_logit(probability_test[name]) for name in base])
                        stack = make_pipeline(
                            StandardScaler(),
                            LogisticRegression(C=0.25, max_iter=1000, solver="lbfgs"),
                        )
                        stack.fit(x_cal, cal_target)
                        probability_test["calibration_stack"] = stack.predict_proba(x_test)[:, 1]
                        if "state_dependent_stack" in methods:
                            cal_state = np.column_stack([
                                np.log(np.maximum(calibration["q"].to_numpy(float), 1e-10) / threshold),
                                np.log1p(calibration["dry_spell_age"].to_numpy(float)),
                                calibration["dlog_q_7"].to_numpy(float),
                            ])
                            test_state = np.column_stack([
                                np.log(np.maximum(test["q"].to_numpy(float), 1e-10) / threshold),
                                np.log1p(test["dry_spell_age"].to_numpy(float)),
                                test["dlog_q_7"].to_numpy(float),
                            ])
                            state_cal = np.column_stack([
                                x_cal, cal_state, x_cal[:, :3] * cal_state[:, [0]],
                            ])
                            state_test = np.column_stack([
                                x_test, test_state, x_test[:, :3] * test_state[:, [0]],
                            ])
                            state_stack = make_pipeline(
                                SimpleImputer(strategy="median"),
                                StandardScaler(),
                                LogisticRegression(C=0.10, max_iter=1000, solver="lbfgs"),
                            )
                            state_stack.fit(state_cal, cal_target)
                            probability_test["state_dependent_stack"] = (
                                state_stack.predict_proba(state_test)[:, 1]
                            )
                    except Exception as error:
                        failures.append({"GAGE_ID": gage, "lead_days": lead,
                                         "threshold_name": f"Q{int(100 * quantile)}",
                                         "reason": f"stack:{error!r}"})

            requested = set(methods) | {
                "training_climatology", "persistence", "qbin_probability",
                "empirical_heavytail", "adaptive_heavytail_recession_logistic",
            }
            for name in sorted(requested):
                if name not in probability_test:
                    continue
                ensemble = ensembles.get(name, (None, None))[1]
                diagnostics = ensemble_diagnostics.get(name, {})
                row, yearly = _score_rows(
                    gage, spatial_group, int(lead), quantile, threshold, name,
                    probability_test[name], test_target, test, ensemble,
                    diagnostics, config,
                )
                rows.append(row)
                yearly_rows.extend(yearly)
    return pd.DataFrame(rows), pd.DataFrame(yearly_rows), failures


def run_endpoint(panel: pd.DataFrame, methods: list[str], leads: list[int], workers: int,
                 output_subdir: str, force: bool = False) -> dict:
    config = load_benchmark_config()
    output = RESULTS_ROOT / output_subdir
    basin_root = output / "by_basin"
    basin_root.mkdir(parents=True, exist_ok=True)
    tasks = []
    cached_metrics, cached_yearly = [], []
    for row in panel.itertuples(index=False):
        metric_path = basin_root / f"{row.GAGE_ID}_metrics.csv.gz"
        yearly_path = basin_root / f"{row.GAGE_ID}_water_year.csv.gz"
        if metric_path.exists() and yearly_path.exists() and not force:
            candidate_cache = pd.read_csv(metric_path, dtype={"GAGE_ID": str})
            required_methods = set(methods) | {
                "training_climatology", "persistence", "qbin_probability",
                "empirical_heavytail", "adaptive_heavytail_recession_logistic",
            }
            cache_valid = (
                "benchmark_version" in candidate_cache
                and int(candidate_cache["benchmark_version"].min()) >= ENDPOINT_VERSION
                and required_methods.issubset(set(candidate_cache["model"]))
                and set(leads).issubset(set(candidate_cache["lead_days"].astype(int)))
            )
            if cache_valid:
                cached_metrics.append(candidate_cache)
                cached_yearly.append(pd.read_csv(yearly_path, dtype={"GAGE_ID": str}))
            else:
                tasks.append((row.GAGE_ID, row.file, row.spatial_group, methods, leads, config))
        else:
            tasks.append((row.GAGE_ID, row.file, row.spatial_group, methods, leads, config))
    metrics, yearly, failures = list(cached_metrics), list(cached_yearly), []

    def retain(gage, result):
        metric_frame, yearly_frame, basin_failures = result
        if not metric_frame.empty:
            atomic_csv(metric_frame, basin_root / f"{gage}_metrics.csv.gz", compression="gzip")
            metrics.append(metric_frame)
        if not yearly_frame.empty:
            atomic_csv(yearly_frame, basin_root / f"{gage}_water_year.csv.gz", compression="gzip")
            yearly.append(yearly_frame)
        failures.extend(basin_failures)

    if workers == 1:
        for number, task in enumerate(tasks, start=1):
            retain(task[0], basin_endpoint_worker(task))
            if number % 8 == 0 or number == len(tasks):
                print(f"Completed {number}/{len(tasks)} endpoint basins", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            future_map = {executor.submit(basin_endpoint_worker, task): task[0] for task in tasks}
            for number, future in enumerate(as_completed(future_map), start=1):
                gage = future_map[future]
                try:
                    retain(gage, future.result())
                except Exception as error:
                    failures.append({"GAGE_ID": gage, "reason": f"worker:{error!r}"})
                if number % 8 == 0 or number == len(tasks):
                    print(f"Completed {number}/{len(tasks)} endpoint basins", flush=True)
    combined_metrics = pd.concat(metrics, ignore_index=True) if metrics else pd.DataFrame()
    combined_yearly = pd.concat(yearly, ignore_index=True) if yearly else pd.DataFrame()
    if combined_metrics.empty:
        raise RuntimeError("No endpoint results were produced")
    atomic_csv(combined_metrics, output / "endpoint_basin_metrics.csv.gz", compression="gzip")
    atomic_csv(combined_yearly, output / "endpoint_water_year_scores.csv.gz", compression="gzip")
    atomic_csv(pd.DataFrame(failures), output / "endpoint_failures.csv")
    receipt = {
        "status": "complete", "basins_requested": int(len(panel)),
        "benchmark_version": ENDPOINT_VERSION,
        "basins_with_results": int(combined_metrics["GAGE_ID"].nunique()),
        "methods": sorted(combined_metrics["model"].unique().tolist()),
        "leads": sorted(combined_metrics["lead_days"].unique().astype(int).tolist()),
        "fit_end": config["fit_end"], "calibration_end": config["calibration_end"],
        "evaluation_start": config["evaluation_start"],
        "target": "endpoint_low_flow_conditional_on_observed_dry_spell_continuation",
    }
    atomic_json(receipt, output / "_SUCCESS.json")
    return receipt
