"""Initialization-time dry-spell survival mixture without future-forcing selection."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from .common import (
    RESULTS_ROOT, atomic_csv, atomic_json, current_dry_cases, fit_logistic,
    load_benchmark_config, load_daily, qbin_probability, score_probability,
    split_cases,
)
from .endpoint import _score_rows, _valid_binary


def operational_worker(task):
    gage, source, spatial_group, config, leads = task
    rows, yearly_rows, survival_rows, failures = [], [], [], []
    frame = load_daily(source)
    qfit = frame.loc[frame["date"].le(pd.Timestamp(config["fit_end"])), "q_mm_day"]
    qfit = qfit[qfit.gt(0)]
    if qfit.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), [{"GAGE_ID": gage, "reason": "no_scale"}]
    scale = float(qfit.median())
    history = frame.loc[frame["date"].le(pd.Timestamp(config["threshold_reference_end"])), "q_mm_day"] / scale
    thresholds = {float(q): float(history.quantile(float(q))) for q in config["low_flow_quantiles"]}
    for lead in leads:
        try:
            cases = current_dry_cases(frame, int(lead), scale, config)
            fit, calibration, test = split_cases(cases, config)
            full_train = cases[
                cases["future_date"].le(pd.Timestamp(config["calibration_end"]))
            ].copy()
        except Exception as error:
            failures.append({"GAGE_ID": gage, "lead_days": lead, "reason": f"cases:{error!r}"})
            continue
        if (len(fit) < config["minimum_fit_cases"] or
                len(calibration) < config["minimum_calibration_cases"] or
                len(test) < config["minimum_evaluation_cases"]):
            continue
        feature_columns = [column for column in fit.columns if column not in {
            "date", "future_date", "q", "q_next", "future_dry"
        }]
        survive_fit = fit["future_dry"].to_numpy(bool)
        survive_full = full_train["future_dry"].to_numpy(bool)
        survive_cal = calibration["future_dry"].to_numpy(bool)
        survive_test = test["future_dry"].to_numpy(bool)
        survival_cal = survival_test = None
        if _valid_binary(survive_fit, config["minimum_events"]):
            try:
                calibration_survival_model = fit_logistic(fit, survive_fit, feature_columns)
                test_survival_model = fit_logistic(full_train, survive_full, feature_columns)
                survival_cal = calibration_survival_model.predict_proba(
                    calibration[feature_columns]
                )[:, 1]
                survival_test = test_survival_model.predict_proba(test[feature_columns])[:, 1]
                survival_rows.append({
                    "GAGE_ID": str(gage).zfill(8), "spatial_group": spatial_group,
                    "lead_days": int(lead),
                    **score_probability(survival_test, survive_test, config["probability_clip"]),
                })
            except Exception as error:
                failures.append({"GAGE_ID": gage, "lead_days": lead, "reason": f"survival:{error!r}"})
        for quantile, threshold in thresholds.items():
            fit_target = fit["q_next"].to_numpy(float) <= threshold
            full_target = full_train["q_next"].to_numpy(float) <= threshold
            cal_target = calibration["q_next"].to_numpy(float) <= threshold
            test_target = test["q_next"].to_numpy(float) <= threshold
            if not _valid_binary(fit_target, config["minimum_events"]):
                continue
            if not _valid_binary(test_target, config["minimum_events"]):
                continue
            probability_cal = {
                "operational_training_climatology": np.full(
                    len(calibration), (fit_target.sum() + 0.5) / (len(fit_target) + 1)
                ),
                "operational_persistence": (calibration["q"].to_numpy(float) <= threshold).astype(float),
                "operational_qbin": qbin_probability(
                    fit["q"].to_numpy(float), fit_target, calibration["q"].to_numpy(float)
                ),
            }
            probability_test = {
                "operational_training_climatology": np.full(
                    len(test), (full_target.sum() + 0.5) / (len(full_target) + 1)
                ),
                "operational_persistence": (test["q"].to_numpy(float) <= threshold).astype(float),
                "operational_qbin": qbin_probability(
                    full_train["q"].to_numpy(float), full_target,
                    test["q"].to_numpy(float),
                ),
            }
            try:
                calibration_direct = fit_logistic(fit, fit_target, feature_columns)
                test_direct = fit_logistic(full_train, full_target, feature_columns)
                probability_cal["operational_direct_logistic"] = calibration_direct.predict_proba(
                    calibration[feature_columns]
                )[:, 1]
                probability_test["operational_direct_logistic"] = test_direct.predict_proba(
                    test[feature_columns]
                )[:, 1]
            except Exception as error:
                failures.append({"GAGE_ID": gage, "lead_days": lead,
                                 "threshold_name": f"Q{int(100 * quantile)}",
                                 "reason": f"direct:{error!r}"})
            if survival_cal is None or survival_test is None:
                for model, probability in probability_test.items():
                    row, yearly = _score_rows(
                        gage, spatial_group, int(lead), quantile, threshold, model,
                        probability, test_target, test, None,
                        {"future_forcing_used_as_predictor": False}, config,
                    )
                    rows.append(row)
                    yearly_rows.extend(yearly)
                continue
            dry_fit = fit["future_dry"].to_numpy(bool)
            dry_full = full_train["future_dry"].to_numpy(bool)
            if dry_fit.sum() < 50 or (~dry_fit).sum() < 50:
                for model, probability in probability_test.items():
                    row, yearly = _score_rows(
                        gage, spatial_group, int(lead), quantile, threshold, model,
                        probability, test_target, test, None,
                        {"future_forcing_used_as_predictor": False}, config,
                    )
                    rows.append(row)
                    yearly_rows.extend(yearly)
                continue
            dry_cal = qbin_probability(
                fit.loc[dry_fit, "q"].to_numpy(float), fit_target[dry_fit],
                calibration["q"].to_numpy(float),
            )
            wet_cal = qbin_probability(
                fit.loc[~dry_fit, "q"].to_numpy(float), fit_target[~dry_fit],
                calibration["q"].to_numpy(float),
            )
            dry_test = qbin_probability(
                full_train.loc[dry_full, "q"].to_numpy(float), full_target[dry_full],
                test["q"].to_numpy(float),
            )
            wet_test = qbin_probability(
                full_train.loc[~dry_full, "q"].to_numpy(float), full_target[~dry_full],
                test["q"].to_numpy(float),
            )
            probability_cal["survival_mixture"] = survival_cal * dry_cal + (1 - survival_cal) * wet_cal
            probability_test["survival_mixture"] = survival_test * dry_test + (1 - survival_test) * wet_test
            probability_test["oracle_future_forcing_branch"] = np.where(survive_test, dry_test, wet_test)
            if _valid_binary(cal_target, config["minimum_events"]):
                try:
                    calibrator = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip")
                    calibrator.fit(probability_cal["survival_mixture"], cal_target.astype(float))
                    probability_test["survival_mixture_calibrated"] = calibrator.predict(
                        probability_test["survival_mixture"]
                    )
                except Exception as error:
                    failures.append({"GAGE_ID": gage, "lead_days": lead,
                                     "threshold_name": f"Q{int(100 * quantile)}",
                                     "reason": f"mixture_calibration:{error!r}"})
            for model, probability in probability_test.items():
                row, yearly = _score_rows(
                    gage, spatial_group, int(lead), quantile, threshold, model,
                    probability, test_target, test, None,
                    {"future_forcing_used_as_predictor": model == "oracle_future_forcing_branch"},
                    config,
                )
                rows.append(row)
                yearly_rows.extend(yearly)
    return pd.DataFrame(rows), pd.DataFrame(yearly_rows), pd.DataFrame(survival_rows), failures


def run_operational(panel: pd.DataFrame, workers: int,
                    output_subdir: str = "05_operational",
                    leads: list[int] | None = None) -> dict:
    config = load_benchmark_config()
    output = RESULTS_ROOT / output_subdir
    selected_leads = [int(v) for v in (
        leads if leads is not None else config["operational_lead_days"]
    )]
    tasks = [(r.GAGE_ID, r.file, r.spatial_group, config, selected_leads)
             for r in panel.itertuples(index=False)]
    metrics, yearly, survival, failures = [], [], [], []
    def retain(gage, result):
        frame, annual, dry_skill, failed = result
        if not frame.empty:
            metrics.append(frame)
        if not annual.empty:
            yearly.append(annual)
        if not dry_skill.empty:
            survival.append(dry_skill)
        failures.extend(failed)

    if workers == 1:
        for number, task in enumerate(tasks, start=1):
            retain(task[0], operational_worker(task))
            if number % 8 == 0 or number == len(tasks):
                print(f"Operational mixture {number}/{len(tasks)} basins", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(operational_worker, task): task[0] for task in tasks}
            for number, future in enumerate(as_completed(futures), start=1):
                gage = futures[future]
                try:
                    retain(gage, future.result())
                except Exception as error:
                    failures.append({"GAGE_ID": gage, "reason": repr(error)})
                if number % 8 == 0 or number == len(tasks):
                    print(f"Operational mixture {number}/{len(tasks)} basins", flush=True)
    combined = pd.concat(metrics, ignore_index=True) if metrics else pd.DataFrame()
    annual = pd.concat(yearly, ignore_index=True) if yearly else pd.DataFrame()
    dry_skill = pd.concat(survival, ignore_index=True) if survival else pd.DataFrame()
    atomic_csv(combined, output / "operational_basin_metrics.csv.gz", compression="gzip")
    atomic_csv(annual, output / "operational_water_year_scores.csv.gz", compression="gzip")
    atomic_csv(dry_skill, output / "dry_spell_survival_metrics.csv.gz", compression="gzip")
    atomic_csv(pd.DataFrame(failures), output / "operational_failures.csv")
    receipt = {
        "status": "complete", "basins": int(
            combined["GAGE_ID"].nunique() if "GAGE_ID" in combined else 0
        ),
        "forecast_initialization": "observed_dry_day",
        "future_forcing_selection": False,
        "oracle_branch_is_diagnostic_only": True,
    }
    atomic_json(receipt, output / "_SUCCESS.json")
    return receipt
