#!/usr/bin/env python3
"""Test robustness to analog size, shuffle seed, dry screen, and threshold form."""

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
    RESULTS_ROOT, atomic_csv, atomic_json, ensemble_crps, load_benchmark_config,
    load_daily,
)
from lib.extended_trajectory import (  # noqa: E402
    HYDROGRAPH_FEATURES, LEADS, analog_paths, discrete_crps, extended_cases,
    first_event_time, shuffle_members_by_day,
)


SENSITIVITY_LEADS = (30, 45, 60, 90)
SCORES = ("event_brier", "timing_crps", "deficit_crps", "duration_crps")


def seasonal_threshold_lookup(frame: pd.DataFrame, scale: float, end: str,
                              quantile: float, half_window: int = 15) -> np.ndarray:
    history = frame[
        frame["date"].le(pd.Timestamp(end)) & frame["q_mm_day"].gt(0)
    ].copy()
    values = history["q_mm_day"].to_numpy(float) / scale
    doy = history["date"].dt.dayofyear.to_numpy(int)
    lookup = np.empty(367, float)
    for day in range(1, 367):
        distance = np.abs(doy - day)
        distance = np.minimum(distance, 366 - distance)
        selected = values[distance <= half_window]
        lookup[day] = np.quantile(selected, quantile) if len(selected) else np.nan
    return lookup


def threshold_paths(test: pd.DataFrame, fixed: float,
                    seasonal: np.ndarray | None) -> tuple[np.ndarray, np.ndarray]:
    if seasonal is None:
        initial = np.full(len(test), fixed, float)
        future = np.full((len(test), max(LEADS)), fixed, float)
        return initial, future
    dates = pd.to_datetime(test["date"])
    initial = seasonal[dates.dt.dayofyear.to_numpy(int)]
    future = np.column_stack([
        seasonal[(dates + pd.to_timedelta(day, unit="D")).dt.dayofyear.to_numpy(int)]
        for day in range(1, max(LEADS) + 1)
    ])
    return initial, future


def score_paths(gage: str, spatial_group: str, variant: str, model: str,
                paths: np.ndarray, test: pd.DataFrame, initial_threshold: np.ndarray,
                future_threshold: np.ndarray, threshold_mode: str,
                members: int, shuffle_seed: int) -> list[dict]:
    truth_all = test[
        [f"q_h{day}" for day in range(1, max(LEADS) + 1)]
    ].to_numpy(float)
    eligible = test["q"].to_numpy(float) > initial_threshold
    if eligible.sum() < 50:
        return []
    rows: list[dict] = []
    for lead in SENSITIVITY_LEADS:
        truth = truth_all[eligible, :lead]
        forecast = paths[eligible, :, :lead]
        threshold = future_threshold[eligible, :lead]
        observed_condition = truth <= threshold
        forecast_condition = forecast <= threshold[:, None, :]
        observed_time = first_event_time(
            np.where(observed_condition, 0.0, 1.0), 0.5, "onset"
        )
        member_time = first_event_time(
            np.where(forecast_condition, 0.0, 1.0), 0.5, "onset"
        )
        observed_event = observed_time <= lead
        if min(observed_event.sum(), (~observed_event).sum()) < 10:
            continue
        event_probability = np.mean(member_time <= lead, axis=1)
        observed_deficit = np.sum(np.maximum(threshold - truth, 0), axis=1)
        member_deficit = np.sum(
            np.maximum(threshold[:, None, :] - forecast, 0), axis=2
        )
        observed_duration = np.sum(observed_condition, axis=1).astype(float)
        member_duration = np.sum(forecast_condition, axis=2).astype(float)
        values = {
            "event_brier": float(np.mean((event_probability - observed_event) ** 2)),
            "timing_crps": float(np.mean(discrete_crps(
                member_time, observed_time, lead
            ))),
            "deficit_crps": float(np.mean(ensemble_crps(
                member_deficit, observed_deficit
            ))),
            "duration_crps": float(np.mean(ensemble_crps(
                member_duration, observed_duration
            ))),
        }
        for score, value in values.items():
            rows.append({
                "GAGE_ID": str(gage).zfill(8), "spatial_group": spatial_group,
                "variant": variant, "model": model, "score": score,
                "lead_days": int(lead), "n": int(eligible.sum()),
                "events": int(observed_event.sum()), "value": value,
                "threshold_mode": threshold_mode, "members": int(members),
                "shuffle_seed": int(shuffle_seed),
            })
    return rows


def worker(row, config: dict) -> tuple[pd.DataFrame, list[dict]]:
    gage = str(row.GAGE_ID).zfill(8)
    try:
        frame = load_daily(row.file)
        qfit = frame.loc[
            frame["date"].le(pd.Timestamp(config["fit_end"])), "q_mm_day"
        ]
        qfit = qfit[qfit.gt(0)]
        if qfit.empty:
            raise ValueError("no_scale")
        scale = float(qfit.median())
        history = frame.loc[
            frame["date"].le(pd.Timestamp(config["threshold_reference_end"])),
            "q_mm_day",
        ] / scale
        fixed = float(history.quantile(0.10))
        seasonal = seasonal_threshold_lookup(
            frame, scale, config["threshold_reference_end"], 0.10
        )
        rows: list[dict] = []

        settings = [
            ("base_age3_p0.1", 3, 0.1),
            ("dry_age1", 1, 0.1),
            ("dry_age7", 7, 0.1),
            ("precip_0.0", 3, 0.0),
            ("precip_1.0", 3, 1.0),
        ]
        base_train = base_test = None
        for label, age, precipitation in settings:
            local = dict(config)
            local["antecedent_days"] = age
            local["precipitation_threshold_mm_day"] = precipitation
            cases, _ = extended_cases(frame, scale, local)
            train = cases[
                cases["future_date"].le(pd.Timestamp(local["calibration_end"]))
            ].copy()
            test = cases[
                cases["date"].ge(pd.Timestamp(local["evaluation_start"]))
                & cases["future_date"].le(pd.Timestamp(local["evaluation_end"]))
            ].copy()
            if len(train) < 200 or len(test) < 50:
                continue
            if label == "base_age3_p0.1":
                base_train, base_test = train, test
            coherent = analog_paths(train, test, HYDROGRAPH_FEATURES, 101, True)
            shuffled = shuffle_members_by_day(
                coherent, int(config["random_seed"] + int(gage[-5:]) + age * 100 + precipitation * 10)
            )
            initial, future = threshold_paths(test, fixed, None)
            for model, paths in (("coherent", coherent), ("shuffled", shuffled)):
                rows.extend(score_paths(
                    gage, row.spatial_group, label, model, paths, test,
                    initial, future, "fixed", 101, 0,
                ))

        if base_train is None or base_test is None:
            raise ValueError("base_setting_unavailable")
        initial_fixed, future_fixed = threshold_paths(base_test, fixed, None)
        for members in (31, 51, 151):
            coherent = analog_paths(
                base_train, base_test, HYDROGRAPH_FEATURES, members, True
            )
            shuffled = shuffle_members_by_day(
                coherent, int(config["random_seed"] + int(gage[-5:]) + members)
            )
            label = f"members_{members}"
            for model, paths in (("coherent", coherent), ("shuffled", shuffled)):
                rows.extend(score_paths(
                    gage, row.spatial_group, label, model, paths, base_test,
                    initial_fixed, future_fixed, "fixed", members, 0,
                ))

        coherent = analog_paths(
            base_train, base_test, HYDROGRAPH_FEATURES, 101, True
        )
        for repeat in range(5):
            shuffled = shuffle_members_by_day(
                coherent, int(config["random_seed"] + int(gage[-5:]) + 500 + repeat)
            )
            label = f"shuffle_repeat_{repeat + 1}"
            rows.extend(score_paths(
                gage, row.spatial_group, label, "coherent", coherent, base_test,
                initial_fixed, future_fixed, "fixed", 101, repeat + 1,
            ))
            rows.extend(score_paths(
                gage, row.spatial_group, label, "shuffled", shuffled, base_test,
                initial_fixed, future_fixed, "fixed", 101, repeat + 1,
            ))

        initial_seasonal, future_seasonal = threshold_paths(
            base_test, fixed, seasonal
        )
        seasonal_shuffle = shuffle_members_by_day(
            coherent, int(config["random_seed"] + int(gage[-5:]) + 900)
        )
        for model, paths in (("coherent", coherent), ("shuffled", seasonal_shuffle)):
            rows.extend(score_paths(
                gage, row.spatial_group, "seasonal_q10", model, paths, base_test,
                initial_seasonal, future_seasonal, "seasonal", 101, 0,
            ))
        return pd.DataFrame(rows), []
    except Exception as error:
        return pd.DataFrame(), [{"GAGE_ID": gage, "reason": repr(error)}]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-basins", type=int)
    args = parser.parse_args()
    panel = pd.read_csv(
        RESULTS_ROOT / "01_panel" / "candidate_screening_panel.csv",
        dtype={"GAGE_ID": str},
    )
    if args.max_basins:
        panel = panel.head(int(args.max_basins)).copy()
    config = load_benchmark_config()
    frames: list[pd.DataFrame] = []
    failures: list[dict] = []
    for number, row in enumerate(panel.itertuples(index=False), start=1):
        frame, failed = worker(row, config)
        if not frame.empty:
            frames.append(frame)
        failures.extend(failed)
        if number % 4 == 0 or number == len(panel):
            print(f"Extended sensitivities {number}/{len(panel)} basins", flush=True)
    metrics = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    output = RESULTS_ROOT / "09_extended_forecast" / "sensitivities"
    atomic_csv(metrics, output / "sensitivity_metrics.csv.gz", compression="gzip")
    atomic_csv(pd.DataFrame(failures), output / "sensitivity_failures.csv")
    receipt = {
        "status": "complete", "basins_requested": int(len(panel)),
        "basins_with_results": int(metrics["GAGE_ID"].nunique() if not metrics.empty else 0),
        "variants": sorted(metrics["variant"].unique().tolist()) if not metrics.empty else [],
        "future_forcing_selection": False,
    }
    atomic_json(receipt, output / "_SUCCESS.json")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
