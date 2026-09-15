#!/usr/bin/env python3
"""Test whether 105/120-day coherence gains survive predeclared alternatives."""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from lib.common import (  # noqa: E402
    RESULTS_ROOT, atomic_csv, atomic_json, ensemble_crps, load_benchmark_config,
    load_daily,
)
from lib.extended_120 import (  # noqa: E402
    EXTENSION_LEADS, HORIZON, OUTPUT_ROOT, analog_paths_120, extension_cases,
)
from lib.extended_trajectory import (  # noqa: E402
    HYDROGRAPH_FEATURES, discrete_crps, first_event_time,
    shuffle_members_by_day,
)


SCORES = ("event_brier", "timing_crps", "deficit_crps", "duration_crps")
SENSITIVITY_ROOT = OUTPUT_ROOT / "sensitivities"


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
        return (
            np.full(len(test), fixed, float),
            np.full((len(test), HORIZON), fixed, float),
        )
    dates = pd.to_datetime(test["date"])
    initial = seasonal[dates.dt.dayofyear.to_numpy(int)]
    future = np.column_stack([
        seasonal[(dates + pd.to_timedelta(day, unit="D")).dt.dayofyear.to_numpy(int)]
        for day in range(1, HORIZON + 1)
    ])
    return initial, future


def score_paths(gage: str, spatial_group: str, variant: str, model: str,
                paths: np.ndarray, test: pd.DataFrame,
                initial_threshold: np.ndarray, future_threshold: np.ndarray,
                threshold_mode: str, members: int,
                shuffle_seed: int) -> list[dict]:
    truth_all = test[[f"q_h{day}" for day in range(1, HORIZON + 1)]].to_numpy(float)
    eligible = test["q"].to_numpy(float) > initial_threshold
    if eligible.sum() < 50:
        return []
    rows: list[dict] = []
    for lead in EXTENSION_LEADS:
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
            "timing_crps": float(np.mean(discrete_crps(member_time, observed_time, lead))),
            "deficit_crps": float(np.mean(ensemble_crps(member_deficit, observed_deficit))),
            "duration_crps": float(np.mean(ensemble_crps(member_duration, observed_duration))),
        }
        for score, value in values.items():
            rows.append({
                "GAGE_ID": str(gage).zfill(8),
                "spatial_group": spatial_group,
                "variant": variant,
                "model": model,
                "score": score,
                "lead_days": int(lead),
                "n": int(eligible.sum()),
                "events": int(observed_event.sum()),
                "value": value,
                "threshold_mode": threshold_mode,
                "members": int(members),
                "shuffle_seed": int(shuffle_seed),
            })
    return rows


def _score_pair(rows: list[dict], gage: str, spatial_group: str, label: str,
                coherent: np.ndarray, test: pd.DataFrame,
                initial: np.ndarray, future: np.ndarray, mode: str,
                members: int, seed: int) -> None:
    shuffled = shuffle_members_by_day(coherent, seed)
    rows.extend(score_paths(
        gage, spatial_group, label, "coherent", coherent, test,
        initial, future, mode, members, seed,
    ))
    rows.extend(score_paths(
        gage, spatial_group, label, "shuffled", shuffled, test,
        initial, future, mode, members, seed,
    ))


def worker(task) -> tuple[str, pd.DataFrame, list[dict]]:
    gage, source, spatial_group, config = task
    gage = str(gage).zfill(8)
    try:
        frame = load_daily(source)
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
        base_train = base_test = base_coherent = None

        settings = [
            ("base_age3_p0.1", 3, 0.1),
            ("dry_age1", 1, 0.1),
            ("dry_age7", 7, 0.1),
            ("precip_0.0", 3, 0.0),
            ("precip_1.0", 3, 1.0),
        ]
        for label, age, precipitation in settings:
            local = dict(config)
            local["antecedent_days"] = age
            local["precipitation_threshold_mm_day"] = precipitation
            cases = extension_cases(frame, scale, local)
            train = cases[cases["future_date"].le(pd.Timestamp(local["calibration_end"]))].copy()
            test = cases[
                cases["date"].ge(pd.Timestamp(local["evaluation_start"]))
                & cases["future_date"].le(pd.Timestamp(local["evaluation_end"]))
            ].copy()
            if len(train) < 200 or len(test) < 50:
                continue
            coherent = analog_paths_120(
                train, test, HYDROGRAPH_FEATURES, True, members=101
            )
            initial, future = threshold_paths(test, fixed, None)
            seed = int(config["random_seed"] + int(gage[-5:]) + age * 100
                       + round(precipitation * 10) + 26000)
            _score_pair(rows, gage, spatial_group, label, coherent, test,
                        initial, future, "fixed", 101, seed)
            if label == "base_age3_p0.1":
                base_train, base_test, base_coherent = train, test, coherent

        if base_train is None or base_test is None or base_coherent is None:
            raise ValueError("base_setting_unavailable")
        initial_fixed, future_fixed = threshold_paths(base_test, fixed, None)
        for members in (31, 51, 151):
            coherent = analog_paths_120(
                base_train, base_test, HYDROGRAPH_FEATURES, True, members=members
            )
            label = f"members_{members}"
            seed = int(config["random_seed"] + int(gage[-5:]) + members + 27000)
            _score_pair(rows, gage, spatial_group, label, coherent, base_test,
                        initial_fixed, future_fixed, "fixed", members, seed)

        for repeat in range(1, 6):
            label = f"shuffle_repeat_{repeat}"
            seed = int(config["random_seed"] + int(gage[-5:]) + 28000 + repeat)
            _score_pair(rows, gage, spatial_group, label, base_coherent,
                        base_test, initial_fixed, future_fixed, "fixed", 101, seed)

        initial_seasonal, future_seasonal = threshold_paths(
            base_test, fixed, seasonal
        )
        seed = int(config["random_seed"] + int(gage[-5:]) + 29000)
        _score_pair(rows, gage, spatial_group, "seasonal_q10", base_coherent,
                    base_test, initial_seasonal, future_seasonal,
                    "seasonal", 101, seed)
        if not rows:
            raise ValueError("no_scored_sensitivity_targets")
        return gage, pd.DataFrame(rows), []
    except Exception as error:
        return gage, pd.DataFrame(), [{"GAGE_ID": gage, "reason": repr(error)}]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-basins", type=int)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    args = parser.parse_args()
    panel = pd.read_csv(
        RESULTS_ROOT / "01_panel" / "candidate_screening_panel.csv",
        dtype={"GAGE_ID": str},
    )
    if args.max_basins:
        panel = panel.head(int(args.max_basins)).copy()
    if args.shard_count < 1 or not 0 <= args.shard_index < args.shard_count:
        raise ValueError("shard-index must be in [0, shard-count)")
    ordered = panel.sort_values("GAGE_ID").reset_index(drop=True)
    panel = ordered.iloc[
        np.arange(len(ordered)) % int(args.shard_count) == int(args.shard_index)
    ].copy()
    config = load_benchmark_config()
    output = (
        SENSITIVITY_ROOT
        if int(args.shard_count) == 1
        else SENSITIVITY_ROOT / f"sensitivity_shard_{int(args.shard_index):02d}"
    )
    by_basin = output / "by_basin"
    by_basin.mkdir(parents=True, exist_ok=True)
    tasks = []
    for row in panel.itertuples(index=False):
        gage = str(row.GAGE_ID).zfill(8)
        metric = by_basin / f"{gage}_metrics.csv.gz"
        failure = by_basin / f"{gage}_failure.json"
        if args.force or not (metric.exists() or failure.exists()):
            tasks.append((gage, row.file, row.spatial_group, config))

    if tasks and int(args.workers) == 1:
        for number, task in enumerate(tasks, start=1):
            gage, frame, failed = worker(task)
            metric = by_basin / f"{gage}_metrics.csv.gz"
            failure = by_basin / f"{gage}_failure.json"
            if not frame.empty:
                atomic_csv(frame, metric, compression="gzip")
                failure.unlink(missing_ok=True)
            else:
                atomic_json(failed[0], failure)
            if number % 4 == 0 or number == len(tasks):
                print(f"120-day sensitivities {number}/{len(tasks)} basins", flush=True)
    elif tasks:
        with ProcessPoolExecutor(max_workers=max(1, int(args.workers))) as pool:
            futures = [pool.submit(worker, task) for task in tasks]
            for number, future in enumerate(as_completed(futures), start=1):
                gage, frame, failed = future.result()
                metric = by_basin / f"{gage}_metrics.csv.gz"
                failure = by_basin / f"{gage}_failure.json"
                if not frame.empty:
                    atomic_csv(frame, metric, compression="gzip")
                    failure.unlink(missing_ok=True)
                else:
                    atomic_json(failed[0], failure)
                if number % 4 == 0 or number == len(tasks):
                    print(f"120-day sensitivities {number}/{len(tasks)} basins", flush=True)

    frames = [
        pd.read_csv(path, dtype={"GAGE_ID": str})
        for path in sorted(by_basin.glob("*_metrics.csv.gz"))
    ]
    metrics = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    failures = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(by_basin.glob("*_failure.json"))
    ]
    atomic_csv(metrics, output / "sensitivity_metrics.csv.gz", compression="gzip")
    atomic_csv(pd.DataFrame(failures), output / "sensitivity_failures.csv")
    receipt = {
        "status": "complete",
        "shard_index": int(args.shard_index),
        "shard_count": int(args.shard_count),
        "basins_requested": int(len(panel)),
        "basins_with_results": int(metrics["GAGE_ID"].nunique() if not metrics.empty else 0),
        "failures": int(len(failures)),
        "leads": list(EXTENSION_LEADS),
        "variants": sorted(metrics["variant"].unique().tolist()) if not metrics.empty else [],
        "future_forcing_selection": False,
        "separate_from_1_90_archive": True,
    }
    atomic_json(receipt, output / "_SUCCESS.json")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
