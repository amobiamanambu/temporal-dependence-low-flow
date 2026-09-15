"""Restartable confirmation for the frozen extended-range trajectory winner."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .common import (
    RESULTS_ROOT, atomic_csv, atomic_json, load_benchmark_config, load_daily,
)
from .extended_trajectory import (
    FLOW_SEASON_FEATURES, HYDROCLIMATE_FEATURES, HYDROGRAPH_FEATURES, LEADS,
    MEMBERS, SEASON_FEATURES, _direct_score_rows, _score_paths, analog_paths,
    extended_cases, shuffle_members_by_day,
)


FROZEN_MODELS = (
    "constant_persistence_path",
    "seasonal_climatology_path",
    "flow_season_analog",
    "hydrograph_analog",
    "hydrograph_marginal_shuffle",
    "hydroclimate_analog",
    "direct_endpoint_logistic",
    "direct_event_logistic",
)


def confirmation_worker(task) -> tuple[pd.DataFrame, list[dict]]:
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
        cases, _ = extended_cases(frame, scale, config)
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
            test["q"].to_numpy(float)[:, None, None], MEMBERS, axis=1
        ).repeat(max(LEADS), axis=2)
        rows.extend(_score_paths(
            gage, spatial_group, "constant_persistence_path", persistence,
            test, thresholds, config, False,
        ))
        del persistence

        for number, (name, features, scale_by_q) in enumerate((
            ("seasonal_climatology_path", SEASON_FEATURES, False),
            ("flow_season_analog", FLOW_SEASON_FEATURES, True),
            ("hydrograph_analog", HYDROGRAPH_FEATURES, True),
            ("hydroclimate_analog", HYDROCLIMATE_FEATURES, True),
        )):
            paths = analog_paths(train, test, features, MEMBERS, scale_by_q)
            rows.extend(_score_paths(
                gage, spatial_group, name, paths, test, thresholds, config, False,
            ))
            if name == "hydrograph_analog":
                seed = int(config["random_seed"] + int(str(gage)[-5:]) + 23000 + number)
                shuffled = shuffle_members_by_day(paths, seed)
                rows.extend(_score_paths(
                    gage, spatial_group, "hydrograph_marginal_shuffle", shuffled,
                    test, thresholds, config, False,
                ))
                del shuffled
            del paths

        rows.extend(_direct_score_rows(
            gage, spatial_group, train, test, thresholds, config
        ))
        return pd.DataFrame(rows), []
    except Exception as error:
        return pd.DataFrame(), [{
            "GAGE_ID": str(gage).zfill(8), "reason": repr(error)
        }]


def _read_metric_files(folder: Path) -> pd.DataFrame:
    paths = sorted(folder.glob("*_metrics.csv.gz"))
    frames = [pd.read_csv(path, dtype={"GAGE_ID": str}) for path in paths]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def run_confirmation_shard(panel: pd.DataFrame, shard_index: int, shard_count: int,
                           force: bool = False) -> dict:
    if shard_count < 1 or not 0 <= shard_index < shard_count:
        raise ValueError("shard_index must be in [0, shard_count)")
    config = load_benchmark_config()
    ordered = panel.sort_values("GAGE_ID").reset_index(drop=True)
    selected = ordered.iloc[np.arange(len(ordered)) % shard_count == shard_index].copy()
    output = RESULTS_ROOT / "09_extended_forecast" / f"confirmation_shard_{shard_index:02d}"
    by_basin = output / "by_basin"
    by_basin.mkdir(parents=True, exist_ok=True)
    failures: list[dict] = []
    completed = 0
    for number, row in enumerate(selected.itertuples(index=False), start=1):
        gage = str(row.GAGE_ID).zfill(8)
        metric_path = by_basin / f"{gage}_metrics.csv.gz"
        failure_path = by_basin / f"{gage}_failure.json"
        if not force and (metric_path.exists() or failure_path.exists()):
            completed += 1
        else:
            metrics, failed = confirmation_worker(
                (gage, row.file, row.spatial_group, config)
            )
            if not metrics.empty:
                atomic_csv(metrics, metric_path, compression="gzip")
                if failure_path.exists():
                    failure_path.unlink()
            else:
                record = failed[0] if failed else {
                    "GAGE_ID": gage, "reason": "no_scored_targets"
                }
                atomic_json(record, failure_path)
                failures.append(record)
            completed += 1
        if number % 10 == 0 or number == len(selected):
            print(
                f"Confirmation shard {shard_index + 1}/{shard_count}: "
                f"{number}/{len(selected)} basins",
                flush=True,
            )

    metrics = _read_metric_files(by_basin)
    failure_files = sorted(by_basin.glob("*_failure.json"))
    archived_failures = [
        json.loads(path.read_text(encoding="utf-8")) for path in failure_files
    ]
    atomic_csv(
        metrics, output / "extended_confirmation_metrics.csv.gz", compression="gzip"
    )
    atomic_csv(
        pd.DataFrame(archived_failures), output / "extended_confirmation_failures.csv"
    )
    receipt = {
        "status": "complete", "shard_index": int(shard_index),
        "shard_count": int(shard_count), "basins_requested": int(len(selected)),
        "basins_completed": int(completed),
        "basins_with_results": int(
            metrics["GAGE_ID"].nunique() if not metrics.empty else 0
        ),
        "failures": int(len(archived_failures)), "models": list(FROZEN_MODELS),
        "future_forcing_selection": False,
        "future_observations_used_by_eligible_models": False,
    }
    atomic_json(receipt, output / "_SUCCESS.json")
    return receipt
