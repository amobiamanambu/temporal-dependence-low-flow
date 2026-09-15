#!/usr/bin/env python3
"""Export restartable issue-level Q10 low-flow forecasts for 30--90 days.

This script reproduces the frozen hydrograph-state analog used in the paper.
It writes one Parquet file per basin so an interrupted continental run can be
resumed without recomputing completed basins.  The companion Stage 31 script
assembles these files into a versioned public data package.

Run from the project root, for example:

    python temporal_coherence_paper/scripts/30_export_public_forecast_dataset.py --workers 4

The output is retrospective research forecasts for the held-out 2016--2025
period.  It is not an operational USGS forecast product.
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


PROJECT_ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_ROOT = PROJECT_ROOT / "lowflow_forecast_benchmark"
if str(BENCHMARK_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(BENCHMARK_ROOT / "scripts"))

from lib.common import all_accepted_inventory, load_benchmark_config, load_daily  # noqa: E402
from lib.extended_trajectory import (  # noqa: E402
    HYDROGRAPH_FEATURES,
    MEMBERS,
    analog_paths,
    extended_cases,
    first_event_time,
)


OUTPUT_ROOT = PROJECT_ROOT / "temporal_coherence_paper" / "data_release" / "v1"
WORK_ROOT = OUTPUT_ROOT / "work" / "by_basin"
DEVELOPMENT_PANEL = (
    BENCHMARK_ROOT / "results" / "01_panel" / "candidate_screening_panel.csv"
)
ARCHIVED_METRIC_ROOT = BENCHMARK_ROOT / "results" / "09_extended_forecast"
LEADS = (30, 45, 60, 90)
QUANTILE = 0.10
VERSION = "1.0.0"


def normalize_gage(values: pd.Series) -> pd.Series:
    return values.astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(8)


def atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    frame.to_parquet(temporary, index=False, compression="zstd")
    temporary.replace(path)


def atomic_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def qtile(values: np.ndarray, probability: float) -> np.ndarray:
    return np.quantile(values, probability, axis=1).astype(np.float32)


def basin_forecasts(task: tuple[dict, dict, str]) -> dict:
    """Recreate the frozen Q10 onset ensemble and write one basin file."""
    row, config, output_name = task
    gage = str(row["GAGE_ID"]).zfill(8)
    output_path = WORK_ROOT / output_name
    failure_path = WORK_ROOT / f"{gage}_failure.json"
    try:
        frame = load_daily(row["file"])
        fit_end = pd.Timestamp(config["fit_end"])
        threshold_end = pd.Timestamp(config["threshold_reference_end"])
        qfit = frame.loc[frame["date"].le(fit_end), "q_mm_day"]
        qfit = qfit[qfit.gt(0)]
        if qfit.empty:
            raise ValueError("no_positive_training_scale")
        scale = float(qfit.median())

        history = frame.loc[frame["date"].le(threshold_end), "q_mm_day"] / scale
        threshold = float(history.quantile(QUANTILE))
        if not np.isfinite(threshold) or threshold <= 0:
            raise ValueError("invalid_Q10_threshold")

        cases, _ = extended_cases(frame, scale, config)
        train = cases[
            cases["future_date"].le(pd.Timestamp(config["calibration_end"]))
        ].copy()
        test = cases[
            cases["date"].ge(pd.Timestamp(config["evaluation_start"]))
            & cases["future_date"].le(pd.Timestamp(config["evaluation_end"]))
        ].copy()
        if len(train) < int(config["minimum_trajectory_fit_cases"]):
            raise ValueError("insufficient_training_initializations")
        if len(test) < int(config["minimum_trajectory_evaluation_cases"]):
            raise ValueError("insufficient_test_initializations")

        onset = test["q"].to_numpy(float) > threshold
        test = test.loc[onset].reset_index(drop=True)
        if len(test) < int(config["minimum_trajectory_evaluation_cases"]):
            raise ValueError("insufficient_Q10_onset_initializations")

        paths = analog_paths(
            train,
            test,
            HYDROGRAPH_FEATURES,
            members=MEMBERS,
            scale_by_initial_q=True,
        ).astype(np.float32, copy=False)
        truth_all = test[[f"q_h{day}" for day in range(1, 91)]].to_numpy(np.float32)
        frames: list[pd.DataFrame] = []
        for lead in LEADS:
            truth = truth_all[:, :lead]
            forecast = paths[:, :, :lead]
            observed_below = truth <= threshold
            forecast_below = forecast <= threshold
            observed_event = observed_below.any(axis=1)
            event_probability = forecast_below.any(axis=2).mean(axis=1)
            observed_endpoint = observed_below[:, -1]
            endpoint_probability = forecast_below[:, :, -1].mean(axis=1)

            observed_time_censored = first_event_time(truth, threshold, "onset")
            member_time = first_event_time(forecast, threshold, "onset")
            observed_duration = observed_below.sum(axis=1).astype(np.int16)
            member_duration = forecast_below.sum(axis=2).astype(np.int16)
            observed_deficit = (
                np.maximum(threshold - truth, 0).sum(axis=1) * scale
            ).astype(np.float32)
            member_deficit = (
                np.maximum(threshold - forecast, 0).sum(axis=2) * scale
            ).astype(np.float32)
            observed_minimum = (truth.min(axis=1) * scale).astype(np.float32)
            member_minimum = (forecast.min(axis=2) * scale).astype(np.float32)

            events = int(observed_event.sum())
            nonevents = int((~observed_event).sum())
            base = pd.DataFrame({
                "data_version": VERSION,
                "GAGE_ID": gage,
                "issue_date": pd.to_datetime(test["date"]).dt.normalize(),
                "valid_end_date": pd.to_datetime(test["date"]).dt.normalize()
                + pd.to_timedelta(lead, unit="D"),
                "lead_days": np.int16(lead),
                "threshold_name": "Q10",
                "target": "onset",
                "model": "hydrograph_analog",
                "spatial_group": str(row.get("spatial_group", "Unknown")),
                "quality_tier": str(row.get("quality_tier", "")),
                "is_reference": bool(row.get("is_reference", False)),
                "record_years": np.float32(row.get("record_years", np.nan)),
                "latitude": np.float32(row.get("LAT_GAGE", np.nan)),
                "longitude": np.float32(row.get("LNG_GAGE", np.nan)),
                "initial_flow_mm_day": (test["q"].to_numpy(float) * scale).astype(np.float32),
                "threshold_q10_mm_day": np.float32(threshold * scale),
                "dry_spell_age_days": test["dry_spell_age"].to_numpy(np.float32),
                "ensemble_members": np.int16(MEMBERS),
                "lead_is_scorable": bool(
                    events >= int(config["minimum_events"])
                    and nonevents >= int(config["minimum_nonevents"])
                ),
                "observed_event": observed_event,
                "forecast_event_probability": event_probability.astype(np.float32),
                "observed_endpoint_low": observed_endpoint,
                "forecast_endpoint_probability": endpoint_probability.astype(np.float32),
                "observed_onset_day": np.where(
                    observed_event, observed_time_censored, np.nan
                ).astype(np.float32),
                "observed_onset_day_censored": observed_time_censored.astype(np.int16),
                "forecast_onset_p05": qtile(member_time, 0.05),
                "forecast_onset_p50": qtile(member_time, 0.50),
                "forecast_onset_p95": qtile(member_time, 0.95),
                "observed_duration_days": observed_duration,
                "forecast_duration_p05": qtile(member_duration, 0.05),
                "forecast_duration_p50": qtile(member_duration, 0.50),
                "forecast_duration_p95": qtile(member_duration, 0.95),
                "observed_deficit_mm": observed_deficit,
                "forecast_deficit_mm_p05": qtile(member_deficit, 0.05),
                "forecast_deficit_mm_p50": qtile(member_deficit, 0.50),
                "forecast_deficit_mm_p95": qtile(member_deficit, 0.95),
                "observed_minimum_flow_mm_day": observed_minimum,
                "forecast_minimum_flow_mm_day_p05": qtile(member_minimum, 0.05),
                "forecast_minimum_flow_mm_day_p50": qtile(member_minimum, 0.50),
                "forecast_minimum_flow_mm_day_p95": qtile(member_minimum, 0.95),
            })
            frames.append(base)

        output = pd.concat(frames, ignore_index=True)
        output = output.sort_values(["issue_date", "lead_days"]).reset_index(drop=True)
        atomic_parquet(output, output_path)
        failure_path.unlink(missing_ok=True)
        return {
            "GAGE_ID": gage,
            "status": "complete",
            "rows": int(len(output)),
            "issues": int(output["issue_date"].nunique()),
        }
    except Exception as error:
        payload = {"GAGE_ID": gage, "status": "excluded", "reason": repr(error)}
        atomic_json(payload, failure_path)
        output_path.unlink(missing_ok=True)
        return payload


def confirmation_inventory(max_basins: int | None = None) -> pd.DataFrame:
    inventory = all_accepted_inventory().copy()
    attributes = pd.read_csv(
        PROJECT_ROOT / "continental_run" / "06_basin_index" / "basin_index.csv",
        usecols=["GAGE_ID", "LAT_GAGE", "LNG_GAGE"],
        dtype={"GAGE_ID": str},
    )
    inventory["GAGE_ID"] = normalize_gage(inventory["GAGE_ID"])
    attributes["GAGE_ID"] = normalize_gage(attributes["GAGE_ID"])
    inventory = inventory.merge(attributes, on="GAGE_ID", how="left")
    development = pd.read_csv(DEVELOPMENT_PANEL, dtype={"GAGE_ID": str})
    development_ids = set(normalize_gage(development["GAGE_ID"]))
    inventory = inventory[~inventory["GAGE_ID"].isin(development_ids)]
    inventory = inventory.sort_values("GAGE_ID").reset_index(drop=True)
    if max_basins is not None:
        inventory = inventory.head(int(max_basins)).copy()
    return inventory


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--max-basins", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    WORK_ROOT.mkdir(parents=True, exist_ok=True)
    inventory = confirmation_inventory(args.max_basins)
    config = load_benchmark_config()
    tasks: list[tuple[dict, dict, str]] = []
    skipped = 0
    for row in inventory.to_dict("records"):
        gage = str(row["GAGE_ID"]).zfill(8)
        output_name = f"{gage}_forecast.parquet"
        output_path = WORK_ROOT / output_name
        failure_path = WORK_ROOT / f"{gage}_failure.json"
        if not args.force and (output_path.exists() or failure_path.exists()):
            skipped += 1
            continue
        tasks.append((row, config, output_name))

    print(
        f"Stage 30: {len(inventory):,} confirmation basins; "
        f"{skipped:,} already recorded; {len(tasks):,} to process",
        flush=True,
    )
    results: list[dict] = []
    if int(args.workers) <= 1:
        for number, task in enumerate(tasks, start=1):
            results.append(basin_forecasts(task))
            if number % 25 == 0 or number == len(tasks):
                print(f"Completed {number:,}/{len(tasks):,}", flush=True)
    else:
        try:
            executor = ProcessPoolExecutor(max_workers=int(args.workers))
        except (PermissionError, OSError):
            # Some managed macOS environments deny the semaphore-limit query
            # used by ProcessPoolExecutor. NumPy and scikit-learn release the
            # GIL for the expensive operations, so a thread pool is a safe
            # fallback and preserves identical numerical results.
            print("Process workers unavailable; using thread workers", flush=True)
            executor = ThreadPoolExecutor(max_workers=int(args.workers))
        with executor:
            futures = [executor.submit(basin_forecasts, task) for task in tasks]
            for number, future in enumerate(as_completed(futures), start=1):
                results.append(future.result())
                if number % 25 == 0 or number == len(futures):
                    print(f"Completed {number:,}/{len(futures):,}", flush=True)

    receipt = {
        "stage": 30,
        "status": "complete",
        "data_version": VERSION,
        "confirmation_basins": int(len(inventory)),
        "newly_processed": int(len(tasks)),
        "previously_recorded": int(skipped),
        "new_successes": int(sum(r.get("status") == "complete" for r in results)),
        "new_exclusions": int(sum(r.get("status") == "excluded" for r in results)),
        "forecast_model": "hydrograph_analog",
        "threshold": "Q10",
        "leads_days": list(LEADS),
        "evaluation_period": [config["evaluation_start"], config["evaluation_end"]],
        "future_forcing_used": False,
    }
    atomic_json(receipt, OUTPUT_ROOT / "STAGE30_SUCCESS.json")
    print(json.dumps(receipt, indent=2), flush=True)


if __name__ == "__main__":
    main()
