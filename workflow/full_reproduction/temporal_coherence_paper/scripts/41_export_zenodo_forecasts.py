#!/usr/bin/env python3
"""Export final issue-level Q10 forecasts for the Zenodo data release.

The export uses the frozen 120-day hydrograph-state analog and the 3,402-basin
sample retained in the final temporal-dependence analysis.  One restartable
Parquet file is written per basin.  Stage 42 assembles these files into six
lead-specific tables and validates them against the archived basin scores.

Run from the project root:

    python temporal_coherence_paper/scripts/41_export_zenodo_forecasts.py --workers 4

The outputs are retrospective hindcasts for research and verification.  They
are not operational USGS forecasts.
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
BENCHMARK_SCRIPTS = BENCHMARK_ROOT / "scripts"
if str(BENCHMARK_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_SCRIPTS))

from lib.common import all_accepted_inventory, load_benchmark_config, load_daily  # noqa: E402
from lib.extended_120 import MEMBERS, analog_paths_120, extension_cases  # noqa: E402
from lib.extended_trajectory import HYDROGRAPH_FEATURES, first_event_time  # noqa: E402


PAPER_ROOT = PROJECT_ROOT / "temporal_coherence_paper"
FINAL_METRICS = (
    PAPER_ROOT
    / "reviewer_strengthening"
    / "full"
    / "reviewer_strengthening_metrics.csv.gz"
)
DEFAULT_OUTPUT_ROOT = PAPER_ROOT / "zenodo_deposit" / "work_v1"
RELEASE_LEADS = (30, 45, 60, 90, 105, 120)
QUANTILE = 0.10
DATA_VERSION = "1.0.0"
EXPERIMENT = "frozen_120_day_common_support"


def normalize_gage(values: pd.Series) -> pd.Series:
    """Return USGS identifiers as text, padded to at least eight characters."""
    return values.astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(8)


def atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    """Write a Parquet file atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    frame.to_parquet(temporary, index=False, compression="zstd")
    temporary.replace(path)


def atomic_json(payload: dict, path: Path) -> None:
    """Write JSON atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def row_quantile(values: np.ndarray, probability: float) -> np.ndarray:
    """Calculate a quantile across ensemble members for every initialization."""
    return np.quantile(values, probability, axis=1).astype(np.float32)


def final_inventory(max_basins: int | None = None) -> pd.DataFrame:
    """Return the basins represented in the final six-window experiment."""
    if not FINAL_METRICS.exists():
        raise FileNotFoundError(f"Missing final metric archive: {FINAL_METRICS}")
    metrics = pd.read_csv(
        FINAL_METRICS,
        usecols=["aggregation", "GAGE_ID", "model", "score"],
        dtype={"GAGE_ID": str},
    )
    selected = metrics[
        metrics["aggregation"].eq("basin")
        & metrics["model"].eq("hydrograph_analog")
        & metrics["score"].eq("event_brier")
    ].copy()
    selected["GAGE_ID"] = normalize_gage(selected["GAGE_ID"])
    valid_ids = set(selected["GAGE_ID"])
    inventory = all_accepted_inventory().copy()
    inventory["GAGE_ID"] = normalize_gage(inventory["GAGE_ID"])
    inventory = inventory[inventory["GAGE_ID"].isin(valid_ids)]
    inventory = inventory.sort_values("GAGE_ID").reset_index(drop=True)
    if inventory["GAGE_ID"].nunique() != len(valid_ids):
        missing = sorted(valid_ids.difference(set(inventory["GAGE_ID"])))
        raise RuntimeError(f"Final metric basins missing from inventory: {missing[:10]}")
    if max_basins is not None:
        inventory = inventory.head(int(max_basins)).copy()
    return inventory


def basin_forecasts(task: tuple[dict, dict, str]) -> dict:
    """Recreate all six final forecast windows for one basin."""
    row, config, output_root_text = task
    output_root = Path(output_root_text)
    by_basin = output_root / "by_basin"
    gage = str(row["GAGE_ID"]).zfill(8)
    output_path = by_basin / f"{gage}_forecast.parquet"
    failure_path = by_basin / f"{gage}_failure.json"
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

        cases = extension_cases(frame, scale, config)
        train = cases[
            cases["future_date"].le(pd.Timestamp(config["calibration_end"]))
        ].copy()
        test = cases[
            cases["date"].ge(pd.Timestamp(config["evaluation_start"]))
            & cases["future_date"].le(pd.Timestamp(config["evaluation_end"]))
            & cases["q"].gt(threshold)
        ].copy().reset_index(drop=True)
        if len(train) < int(config["minimum_trajectory_fit_cases"]):
            raise ValueError("insufficient_training_initializations")
        if len(test) < int(config["minimum_trajectory_evaluation_cases"]):
            raise ValueError("insufficient_test_initializations")

        paths = analog_paths_120(
            train,
            test,
            HYDROGRAPH_FEATURES,
            scale_by_initial_q=True,
            members=MEMBERS,
        ).astype(np.float32, copy=False)
        truth_all = test[
            [f"q_h{day}" for day in range(1, 121)]
        ].to_numpy(np.float32)
        issue_dates = pd.to_datetime(test["date"]).dt.normalize()
        output_frames: list[pd.DataFrame] = []

        for lead in RELEASE_LEADS:
            truth = truth_all[:, :lead]
            forecast = paths[:, :, :lead]
            observed_below = truth <= threshold
            forecast_below = forecast <= threshold
            observed_event = observed_below.any(axis=1)
            forecast_event_probability = forecast_below.any(axis=2).mean(axis=1)
            observed_endpoint = observed_below[:, -1]
            forecast_endpoint_probability = forecast_below[:, :, -1].mean(axis=1)

            observed_onset_censored = first_event_time(truth, threshold, "onset")
            member_onset = first_event_time(forecast, threshold, "onset")
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
            output_frames.append(pd.DataFrame({
                "data_version": DATA_VERSION,
                "experiment": EXPERIMENT,
                "GAGE_ID": gage,
                "initialization_date": issue_dates,
                "valid_end_date": issue_dates + pd.to_timedelta(lead, unit="D"),
                "lead_days": np.int16(lead),
                "threshold_name": "Q10",
                "target": "onset",
                "model": "hydrograph_analog",
                "spatial_group": str(row.get("spatial_group", "Unknown")),
                "quality_tier": str(row.get("quality_tier", "")),
                "is_reference": bool(row.get("is_reference", False)),
                "record_years": np.float32(row.get("record_years", np.nan)),
                "initial_flow_mm_day": (
                    test["q"].to_numpy(float) * scale
                ).astype(np.float32),
                "threshold_q10_mm_day": np.float32(threshold * scale),
                "dry_spell_age_days": test["dry_spell_age"].to_numpy(np.float32),
                "ensemble_members": np.int16(MEMBERS),
                "lead_is_scorable": bool(
                    events >= int(config["minimum_events"])
                    and nonevents >= int(config["minimum_nonevents"])
                ),
                "observed_event": observed_event,
                "forecast_event_probability": forecast_event_probability.astype(np.float32),
                "observed_endpoint_low": observed_endpoint,
                "forecast_endpoint_probability": forecast_endpoint_probability.astype(np.float32),
                "observed_onset_day": np.where(
                    observed_event, observed_onset_censored, np.nan
                ).astype(np.float32),
                "observed_onset_day_censored": observed_onset_censored.astype(np.int16),
                "forecast_onset_p05": row_quantile(member_onset, 0.05),
                "forecast_onset_p50": row_quantile(member_onset, 0.50),
                "forecast_onset_p95": row_quantile(member_onset, 0.95),
                "observed_duration_days": observed_duration,
                "forecast_duration_p05": row_quantile(member_duration, 0.05),
                "forecast_duration_p50": row_quantile(member_duration, 0.50),
                "forecast_duration_p95": row_quantile(member_duration, 0.95),
                "observed_deficit_mm": observed_deficit,
                "forecast_deficit_mm_p05": row_quantile(member_deficit, 0.05),
                "forecast_deficit_mm_p50": row_quantile(member_deficit, 0.50),
                "forecast_deficit_mm_p95": row_quantile(member_deficit, 0.95),
                "observed_minimum_flow_mm_day": observed_minimum,
                "forecast_minimum_flow_mm_day_p05": row_quantile(member_minimum, 0.05),
                "forecast_minimum_flow_mm_day_p50": row_quantile(member_minimum, 0.50),
                "forecast_minimum_flow_mm_day_p95": row_quantile(member_minimum, 0.95),
            }))

        output = pd.concat(output_frames, ignore_index=True)
        output = output.sort_values(
            ["initialization_date", "lead_days"]
        ).reset_index(drop=True)
        atomic_parquet(output, output_path)
        failure_path.unlink(missing_ok=True)
        return {
            "GAGE_ID": gage,
            "status": "complete",
            "rows": int(len(output)),
            "initializations": int(output["initialization_date"].nunique()),
        }
    except Exception as error:  # basin-level exclusions are recorded, not hidden
        payload = {"GAGE_ID": gage, "status": "excluded", "reason": repr(error)}
        atomic_json(payload, failure_path)
        output_path.unlink(missing_ok=True)
        return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--max-basins", type=int)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = args.output_root.resolve()
    by_basin = output_root / "by_basin"
    by_basin.mkdir(parents=True, exist_ok=True)
    inventory = final_inventory(args.max_basins)
    config = load_benchmark_config()

    tasks: list[tuple[dict, dict, str]] = []
    retained = 0
    for row in inventory.to_dict("records"):
        gage = str(row["GAGE_ID"]).zfill(8)
        output = by_basin / f"{gage}_forecast.parquet"
        failure = by_basin / f"{gage}_failure.json"
        if not args.force and (output.exists() or failure.exists()):
            retained += 1
        else:
            tasks.append((row, config, str(output_root)))

    print(
        f"Stage 41: {len(inventory):,} final basins; {retained:,} retained; "
        f"{len(tasks):,} to process",
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
            print("Process workers unavailable; using thread workers", flush=True)
            executor = ThreadPoolExecutor(max_workers=int(args.workers))
        with executor:
            futures = [executor.submit(basin_forecasts, task) for task in tasks]
            for number, future in enumerate(as_completed(futures), start=1):
                results.append(future.result())
                if number % 25 == 0 or number == len(futures):
                    print(f"Completed {number:,}/{len(futures):,}", flush=True)

    completed_files = list(by_basin.glob("*_forecast.parquet"))
    failure_files = list(by_basin.glob("*_failure.json"))
    receipt = {
        "stage": 41,
        "status": "complete",
        "data_version": DATA_VERSION,
        "experiment": EXPERIMENT,
        "requested_basins": int(len(inventory)),
        "completed_basins": int(len(completed_files)),
        "excluded_basins": int(len(failure_files)),
        "new_successes": int(sum(item.get("status") == "complete" for item in results)),
        "new_exclusions": int(sum(item.get("status") == "excluded" for item in results)),
        "retained_records": int(retained),
        "leads_days": list(RELEASE_LEADS),
        "threshold": "Q10",
        "forecast_model": "hydrograph_analog",
        "evaluation_period": [config["evaluation_start"], config["evaluation_end"]],
        "common_case_requirement": "complete 120-day observed path",
        "future_forcing_used": False,
    }
    atomic_json(receipt, output_root / "STAGE41_SUCCESS.json")
    print(json.dumps(receipt, indent=2), flush=True)
    if len(completed_files) + len(failure_files) != len(inventory):
        raise RuntimeError("Stage 41 did not record every requested basin")


if __name__ == "__main__":
    main()
