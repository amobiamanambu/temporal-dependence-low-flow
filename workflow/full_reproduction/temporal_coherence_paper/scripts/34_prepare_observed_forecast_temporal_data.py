#!/usr/bin/env python3
"""Prepare basin and ecoregion forecast-versus-observation summaries.

The script reproduces the frozen 120-day hydrograph-state analog once per
eligible confirmation basin.  It saves compact summaries of forecast Q10
occurrence probability and the corresponding USGS-observed event frequency at
all evaluated windows.  Basin files make the stage restartable.

Run from the project root:

    python temporal_coherence_paper/scripts/34_prepare_observed_forecast_temporal_data.py --workers 4
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


PAPER_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = PAPER_ROOT.parent
BENCHMARK_ROOT = PROJECT_ROOT / "lowflow_forecast_benchmark"
BENCHMARK_SCRIPTS = BENCHMARK_ROOT / "scripts"
if str(BENCHMARK_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_SCRIPTS))

from lib.common import all_accepted_inventory, load_benchmark_config, load_daily  # noqa: E402
from lib.extended_120 import (  # noqa: E402
    LEADS_120, MEMBERS, analog_paths_120, extension_cases,
)
from lib.extended_trajectory import HYDROGRAPH_FEATURES  # noqa: E402


OUTPUT_ROOT = PAPER_ROOT / "temporal_validation"
BY_BASIN = OUTPUT_ROOT / "by_basin"
METRICS_FILE = (
    BENCHMARK_ROOT / "results" / "10_extension_120" / "extension_120_metrics.csv.gz"
)
SUMMARY_FILE = OUTPUT_ROOT / "basin_horizon_calibration.csv.gz"
ECOREGION_FILE = OUTPUT_ROOT / "ecoregion_horizon_calibration.csv"
RECEIPT_FILE = OUTPUT_ROOT / "STAGE34_SUCCESS.json"
QUANTILE = 0.10
BOOTSTRAP_REPLICATES = 2000
RANDOM_SEED = 24013


def normalize_gage(values: pd.Series) -> pd.Series:
    return values.astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(8)


def atomic_csv(frame: pd.DataFrame, path: Path, compression: str | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    frame.to_csv(temporary, index=False, compression=compression)
    temporary.replace(path)


def atomic_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def eligible_inventory(max_basins: int | None = None) -> pd.DataFrame:
    metrics = pd.read_csv(METRICS_FILE, dtype={"GAGE_ID": str})
    valid = metrics[
        metrics["model"].eq("hydrograph_analog")
        & metrics["threshold_name"].eq("Q10")
        & metrics["target"].eq("onset")
    ].copy()
    valid["GAGE_ID"] = normalize_gage(valid["GAGE_ID"])
    valid_ids = set(valid["GAGE_ID"])
    inventory = all_accepted_inventory().copy()
    inventory["GAGE_ID"] = normalize_gage(inventory["GAGE_ID"])
    inventory = inventory[inventory["GAGE_ID"].isin(valid_ids)]
    inventory = inventory.sort_values("GAGE_ID").reset_index(drop=True)
    if max_basins is not None:
        inventory = inventory.head(int(max_basins)).copy()
    return inventory


def basin_summary(task: tuple[dict, dict]) -> dict:
    row, config = task
    gage = str(row["GAGE_ID"]).zfill(8)
    output = BY_BASIN / f"{gage}_calibration.csv.gz"
    failure = BY_BASIN / f"{gage}_failure.json"
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
        train = cases[cases["future_date"].le(pd.Timestamp(config["calibration_end"]))].copy()
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
            train, test, HYDROGRAPH_FEATURES, scale_by_initial_q=True,
            members=MEMBERS,
        )
        truth = test[[f"q_h{day}" for day in range(1, 121)]].to_numpy(np.float32)
        rows = []
        for lead in LEADS_120:
            observed = (truth[:, :lead] <= threshold).any(axis=1)
            probability = (paths[:, :, :lead] <= threshold).any(axis=2).mean(axis=1)
            rows.append({
                "GAGE_ID": gage,
                "spatial_group": str(row.get("spatial_group", "Unknown")),
                "quality_tier": str(row.get("quality_tier", "")),
                "is_reference": bool(row.get("is_reference", False)),
                "lead_days": int(lead),
                "n_issues": int(len(test)),
                "events": int(observed.sum()),
                "observed_event_frequency": float(observed.mean()),
                "mean_forecast_probability": float(probability.mean()),
                "forecast_minus_observed": float(probability.mean() - observed.mean()),
                "event_brier": float(np.mean((probability - observed.astype(float)) ** 2)),
            })
        atomic_csv(pd.DataFrame(rows), output, compression="gzip")
        failure.unlink(missing_ok=True)
        return {"GAGE_ID": gage, "status": "complete"}
    except Exception as error:
        output.unlink(missing_ok=True)
        atomic_json({"GAGE_ID": gage, "reason": repr(error)}, failure)
        return {"GAGE_ID": gage, "status": "excluded", "reason": repr(error)}


def assemble(inventory: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    frames = []
    for gage in inventory["GAGE_ID"]:
        path = BY_BASIN / f"{gage}_calibration.csv.gz"
        if path.exists():
            frames.append(pd.read_csv(path, dtype={"GAGE_ID": str}))
    if not frames:
        raise RuntimeError("No completed basin calibration files were found")
    basins = pd.concat(frames, ignore_index=True)
    basins["GAGE_ID"] = normalize_gage(basins["GAGE_ID"])
    atomic_csv(basins, SUMMARY_FILE, compression="gzip")

    rng = np.random.default_rng(RANDOM_SEED)
    records = []
    for (region, lead), group in basins.groupby(["spatial_group", "lead_days"], sort=True):
        values = group[["mean_forecast_probability", "observed_event_frequency"]].to_numpy(float)
        indices = rng.integers(0, len(values), size=(BOOTSTRAP_REPLICATES, len(values)))
        replicates = values[indices].mean(axis=1)
        difference = replicates[:, 0] - replicates[:, 1]
        records.append({
            "spatial_group": region,
            "lead_days": int(lead),
            "basins": int(len(group)),
            "issues": int(group["n_issues"].sum()),
            "mean_forecast_probability": float(values[:, 0].mean()),
            "forecast_ci_low": float(np.quantile(replicates[:, 0], 0.025)),
            "forecast_ci_high": float(np.quantile(replicates[:, 0], 0.975)),
            "observed_event_frequency": float(values[:, 1].mean()),
            "observed_ci_low": float(np.quantile(replicates[:, 1], 0.025)),
            "observed_ci_high": float(np.quantile(replicates[:, 1], 0.975)),
            "forecast_minus_observed": float((values[:, 0] - values[:, 1]).mean()),
            "difference_ci_low": float(np.quantile(difference, 0.025)),
            "difference_ci_high": float(np.quantile(difference, 0.975)),
        })
    ecoregions = pd.DataFrame(records)
    atomic_csv(ecoregions, ECOREGION_FILE)
    return basins, ecoregions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--max-basins", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    BY_BASIN.mkdir(parents=True, exist_ok=True)
    inventory = eligible_inventory(args.max_basins)
    config = load_benchmark_config()
    tasks = []
    retained = 0
    for row in inventory.to_dict("records"):
        gage = str(row["GAGE_ID"]).zfill(8)
        output = BY_BASIN / f"{gage}_calibration.csv.gz"
        failure = BY_BASIN / f"{gage}_failure.json"
        if not args.force and (output.exists() or failure.exists()):
            retained += 1
        else:
            tasks.append((row, config))
    print(
        f"Stage 34: {len(inventory):,} eligible basins; {retained:,} retained; "
        f"{len(tasks):,} to process",
        flush=True,
    )

    results = []
    if int(args.workers) <= 1:
        for number, task in enumerate(tasks, 1):
            results.append(basin_summary(task))
            if number % 25 == 0 or number == len(tasks):
                print(f"Completed {number:,}/{len(tasks):,}", flush=True)
    else:
        try:
            executor = ProcessPoolExecutor(max_workers=int(args.workers))
        except (PermissionError, OSError):
            print("Process workers unavailable; using thread workers", flush=True)
            executor = ThreadPoolExecutor(max_workers=int(args.workers))
        with executor:
            futures = [executor.submit(basin_summary, task) for task in tasks]
            for number, future in enumerate(as_completed(futures), 1):
                results.append(future.result())
                if number % 25 == 0 or number == len(futures):
                    print(f"Completed {number:,}/{len(futures):,}", flush=True)

    basins, ecoregions = assemble(inventory)
    receipt = {
        "stage": 34,
        "status": "complete",
        "eligible_basins": int(len(inventory)),
        "completed_basins": int(basins["GAGE_ID"].nunique()),
        "ecoregions": int(ecoregions["spatial_group"].nunique()),
        "leads_days": list(LEADS_120),
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "new_exclusions": int(sum(item["status"] == "excluded" for item in results)),
    }
    atomic_json(receipt, RECEIPT_FILE)
    print(json.dumps(receipt, indent=2), flush=True)


if __name__ == "__main__":
    main()
