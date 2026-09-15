#!/usr/bin/env python3
"""Test split-conformal correction of 90% heavy-tail prediction intervals."""

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
    RESULTS_ROOT, atomic_csv, atomic_json, conditional_cases, empirical_members,
    fit_location_scale, load_benchmark_config, load_daily, split_cases,
)


def worker(task):
    gage, source, spatial_group, leads, config = task
    frame = load_daily(source)
    scale_values = frame.loc[frame["date"].le(pd.Timestamp(config["fit_end"])), "q_mm_day"]
    scale_values = scale_values[scale_values.gt(0)]
    if scale_values.empty:
        return pd.DataFrame(), [{"GAGE_ID": gage, "reason": "no_scale"}]
    scale = float(scale_values.median())
    rows, failures = [], []
    for lead in leads:
        try:
            fit, calibration, test = split_cases(
                conditional_cases(frame, lead, scale, config), config
            )
            if (len(fit) < config["minimum_fit_cases"] or
                    len(calibration) < config["minimum_calibration_cases"] or
                    len(test) < config["minimum_evaluation_cases"]):
                continue
            model = fit_location_scale(fit)
            combined = pd.concat([calibration, test], ignore_index=True)
            ensemble = empirical_members(
                combined["q"].to_numpy(float), model, config["ensemble_members"]
            )
            cal_ensemble = ensemble[:len(calibration)]
            test_ensemble = ensemble[len(calibration):]
            cal_low, cal_high = np.quantile(cal_ensemble, [0.05, 0.95], axis=1)
            test_low, test_high = np.quantile(test_ensemble, [0.05, 0.95], axis=1)
            cal_y = calibration["q_next"].to_numpy(float)
            test_y = test["q_next"].to_numpy(float)
            nonconformity = np.maximum(cal_low - cal_y, cal_y - cal_high)
            rank = min(1.0, np.ceil((len(nonconformity) + 1) * 0.90) / len(nonconformity))
            correction = max(0.0, float(np.quantile(nonconformity, rank, method="higher")))
            conformal_low = np.maximum(0, test_low - correction)
            conformal_high = test_high + correction
            for method, low, high in [
                ("uncalibrated_empirical_interval", test_low, test_high),
                ("split_conformal_empirical_interval", conformal_low, conformal_high),
            ]:
                rows.append({
                    "GAGE_ID": str(gage).zfill(8), "spatial_group": spatial_group,
                    "lead_days": int(lead), "model": method, "n": int(len(test)),
                    "coverage_90": float(np.mean((test_y >= low) & (test_y <= high))),
                    "coverage_abs_error": float(abs(np.mean((test_y >= low) & (test_y <= high)) - 0.90)),
                    "mean_interval_width": float(np.mean(high - low)),
                    "conformal_additive_correction_qnorm": correction,
                })
        except Exception as error:
            failures.append({"GAGE_ID": gage, "lead_days": lead, "reason": repr(error)})
    return pd.DataFrame(rows), failures


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-basins", type=int)
    args = parser.parse_args()
    config = load_benchmark_config()
    panel = pd.read_csv(RESULTS_ROOT / "01_panel" / "candidate_screening_panel.csv",
                        dtype={"GAGE_ID": str})
    if args.max_basins:
        panel = panel.head(int(args.max_basins)).copy()
    tasks = [(r.GAGE_ID, r.file, r.spatial_group,
              [int(v) for v in config["sentinel_leads"]], config)
             for r in panel.itertuples(index=False)]
    frames, failures = [], []
    if args.workers == 1:
        iterator = ((task[0], worker(task)) for task in tasks)
        for number, (gage, result) in enumerate(iterator, start=1):
            frame, failed = result
            if not frame.empty:
                frames.append(frame)
            failures.extend(failed)
            if number % 8 == 0 or number == len(tasks):
                print(f"Conformal intervals {number}/{len(tasks)} basins", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(worker, task): task[0] for task in tasks}
            for number, future in enumerate(as_completed(futures), start=1):
                gage = futures[future]
                try:
                    frame, failed = future.result()
                    if not frame.empty:
                        frames.append(frame)
                    failures.extend(failed)
                except Exception as error:
                    failures.append({"GAGE_ID": gage, "reason": repr(error)})
                if number % 8 == 0 or number == len(tasks):
                    print(f"Conformal intervals {number}/{len(tasks)} basins", flush=True)
    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    output = RESULTS_ROOT / "02_endpoint_candidates"
    atomic_csv(combined, output / "conformal_interval_metrics.csv.gz", compression="gzip")
    atomic_csv(pd.DataFrame(failures), output / "conformal_interval_failures.csv")
    receipt = {
        "status": "complete", "basins": int(
            combined["GAGE_ID"].nunique() if "GAGE_ID" in combined else 0
        ),
        "claim_boundary": "marginal coverage calibration, not added discrimination",
    }
    atomic_json(receipt, output / "conformal_SUCCESS.json")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()

