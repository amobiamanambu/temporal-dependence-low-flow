"""Regional standardized-innovation library and forecast evaluation."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

from .common import (
    RESULTS_ROOT, atomic_csv, atomic_json, conditional_cases, eligible_inventory,
    ensemble_probability, fit_location_scale, load_benchmark_config, load_daily,
    location_scale, split_cases,
)
from .endpoint import _score_rows


_POOL_CACHE: dict | None = None


def load_pool_cache(library_path: str) -> dict:
    global _POOL_CACHE
    if _POOL_CACHE is not None:
        return _POOL_CACHE
    library = pd.read_csv(library_path)
    pools = {}
    for key, group in library.groupby(["spatial_group", "lead_days", "flow_group"]):
        pools[(str(key[0]), int(key[1]), int(key[2]))] = group["standardized_innovation"].to_numpy(float)
    for key, group in library.groupby(["spatial_group", "lead_days"]):
        pools[(str(key[0]), int(key[1]), -1)] = group["standardized_innovation"].to_numpy(float)
    _POOL_CACHE = pools
    return pools


def residual_worker(task):
    gage, source, spatial_group, leads, config = task
    rows, failures = [], []
    frame = load_daily(source)
    scale_source = frame.loc[frame["date"].le(pd.Timestamp(config["fit_end"])), "q_mm_day"]
    scale_source = scale_source[scale_source.gt(0)]
    if scale_source.empty:
        return pd.DataFrame(), [{"GAGE_ID": gage, "reason": "no_fit_scale"}]
    scale = float(scale_source.median())
    for lead in leads:
        try:
            cases = conditional_cases(frame, lead, scale, config)
            fit = cases[
                cases["future_date"].le(pd.Timestamp(config["calibration_end"]))
            ].copy()
            if len(fit) < int(config["minimum_fit_cases"]):
                continue
            model = fit_location_scale(fit)
            q_model, mean, sigma = location_scale(fit["q"].to_numpy(float), model)
            residual = (fit["dq"].to_numpy(float) - mean) / sigma
            group = np.searchsorted(model["residual_edges"][1:-1], q_model, side="right")
            rng = np.random.default_rng(config["random_seed"] + int(str(gage)[-5:]) + lead)
            maximum = int(config["regional_residuals_per_basin_lead"])
            selected = []
            per_group = max(1, maximum // max(1, len(np.unique(group))))
            for value in np.unique(group):
                indices = np.flatnonzero((group == value) & np.isfinite(residual))
                if len(indices) > per_group:
                    indices = rng.choice(indices, per_group, replace=False)
                selected.extend(indices.tolist())
            for index in selected[:maximum]:
                rows.append({
                    "GAGE_ID": str(gage).zfill(8), "spatial_group": spatial_group,
                    "lead_days": int(lead), "flow_group": int(group[index]),
                    "standardized_innovation": float(residual[index]),
                })
        except Exception as error:
            failures.append({"GAGE_ID": gage, "lead_days": lead, "reason": repr(error)})
    return pd.DataFrame(rows), failures


def build_library(panel: pd.DataFrame, leads: list[int], workers: int, force: bool,
                  max_training_basins: int | None = None) -> pd.DataFrame:
    config = load_benchmark_config()
    output = RESULTS_ROOT / "03_regional"
    library_path = output / "regional_innovation_library.csv.gz"
    if library_path.exists() and not force:
        receipt_path = output / "regional_library_SUCCESS.json"
        receipt = {}
        if receipt_path.exists():
            import json
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        expected = (
            min(int(max_training_basins), len(eligible_inventory()) - len(panel))
            if max_training_basins else len(eligible_inventory()) - len(panel)
        )
        if (
            int(receipt.get("training_basins", -1)) == expected
            and receipt.get("training_end") == config["calibration_end"]
        ):
            return pd.read_csv(library_path, dtype={"GAGE_ID": str})
    inventory = eligible_inventory()
    excluded = set(panel["GAGE_ID"].astype(str).str.zfill(8))
    training = inventory[~inventory["GAGE_ID"].isin(excluded)].copy()
    if max_training_basins:
        rng = np.random.default_rng(config["random_seed"] + 3010)
        training["random_order"] = rng.random(len(training))
        queues = {
            name: group.sort_values("random_order").index.tolist()
            for name, group in training.groupby("spatial_group", dropna=False)
        }
        selected = []
        while len(selected) < min(int(max_training_basins), len(training)):
            changed = False
            for name in sorted(queues):
                if queues[name] and len(selected) < int(max_training_basins):
                    selected.append(queues[name].pop(0))
                    changed = True
            if not changed:
                break
        training = training.loc[selected].copy()
    tasks = [(r.GAGE_ID, r.file, r.spatial_group, leads, config)
             for r in training.itertuples(index=False)]
    frames, failures = [], []
    if workers == 1:
        iterator = ((task[0], residual_worker(task)) for task in tasks)
        for number, (gage, result) in enumerate(iterator, start=1):
            frame, failed = result
            if not frame.empty:
                frames.append(frame)
            failures.extend(failed)
            if number % 100 == 0 or number == len(tasks):
                print(f"Regional library {number}/{len(tasks)} basins", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(residual_worker, task): task[0] for task in tasks}
            for number, future in enumerate(as_completed(futures), start=1):
                gage = futures[future]
                try:
                    frame, failed = future.result()
                    if not frame.empty:
                        frames.append(frame)
                    failures.extend(failed)
                except Exception as error:
                    failures.append({"GAGE_ID": gage, "reason": repr(error)})
                if number % 100 == 0 or number == len(tasks):
                    print(f"Regional library {number}/{len(tasks)} basins", flush=True)
    library = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if library.empty:
        raise RuntimeError("No regional innovations were produced")
    cap = int(config["regional_pool_cap"])
    rng = np.random.default_rng(config["random_seed"] + 3000)
    retained = []
    for _, group in library.groupby(["spatial_group", "lead_days", "flow_group"]):
        if len(group) > cap:
            group = group.iloc[rng.choice(len(group), cap, replace=False)]
        retained.append(group)
    library = pd.concat(retained, ignore_index=True)
    atomic_csv(library, library_path, compression="gzip")
    atomic_csv(pd.DataFrame(failures), output / "regional_library_failures.csv")
    atomic_json({
        "status": "complete", "training_basins": int(training.shape[0]),
        "screening_basins_excluded": int(len(excluded)), "rows": int(len(library)),
        "training_end": config["calibration_end"], "uses_evaluation_period": False,
    }, output / "regional_library_SUCCESS.json")
    return library


def regional_members(q: np.ndarray, model: dict, spatial_group: str, lead: int,
                     pools: dict, members: int, shrinkage: float, seed: int) -> np.ndarray:
    q_model, mean, sigma = location_scale(q, model)
    group = np.searchsorted(model["residual_edges"][1:-1], q_model, side="right")
    probabilities = (np.arange(members) + 0.5) / members
    result = np.empty((len(q), members), float)
    rng = np.random.default_rng(seed)
    global_pool = pools.get((spatial_group, lead, -1), model["residual"])
    for value in np.unique(group):
        selected = group == value
        local = model["residual_groups"].get(int(value), model["residual"])
        regional = pools.get((spatial_group, lead, int(value)), global_pool)
        weight = len(local) / (len(local) + shrinkage)
        draws = max(10000, members * 50)
        choose_local = rng.random(draws) < weight
        mixture = np.empty(draws)
        mixture[choose_local] = rng.choice(local, choose_local.sum(), replace=True)
        mixture[~choose_local] = rng.choice(regional, (~choose_local).sum(), replace=True)
        innovation = np.quantile(mixture, probabilities)
        result[selected] = q[selected, None] + mean[selected, None] + sigma[selected, None] * innovation
    return np.maximum(result, 0)


def regional_test_worker(task):
    gage, source, spatial_group, leads, library_path, config = task
    pools = load_pool_cache(library_path)
    rows, yearly_rows, failures = [], [], []
    frame = load_daily(source)
    fit_q = frame.loc[frame["date"].le(pd.Timestamp(config["fit_end"])), "q_mm_day"]
    fit_q = fit_q[fit_q.gt(0)]
    if fit_q.empty:
        return pd.DataFrame(), pd.DataFrame(), [{"GAGE_ID": gage, "reason": "no_scale"}]
    scale = float(fit_q.median())
    history = frame.loc[frame["date"].le(pd.Timestamp(config["threshold_reference_end"])), "q_mm_day"] / scale
    thresholds = {q: float(history.quantile(q)) for q in config["low_flow_quantiles"]}
    for lead in leads:
        try:
            cases = conditional_cases(frame, lead, scale, config)
            _, _, test = split_cases(cases, config)
            fit = cases[
                cases["future_date"].le(pd.Timestamp(config["calibration_end"]))
            ].copy()
            if len(fit) < config["minimum_fit_cases"] or len(test) < config["minimum_evaluation_cases"]:
                continue
            model = fit_location_scale(fit)
            ensemble = regional_members(
                test["q"].to_numpy(float), model, spatial_group, lead, pools,
                int(config["ensemble_members"]),
                float(config["regional_shrinkage_equivalent_count"]),
                int(config["random_seed"] + int(str(gage)[-5:]) + lead),
            )
            for quantile, threshold in thresholds.items():
                target = test["q_next"].to_numpy(float) <= threshold
                if min(target.sum(), (~target).sum()) < int(config["minimum_events"]):
                    continue
                probability = ensemble_probability(ensemble, threshold)
                row, yearly = _score_rows(
                    gage, spatial_group, lead, quantile, threshold,
                    "regional_innovation_shrinkage", probability, target, test,
                    ensemble, {}, config,
                )
                rows.append(row)
                yearly_rows.extend(yearly)
        except Exception as error:
            failures.append({"GAGE_ID": gage, "lead_days": lead, "reason": repr(error)})
    return pd.DataFrame(rows), pd.DataFrame(yearly_rows), failures


def evaluate_regional(panel: pd.DataFrame, library: pd.DataFrame, leads: list[int],
                      workers: int, output_subdir: str = "03_regional") -> dict:
    config = load_benchmark_config()
    output = RESULTS_ROOT / output_subdir
    library_path = str(RESULTS_ROOT / "03_regional" / "regional_innovation_library.csv.gz")
    tasks = [(r.GAGE_ID, r.file, r.spatial_group, leads, library_path, config)
             for r in panel.itertuples(index=False)]
    frames, yearly, failures = [], [], []
    if workers == 1:
        iterator = ((task[0], regional_test_worker(task)) for task in tasks)
        for number, (gage, result) in enumerate(iterator, start=1):
            frame, annual, failed = result
            if not frame.empty:
                frames.append(frame)
            if not annual.empty:
                yearly.append(annual)
            failures.extend(failed)
            if number % 8 == 0 or number == len(tasks):
                print(f"Regional evaluation {number}/{len(tasks)} basins", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(regional_test_worker, task): task[0] for task in tasks}
            for number, future in enumerate(as_completed(futures), start=1):
                gage = futures[future]
                try:
                    frame, annual, failed = future.result()
                    if not frame.empty:
                        frames.append(frame)
                    if not annual.empty:
                        yearly.append(annual)
                    failures.extend(failed)
                except Exception as error:
                    failures.append({"GAGE_ID": gage, "reason": repr(error)})
                if number % 8 == 0 or number == len(tasks):
                    print(f"Regional evaluation {number}/{len(tasks)} basins", flush=True)
    metrics = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    annual = pd.concat(yearly, ignore_index=True) if yearly else pd.DataFrame()
    atomic_csv(metrics, output / "regional_basin_metrics.csv.gz", compression="gzip")
    atomic_csv(annual, output / "regional_water_year_scores.csv.gz", compression="gzip")
    atomic_csv(pd.DataFrame(failures), output / "regional_evaluation_failures.csv")
    receipt = {"status": "complete", "basins": int(
        metrics["GAGE_ID"].nunique() if "GAGE_ID" in metrics else 0
    ),
               "method": "regional_innovation_shrinkage"}
    atomic_json(receipt, output / "_SUCCESS.json")
    return receipt
