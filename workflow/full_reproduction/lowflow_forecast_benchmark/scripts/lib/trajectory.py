"""Coherent dry-spell trajectories, first passage, and deficit-volume verification."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from .common import (
    RESULTS_ROOT, atomic_csv, atomic_json, ensemble_crps, ensemble_probability,
    future_dry_mask, initialization_features, load_benchmark_config, load_daily,
    water_year,
)


def trajectory_cases(frame: pd.DataFrame, horizon: int, scale: float, config: dict) -> pd.DataFrame:
    dates = frame["date"]
    q = frame["q_mm_day"] / scale
    columns = {f"q_h{lead}": q.shift(-lead) for lead in range(1, horizon + 1)}
    path = pd.DataFrame(columns)
    valid_path = path.notna().all(axis=1) & path.ge(0).all(axis=1)
    consecutive = (dates.shift(-horizon) - dates).dt.days.eq(horizon)
    selected = q.gt(0) & valid_path & consecutive & future_dry_mask(frame, horizon, config)
    features, feature_columns = initialization_features(frame, scale, config)
    output = features.loc[selected].copy()
    output["future_date"] = dates.shift(-horizon).loc[selected].to_numpy()
    output["q"] = q.loc[selected].to_numpy(float)
    for name in path:
        output[name] = path.loc[selected, name].to_numpy(float)
    return output[["date", "future_date", "q", *feature_columns, *path.columns]]


def first_passage_time(paths: np.ndarray, threshold: float) -> np.ndarray:
    crossed = paths <= threshold
    first = np.argmax(crossed, axis=-1) + 1
    return np.where(crossed.any(axis=-1), first, paths.shape[-1] + 1)


def discrete_time_crps(member_time: np.ndarray, observed_time: np.ndarray, horizon: int) -> np.ndarray:
    days = np.arange(1, horizon + 1)
    forecast_cdf = np.mean(member_time[:, :, None] <= days[None, None, :], axis=1)
    observed_cdf = observed_time[:, None] <= days[None, :]
    return np.mean((forecast_cdf - observed_cdf) ** 2, axis=1)


def independent_paths(train: pd.DataFrame, test_q: np.ndarray, horizon: int,
                      members: int, seed: int) -> np.ndarray:
    q0 = train["q"].to_numpy(float)
    q1 = train["q_h1"].to_numpy(float)
    log_return = np.log(np.maximum(q1, 1e-10) / np.maximum(q0, 1e-10))
    edges = np.unique(np.quantile(q0, np.linspace(0, 1, 6)))
    assignment = np.searchsorted(edges[1:-1], q0, side="right")
    pools = {group: log_return[assignment == group] for group in range(len(edges) - 1)}
    rng = np.random.default_rng(seed)
    paths = np.empty((len(test_q), members, horizon), float)
    current = np.repeat(test_q[:, None], members, axis=1)
    for day in range(horizon):
        state = np.searchsorted(edges[1:-1], current, side="right")
        updated = np.empty_like(current)
        for group in np.unique(state):
            selected = state == group
            pool = pools.get(int(group), log_return)
            pool = pool[np.isfinite(pool)]
            draws = rng.choice(pool, selected.sum(), replace=True)
            updated[selected] = current[selected] * np.exp(draws)
        current = np.maximum(updated, 0)
        paths[:, :, day] = current
    return paths


def coherent_analog_paths(train: pd.DataFrame, test: pd.DataFrame, horizon: int,
                          members: int) -> np.ndarray:
    features = ["log_q", "dlog_q_7", "dry_spell_age", "doy_sin", "doy_cos"]
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    x_train = scaler.fit_transform(imputer.fit_transform(train[features]))
    x_test = scaler.transform(imputer.transform(test[features]))
    neighbors = min(members, len(train))
    search = NearestNeighbors(n_neighbors=neighbors)
    search.fit(x_train)
    indices = search.kneighbors(x_test, return_distance=False)
    path_columns = [f"q_h{lead}" for lead in range(1, horizon + 1)]
    ratios = train[path_columns].to_numpy(float) / train["q"].to_numpy(float)[:, None]
    paths = test["q"].to_numpy(float)[:, None, None] * ratios[indices]
    if neighbors < members:
        paths = np.tile(paths, (1, int(np.ceil(members / neighbors)), 1))[:, :members, :]
    return np.maximum(paths, 0)


def shuffle_path_dependence(paths: np.ndarray, seed: int) -> np.ndarray:
    """Destroy cross-day dependence while preserving every daily marginal exactly."""
    rng = np.random.default_rng(seed)
    shuffled = np.empty_like(paths)
    for case in range(paths.shape[0]):
        for day in range(paths.shape[2]):
            shuffled[case, :, day] = paths[case, rng.permutation(paths.shape[1]), day]
    return shuffled


def trajectory_worker(task):
    gage, source, spatial_group, config = task
    horizon = int(max(config["sentinel_leads"]))
    frame = load_daily(source)
    qfit = frame.loc[frame["date"].le(pd.Timestamp(config["fit_end"])), "q_mm_day"]
    qfit = qfit[qfit.gt(0)]
    if qfit.empty:
        return pd.DataFrame(), [{"GAGE_ID": gage, "reason": "no_scale"}]
    scale = float(qfit.median())
    history = frame.loc[frame["date"].le(pd.Timestamp(config["threshold_reference_end"])), "q_mm_day"] / scale
    cases = trajectory_cases(frame, horizon, scale, config)
    train = cases[cases["future_date"].le(pd.Timestamp(config["fit_end"]))].copy()
    test = cases[
        cases["date"].ge(pd.Timestamp(config["evaluation_start"]))
        & cases["future_date"].le(pd.Timestamp(config["evaluation_end"]))
    ].copy()
    if (len(train) < config["minimum_trajectory_fit_cases"] or
            len(test) < config["minimum_trajectory_evaluation_cases"]):
        return pd.DataFrame(), [{"GAGE_ID": gage, "reason": "insufficient_trajectory_cases"}]
    members = int(config["ensemble_members"])
    seed = int(config["random_seed"] + int(str(gage)[-5:]) + 9000)
    try:
        coherent = coherent_analog_paths(train, test, horizon, members)
        forecasts = {
            "constant_persistence_path": np.repeat(
                test["q"].to_numpy(float)[:, None, None], members, axis=1
            ).repeat(horizon, axis=2),
            "independent_innovation_path": independent_paths(
                train, test["q"].to_numpy(float), horizon, members, seed
            ),
            "coherent_block_analog": coherent,
            "analog_marginal_shuffle": shuffle_path_dependence(coherent, seed + 1),
        }
    except Exception as error:
        return pd.DataFrame(), [{"GAGE_ID": gage, "reason": repr(error)}]
    truth_all = test[[f"q_h{lead}" for lead in range(1, horizon + 1)]].to_numpy(float)
    years = water_year(test["date"])
    rows = []
    for quantile in config["low_flow_quantiles"]:
        threshold = float(history.quantile(float(quantile)))
        if not np.isfinite(threshold) or threshold <= 0:
            continue
        eligible = test["q"].to_numpy(float) > threshold
        if eligible.sum() < config["minimum_trajectory_evaluation_cases"]:
            continue
        for lead in [value for value in config["sentinel_leads"] if value >= 3]:
            lead = int(lead)
            truth = truth_all[eligible, :lead]
            observed_endpoint = truth[:, -1] <= threshold
            observed_time = first_passage_time(truth, threshold)
            observed_crossing = observed_time <= lead
            observed_deficit = np.sum(np.maximum(threshold - truth, 0), axis=1)
            if min(observed_crossing.sum(), (~observed_crossing).sum()) < config["minimum_events"]:
                continue
            for model, full_paths in forecasts.items():
                paths = full_paths[eligible, :, :lead]
                endpoint_probability = np.mean(paths[:, :, -1] <= threshold, axis=1)
                member_time = first_passage_time(paths, threshold)
                crossing_probability = np.mean(member_time <= lead, axis=1)
                member_deficit = np.sum(np.maximum(threshold - paths, 0), axis=2)
                rows.append({
                    "GAGE_ID": str(gage).zfill(8), "spatial_group": spatial_group,
                    "threshold_quantile": float(quantile),
                    "threshold_name": f"Q{int(round(100 * quantile))}",
                    "lead_days": lead, "model": model, "n": int(eligible.sum()),
                    "initialization_above_threshold": True,
                    "crossing_events": int(observed_crossing.sum()),
                    "endpoint_brier": float(np.mean((endpoint_probability - observed_endpoint) ** 2)),
                    "first_passage_brier": float(np.mean((crossing_probability - observed_crossing) ** 2)),
                    "first_passage_discrete_crps": float(np.mean(
                        discrete_time_crps(member_time, observed_time, lead)
                    )),
                    "deficit_volume_crps": float(np.mean(
                        ensemble_crps(member_deficit, observed_deficit)
                    )),
                    "test_water_years": int(len(np.unique(years))),
                })
    return pd.DataFrame(rows), []


def run_trajectory(panel: pd.DataFrame, workers: int,
                   output_subdir: str = "04_trajectories") -> dict:
    config = load_benchmark_config()
    output = RESULTS_ROOT / output_subdir
    tasks = [(r.GAGE_ID, r.file, r.spatial_group, config) for r in panel.itertuples(index=False)]
    frames, failures = [], []
    if workers == 1:
        iterator = ((task[0], trajectory_worker(task)) for task in tasks)
        for number, (gage, result) in enumerate(iterator, start=1):
            frame, failed = result
            if not frame.empty:
                frames.append(frame)
            failures.extend(failed)
            if number % 8 == 0 or number == len(tasks):
                print(f"Trajectory evaluation {number}/{len(tasks)} basins", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(trajectory_worker, task): task[0] for task in tasks}
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
                    print(f"Trajectory evaluation {number}/{len(tasks)} basins", flush=True)
    metrics = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    atomic_csv(metrics, output / "trajectory_basin_metrics.csv.gz", compression="gzip")
    atomic_csv(pd.DataFrame(failures), output / "trajectory_failures.csv")
    receipt = {"status": "complete", "basins": int(
        metrics["GAGE_ID"].nunique() if "GAGE_ID" in metrics else 0
    ),
               "target": "conditional_first_passage_and_deficit"}
    atomic_json(receipt, output / "_SUCCESS.json")
    return receipt
