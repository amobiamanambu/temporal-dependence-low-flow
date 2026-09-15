"""Future-unrestricted 90-day low-flow trajectories and hazard verification.

Forecasts are issued once per week from an observed dry initialization.  No
future observation is used by an eligible model.  The perfect-forcing analog is
retained only as an explicitly ineligible ceiling for the value of a weather
forecast archive.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from .common import (
    RESULTS_ROOT, atomic_csv, atomic_json, dry_spell_age, ensemble_crps,
    fit_logistic, initialization_features, load_benchmark_config, load_daily,
    water_year,
)


LEADS = (1, 3, 7, 14, 30, 45, 60, 90)
MEMBERS = 101
ISSUE_WEEKDAY = 2  # Wednesday, matching the weekly extended GEFS schedule.

HYDROGRAPH_FEATURES = [
    "log_q", "dlog_q_1", "dlog_q_3", "dlog_q_7", "dlog_q_14",
    "log_q_minus_mean_7", "log_q_volatility_7", "log_q_volatility_14",
    "dry_spell_age", "doy_sin", "doy_cos",
]
HYDROCLIMATE_FEATURES = HYDROGRAPH_FEATURES + [
    "pr_sum_7", "pr_sum_30", "pr_sum_90", "pet_sum_7", "pet_sum_30",
    "pet_sum_90", "tmean_7", "tmean_30", "antecedent_aridity_30",
]
FLOW_SEASON_FEATURES = ["log_q", "dry_spell_age", "doy_sin", "doy_cos"]
SEASON_FEATURES = ["doy_sin", "doy_cos"]


def _future_matrix(values: pd.Series, horizon: int) -> np.ndarray:
    return np.column_stack([
        values.shift(-lead).to_numpy(float) for lead in range(1, horizon + 1)
    ])


def extended_cases(frame: pd.DataFrame, scale: float, config: dict) -> tuple[pd.DataFrame, list[str]]:
    """Build weekly dry-initialization cases without screening future weather."""
    horizon = max(LEADS)
    dates = frame["date"]
    q = frame["q_mm_day"] / scale
    q_path = _future_matrix(q, horizon)
    pr_path = _future_matrix(frame["pr_mm"], horizon)
    t_path = _future_matrix(frame["tmean_c"], horizon)
    snow_path = np.column_stack([
        frame["snowmelt_risk"].shift(-lead).astype("boolean").fillna(True).to_numpy(bool)
        for lead in range(1, horizon + 1)
    ])
    consecutive = (dates.shift(-horizon) - dates).dt.days.eq(horizon).to_numpy(bool)
    valid_path = np.isfinite(q_path).all(axis=1) & (q_path >= 0).all(axis=1)
    current_dry = (
        frame["pr_mm"].le(config["precipitation_threshold_mm_day"])
        & frame["snowmelt_risk"].eq(False)
    )
    age = dry_spell_age(current_dry)
    selected = (
        consecutive & valid_path & q.gt(0).to_numpy(bool)
        & current_dry.fillna(False).to_numpy(bool)
        & (age >= int(config["antecedent_days"]))
        & dates.dt.dayofweek.eq(ISSUE_WEEKDAY).to_numpy(bool)
    )
    features, feature_columns = initialization_features(frame, scale, config)
    rows = np.flatnonzero(selected)
    output = features.iloc[rows].copy().reset_index(drop=True)
    additions: dict[str, np.ndarray] = {
        "future_date": dates.shift(-horizon).iloc[rows].to_numpy(),
        "q": q.iloc[rows].to_numpy(float),
    }
    additions.update({
        f"q_h{day}": q_path[rows, day - 1]
        for day in range(1, horizon + 1)
    })
    oracle_columns: list[str] = []
    for lead in (7, 14, 30, 45, 60, 90):
        for prefix, values in (
            ("future_pr_sum", np.nansum(pr_path[rows, :lead], axis=1)),
            ("future_tmean", np.nanmean(t_path[rows, :lead], axis=1)),
            ("future_snow_days", np.sum(snow_path[rows, :lead], axis=1)),
        ):
            name = f"{prefix}_{lead}"
            additions[name] = values
            oracle_columns.append(name)
    output = pd.concat([output, pd.DataFrame(additions)], axis=1)
    return output, feature_columns + oracle_columns


def _fit_transform(train: pd.DataFrame, test: pd.DataFrame, features: list[str]):
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    x_train = scaler.fit_transform(imputer.fit_transform(train[features]))
    x_test = scaler.transform(imputer.transform(test[features]))
    return x_train, x_test


def analog_paths(train: pd.DataFrame, test: pd.DataFrame, features: list[str],
                 members: int = MEMBERS, scale_by_initial_q: bool = True) -> np.ndarray:
    x_train, x_test = _fit_transform(train, test, features)
    neighbors = min(members, len(train))
    search = NearestNeighbors(n_neighbors=neighbors)
    search.fit(x_train)
    indices = search.kneighbors(x_test, return_distance=False)
    path_columns = [f"q_h{day}" for day in range(1, max(LEADS) + 1)]
    training_paths = train[path_columns].to_numpy(float)
    if scale_by_initial_q:
        training_paths = training_paths / train["q"].to_numpy(float)[:, None]
        paths = test["q"].to_numpy(float)[:, None, None] * training_paths[indices]
    else:
        paths = training_paths[indices]
    if neighbors < members:
        paths = np.tile(paths, (1, int(np.ceil(members / neighbors)), 1))[:, :members]
    return np.maximum(paths, 0)


def ridge_residual_paths(train: pd.DataFrame, test: pd.DataFrame, features: list[str],
                         members: int = MEMBERS) -> np.ndarray:
    x_train, x_test = _fit_transform(train, test, features)
    path_columns = [f"q_h{day}" for day in range(1, max(LEADS) + 1)]
    floor = 1e-6
    y_train = np.log(
        np.maximum(train[path_columns].to_numpy(float), floor)
        / train["q"].to_numpy(float)[:, None]
    )
    regression = Ridge(alpha=10.0)
    regression.fit(x_train, y_train)
    fitted_train = regression.predict(x_train)
    fitted_test = regression.predict(x_test)
    residual = y_train - fitted_train
    neighbors = min(members, len(train))
    search = NearestNeighbors(n_neighbors=neighbors)
    search.fit(x_train)
    indices = search.kneighbors(x_test, return_distance=False)
    log_ratio = fitted_test[:, None, :] + residual[indices]
    if neighbors < members:
        log_ratio = np.tile(log_ratio, (1, int(np.ceil(members / neighbors)), 1))[:, :members]
    paths = test["q"].to_numpy(float)[:, None, None] * np.exp(
        np.clip(log_ratio, -16, 16)
    )
    return np.maximum(paths, 0)


def extra_trees_paths(train: pd.DataFrame, test: pd.DataFrame, features: list[str],
                      seed: int, members: int = MEMBERS) -> np.ndarray:
    x_train, x_test = _fit_transform(train, test, features)
    path_columns = [f"q_h{day}" for day in range(1, max(LEADS) + 1)]
    y_train = np.log(
        np.maximum(train[path_columns].to_numpy(float), 1e-6)
        / train["q"].to_numpy(float)[:, None]
    )
    forest = ExtraTreesRegressor(
        n_estimators=members, min_samples_leaf=5, max_features=0.8,
        n_jobs=1, random_state=seed,
    )
    forest.fit(x_train, y_train)
    paths = np.empty((len(test), members, max(LEADS)), dtype=np.float32)
    q0 = test["q"].to_numpy(float)
    for member, tree in enumerate(forest.estimators_):
        ratio = np.exp(np.clip(tree.predict(x_test), -16, 16))
        paths[:, member, :] = q0[:, None] * ratio
    return np.maximum(paths, 0)


def shuffle_members_by_day(paths: np.ndarray, seed: int) -> np.ndarray:
    """Preserve each daily ensemble exactly while destroying member trajectories."""
    rng = np.random.default_rng(seed)
    shuffled = np.empty_like(paths)
    for case in range(paths.shape[0]):
        for day in range(paths.shape[2]):
            shuffled[case, :, day] = paths[case, rng.permutation(paths.shape[1]), day]
    return shuffled


def first_event_time(paths: np.ndarray, threshold: float, target: str) -> np.ndarray:
    condition = paths <= threshold if target == "onset" else paths > threshold
    first = np.argmax(condition, axis=-1) + 1
    return np.where(condition.any(axis=-1), first, paths.shape[-1] + 1)


def discrete_crps(member_time: np.ndarray, observed_time: np.ndarray, horizon: int) -> np.ndarray:
    days = np.arange(1, horizon + 1)
    forecast = np.mean(member_time[:, :, None] <= days[None, None, :], axis=1)
    observed = observed_time[:, None] <= days[None, :]
    return np.mean((forecast - observed) ** 2, axis=1)


def _score_paths(gage: str, spatial_group: str, model: str, paths: np.ndarray,
                 test: pd.DataFrame, thresholds: dict[float, float], config: dict,
                 future_forcing_used: bool) -> list[dict]:
    truth_all = test[[f"q_h{day}" for day in range(1, max(LEADS) + 1)]].to_numpy(float)
    q0 = test["q"].to_numpy(float)
    years = water_year(test["date"])
    rows: list[dict] = []
    for quantile, threshold in thresholds.items():
        for target in ("onset", "recovery"):
            eligible = q0 > threshold if target == "onset" else q0 <= threshold
            if eligible.sum() < config["minimum_trajectory_evaluation_cases"]:
                continue
            for lead in LEADS:
                truth = truth_all[eligible, :lead]
                forecast = paths[eligible, :, :lead]
                observed_endpoint = truth[:, -1] <= threshold
                endpoint_probability = np.mean(forecast[:, :, -1] <= threshold, axis=1)
                observed_time = first_event_time(truth, threshold, target)
                member_time = first_event_time(forecast, threshold, target)
                observed_event = observed_time <= lead
                if min(observed_event.sum(), (~observed_event).sum()) < config["minimum_events"]:
                    continue
                event_probability = np.mean(member_time <= lead, axis=1)
                observed_deficit = np.sum(np.maximum(threshold - truth, 0), axis=1)
                member_deficit = np.sum(np.maximum(threshold - forecast, 0), axis=2)
                observed_duration = np.sum(truth <= threshold, axis=1).astype(float)
                member_duration = np.sum(forecast <= threshold, axis=2).astype(float)
                observed_minimum = np.min(truth, axis=1)
                member_minimum = np.min(forecast, axis=2)
                rows.append({
                    "GAGE_ID": str(gage).zfill(8), "spatial_group": spatial_group,
                    "threshold_quantile": float(quantile),
                    "threshold_name": f"Q{int(round(100 * quantile))}",
                    "lead_days": int(lead), "target": target, "model": model,
                    "n": int(eligible.sum()), "events": int(observed_event.sum()),
                    "endpoint_brier": float(np.mean((endpoint_probability - observed_endpoint) ** 2)),
                    "event_brier": float(np.mean((event_probability - observed_event) ** 2)),
                    "timing_crps": float(np.mean(discrete_crps(member_time, observed_time, lead))),
                    "deficit_crps": float(np.mean(ensemble_crps(member_deficit, observed_deficit))),
                    "duration_crps": float(np.mean(ensemble_crps(member_duration, observed_duration))),
                    "minimum_flow_crps": float(np.mean(ensemble_crps(member_minimum, observed_minimum))),
                    "test_water_years": int(len(np.unique(years[eligible]))),
                    "future_forcing_used": bool(future_forcing_used),
                })
    return rows


def _direct_score_rows(gage: str, spatial_group: str, train: pd.DataFrame,
                       test: pd.DataFrame, thresholds: dict[float, float], config: dict) -> list[dict]:
    rows: list[dict] = []
    features = HYDROCLIMATE_FEATURES
    q_train = train["q"].to_numpy(float)
    q_test = test["q"].to_numpy(float)
    for quantile, threshold in thresholds.items():
        for target in ("onset", "recovery"):
            train_selected = q_train > threshold if target == "onset" else q_train <= threshold
            test_selected = q_test > threshold if target == "onset" else q_test <= threshold
            if train_selected.sum() < config["minimum_trajectory_fit_cases"]:
                continue
            if test_selected.sum() < config["minimum_trajectory_evaluation_cases"]:
                continue
            fit = train.loc[train_selected]
            evaluate = test.loc[test_selected]
            for lead in LEADS:
                fit_path = fit[[f"q_h{day}" for day in range(1, lead + 1)]].to_numpy(float)
                test_path = evaluate[[f"q_h{day}" for day in range(1, lead + 1)]].to_numpy(float)
                fit_endpoint = fit_path[:, -1] <= threshold
                test_endpoint = test_path[:, -1] <= threshold
                fit_event = (fit_path <= threshold).any(axis=1) if target == "onset" else (fit_path > threshold).any(axis=1)
                test_event = (test_path <= threshold).any(axis=1) if target == "onset" else (test_path > threshold).any(axis=1)
                if min(fit_endpoint.sum(), (~fit_endpoint).sum()) >= config["minimum_events"]:
                    model = fit_logistic(fit, fit_endpoint, features)
                    probability = model.predict_proba(evaluate[features])[:, 1]
                    rows.append({
                        "GAGE_ID": str(gage).zfill(8), "spatial_group": spatial_group,
                        "threshold_quantile": float(quantile),
                        "threshold_name": f"Q{int(round(100 * quantile))}",
                        "lead_days": int(lead), "target": target,
                        "model": "direct_endpoint_logistic", "n": int(len(evaluate)),
                        "events": int(test_endpoint.sum()),
                        "endpoint_brier": float(np.mean((probability - test_endpoint) ** 2)),
                        "event_brier": np.nan, "timing_crps": np.nan,
                        "deficit_crps": np.nan, "duration_crps": np.nan,
                        "minimum_flow_crps": np.nan,
                        "test_water_years": int(len(np.unique(water_year(evaluate["date"])))),
                        "future_forcing_used": False,
                    })
                if (min(fit_event.sum(), (~fit_event).sum()) >= config["minimum_events"]
                        and min(test_event.sum(), (~test_event).sum()) >= config["minimum_events"]):
                    model = fit_logistic(fit, fit_event, features)
                    probability = model.predict_proba(evaluate[features])[:, 1]
                    rows.append({
                        "GAGE_ID": str(gage).zfill(8), "spatial_group": spatial_group,
                        "threshold_quantile": float(quantile),
                        "threshold_name": f"Q{int(round(100 * quantile))}",
                        "lead_days": int(lead), "target": target,
                        "model": "direct_event_logistic", "n": int(len(evaluate)),
                        "events": int(test_event.sum()), "endpoint_brier": np.nan,
                        "event_brier": float(np.mean((probability - test_event) ** 2)),
                        "timing_crps": np.nan, "deficit_crps": np.nan,
                        "duration_crps": np.nan, "minimum_flow_crps": np.nan,
                        "test_water_years": int(len(np.unique(water_year(evaluate["date"])))),
                        "future_forcing_used": False,
                    })
    return rows


def extended_worker(task):
    gage, source, spatial_group, config = task
    try:
        frame = load_daily(source)
        qfit = frame.loc[frame["date"].le(pd.Timestamp(config["fit_end"])), "q_mm_day"]
        qfit = qfit[qfit.gt(0)]
        if qfit.empty:
            raise ValueError("no_positive_training_scale")
        scale = float(qfit.median())
        history = frame.loc[
            frame["date"].le(pd.Timestamp(config["threshold_reference_end"])), "q_mm_day"
        ] / scale
        thresholds = {
            float(q): float(history.quantile(float(q))) for q in config["low_flow_quantiles"]
        }
        thresholds = {q: value for q, value in thresholds.items() if np.isfinite(value) and value > 0}
        cases, all_features = extended_cases(frame, scale, config)
        train = cases[cases["future_date"].le(pd.Timestamp(config["calibration_end"]))].copy()
        test = cases[
            cases["date"].ge(pd.Timestamp(config["evaluation_start"]))
            & cases["future_date"].le(pd.Timestamp(config["evaluation_end"]))
        ].copy()
        if len(train) < config["minimum_trajectory_fit_cases"]:
            raise ValueError("insufficient_training_initializations")
        if len(test) < config["minimum_trajectory_evaluation_cases"]:
            raise ValueError("insufficient_test_initializations")
        seed = int(config["random_seed"] + int(str(gage)[-5:]) + 17000)
        rows: list[dict] = []
        persistence = np.repeat(
            test["q"].to_numpy(float)[:, None, None], MEMBERS, axis=1
        ).repeat(max(LEADS), axis=2)
        rows.extend(_score_paths(
            gage, spatial_group, "constant_persistence_path", persistence,
            test, thresholds, config, False,
        ))
        del persistence

        model_specs = [
            ("seasonal_climatology_path", SEASON_FEATURES, False, False),
            ("flow_season_analog", FLOW_SEASON_FEATURES, True, True),
            ("hydrograph_analog", HYDROGRAPH_FEATURES, True, True),
            ("hydroclimate_analog", HYDROCLIMATE_FEATURES, True, True),
            ("perfect_forcing_oracle", all_features, True, False),
        ]
        for number, (name, features, scale_by_q, make_shuffle) in enumerate(model_specs):
            paths = analog_paths(train, test, features, MEMBERS, scale_by_q)
            rows.extend(_score_paths(
                gage, spatial_group, name, paths, test, thresholds, config,
                name == "perfect_forcing_oracle",
            ))
            if make_shuffle:
                shuffled = shuffle_members_by_day(paths, seed + number)
                rows.extend(_score_paths(
                    gage, spatial_group, name.replace("_analog", "_marginal_shuffle"), shuffled,
                    test, thresholds, config, False,
                ))
                del shuffled
            del paths

        ridge = ridge_residual_paths(train, test, HYDROCLIMATE_FEATURES, MEMBERS)
        rows.extend(_score_paths(
            gage, spatial_group, "ridge_residual_block", ridge,
            test, thresholds, config, False,
        ))
        ridge_shuffle = shuffle_members_by_day(ridge, seed + 101)
        rows.extend(_score_paths(
            gage, spatial_group, "ridge_residual_shuffle", ridge_shuffle,
            test, thresholds, config, False,
        ))
        del ridge, ridge_shuffle

        trees = extra_trees_paths(train, test, HYDROCLIMATE_FEATURES, seed, MEMBERS)
        rows.extend(_score_paths(
            gage, spatial_group, "extra_trees_path", trees,
            test, thresholds, config, False,
        ))
        tree_shuffle = shuffle_members_by_day(trees, seed + 202)
        rows.extend(_score_paths(
            gage, spatial_group, "extra_trees_marginal_shuffle", tree_shuffle,
            test, thresholds, config, False,
        ))
        del trees, tree_shuffle
        rows.extend(_direct_score_rows(gage, spatial_group, train, test, thresholds, config))
        return pd.DataFrame(rows), []
    except Exception as error:
        return pd.DataFrame(), [{"GAGE_ID": str(gage).zfill(8), "reason": repr(error)}]


def run_extended(panel: pd.DataFrame, workers: int, output_subdir: str,
                 max_basins: int | None = None) -> dict:
    config = load_benchmark_config()
    output = RESULTS_ROOT / output_subdir
    selected = panel.head(int(max_basins)).copy() if max_basins else panel.copy()
    tasks = [
        (row.GAGE_ID, row.file, row.spatial_group, config)
        for row in selected.itertuples(index=False)
    ]
    frames: list[pd.DataFrame] = []
    failures: list[dict] = []
    if workers == 1:
        iterator = ((task[0], extended_worker(task)) for task in tasks)
        for number, (gage, result) in enumerate(iterator, start=1):
            frame, failed = result
            if not frame.empty:
                frames.append(frame)
            failures.extend(failed)
            print(f"Extended trajectories {number}/{len(tasks)} basins", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(extended_worker, task): task[0] for task in tasks}
            for number, future in enumerate(as_completed(futures), start=1):
                gage = futures[future]
                try:
                    frame, failed = future.result()
                    if not frame.empty:
                        frames.append(frame)
                    failures.extend(failed)
                except Exception as error:
                    failures.append({"GAGE_ID": str(gage).zfill(8), "reason": repr(error)})
                if number % 4 == 0 or number == len(tasks):
                    print(f"Extended trajectories {number}/{len(tasks)} basins", flush=True)
    metrics = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    atomic_csv(metrics, output / "extended_trajectory_metrics.csv.gz", compression="gzip")
    atomic_csv(pd.DataFrame(failures), output / "extended_trajectory_failures.csv")
    receipt = {
        "status": "complete", "basins_requested": int(len(selected)),
        "basins_with_results": int(metrics["GAGE_ID"].nunique() if not metrics.empty else 0),
        "forecast_initialization": "Wednesday observed dry day with dry age >= 3",
        "future_forcing_selection": False,
        "leads": list(LEADS), "ensemble_members": MEMBERS,
        "perfect_forcing_oracle_eligible": False,
    }
    atomic_json(receipt, output / "_SUCCESS.json")
    return receipt
