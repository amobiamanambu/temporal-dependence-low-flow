"""Data, model, and verification utilities for the low-flow benchmark.

The code intentionally uses only information dated before each evaluation period.  Future
precipitation and snowmelt are used only by ``conditional_cases`` to define the manuscript's
conditional dry-spell target; they are never passed to a forecast model as predictors.
"""

from __future__ import annotations

import json
import math
import os
import tempfile
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy import optimize, stats
from scipy.special import expit, logsumexp
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


BENCHMARK_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = BENCHMARK_ROOT.parent
RESULTS_ROOT = BENCHMARK_ROOT / "results"
DAILY_ROOT = PROJECT_ROOT / "continental_run" / "10_transitions" / "daily"
INVENTORY_FILE = PROJECT_ROOT / "continental_run" / "10_transitions" / "transition_inventory.csv"
ATTRIBUTES_FILE = PROJECT_ROOT / "continental_run" / "06_basin_index" / "basin_index.csv"
STAGE18_METRICS = (
    PROJECT_ROOT / "continental_run" / "18_lowflow_predictability_horizons"
    / "basin_predictability_metrics.csv.gz"
)


def load_benchmark_config() -> dict:
    return json.loads((BENCHMARK_ROOT / "config.json").read_text(encoding="utf-8"))


def ensure_dirs() -> None:
    for name in ["00_audit", "01_panel", "02_endpoint_candidates", "03_regional",
                 "04_trajectories", "05_operational", "06_first_passage",
                 "07_comparison", "08_full_confirmation"]:
        (RESULTS_ROOT / name).mkdir(parents=True, exist_ok=True)


def atomic_csv(frame: pd.DataFrame, path: Path, **kwargs) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = ".csv.gz" if str(path).endswith(".gz") else ".csv"
    fd, temporary = tempfile.mkstemp(prefix=path.stem + ".", suffix=suffix, dir=path.parent)
    os.close(fd)
    temporary_path = Path(temporary)
    try:
        frame.to_csv(temporary_path, index=False, **kwargs)
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def atomic_json(value: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.stem + ".", suffix=".json", dir=path.parent)
    os.close(fd)
    temporary_path = Path(temporary)
    try:
        temporary_path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def normalize_gage(values: pd.Series) -> pd.Series:
    return values.astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(8)


def eligible_inventory() -> pd.DataFrame:
    """Return every basin that contributed any primary Stage-18 result."""
    inventory = pd.read_csv(INVENTORY_FILE, dtype={"GAGE_ID": str})
    attributes = pd.read_csv(
        ATTRIBUTES_FILE,
        usecols=["GAGE_ID", "AGGECOREGION", "quality_tier", "is_reference",
                 "record_years", "BFI_AVE", "aridity", "snow_fraction", "log_area"],
        dtype={"GAGE_ID": str}, low_memory=False,
    )
    metrics = pd.read_csv(
        STAGE18_METRICS,
        usecols=["GAGE_ID", "model", "threshold_quantile", "lead_days"],
        dtype={"GAGE_ID": str}, low_memory=False,
    )
    for frame in [inventory, attributes, metrics]:
        frame["GAGE_ID"] = normalize_gage(frame["GAGE_ID"])
    primary = metrics[metrics["model"].eq("regularized_power_state_empirical")].copy()
    eligibility = primary.groupby("GAGE_ID", as_index=False).agg(
        max_stage18_lead=("lead_days", "max")
    )
    eligible = eligibility[["GAGE_ID", "max_stage18_lead"]]
    output = eligible.merge(inventory[["GAGE_ID", "file"]], on="GAGE_ID", how="inner")
    output = output.merge(attributes, on="GAGE_ID", how="left")
    output["spatial_group"] = output["AGGECOREGION"].fillna("Unknown").astype(str)
    output["file"] = output["GAGE_ID"].map(lambda g: str(DAILY_ROOT / f"{g}.csv.gz"))
    return output.sort_values("GAGE_ID").reset_index(drop=True)


def all_accepted_inventory() -> pd.DataFrame:
    """Return all 5,227 Tier-1/Tier-2 basins accepted by Stage 8/10."""
    inventory = pd.read_csv(INVENTORY_FILE, dtype={"GAGE_ID": str})
    attributes = pd.read_csv(
        ATTRIBUTES_FILE,
        usecols=["GAGE_ID", "AGGECOREGION", "quality_tier", "is_reference",
                 "record_years", "BFI_AVE", "aridity", "snow_fraction", "log_area"],
        dtype={"GAGE_ID": str}, low_memory=False,
    )
    inventory["GAGE_ID"] = normalize_gage(inventory["GAGE_ID"])
    attributes["GAGE_ID"] = normalize_gage(attributes["GAGE_ID"])
    output = inventory[["GAGE_ID", "file"]].merge(attributes, on="GAGE_ID", how="left")
    output["spatial_group"] = output["AGGECOREGION"].fillna("Unknown").astype(str)
    output["file"] = output["GAGE_ID"].map(lambda g: str(DAILY_ROOT / f"{g}.csv.gz"))
    return output.sort_values("GAGE_ID").reset_index(drop=True)


def load_daily(path: str | Path) -> pd.DataFrame:
    columns = [
        "GAGE_ID", "date", "q_mm_day", "pr_mm", "pet_mm", "tmean_c",
        "snowmelt_risk",
    ]
    frame = pd.read_csv(path, usecols=columns, parse_dates=["date"], low_memory=False)
    frame = frame.sort_values("date").drop_duplicates("date").reset_index(drop=True)
    for name in ["q_mm_day", "pr_mm", "pet_mm", "tmean_c"]:
        frame[name] = pd.to_numeric(frame[name], errors="coerce")
    if frame["snowmelt_risk"].dtype != bool:
        frame["snowmelt_risk"] = (
            frame["snowmelt_risk"].astype(str).str.lower().map({"true": True, "false": False})
        ).astype("boolean")
    return frame


def dry_spell_age(mask: pd.Series) -> np.ndarray:
    values = mask.fillna(False).to_numpy(bool)
    result = np.zeros(len(values), dtype=float)
    count = 0
    for index, value in enumerate(values):
        count = count + 1 if value else 0
        result[index] = count
    return result


def initialization_features(frame: pd.DataFrame, scale: float, config: dict) -> tuple[pd.DataFrame, list[str]]:
    q = frame["q_mm_day"] / scale
    log_q = np.log(np.maximum(q, 1e-10))
    pr = frame["pr_mm"]
    pet = frame["pet_mm"]
    temperature = frame["tmean_c"]
    current_dry = pr.le(config["precipitation_threshold_mm_day"]) & frame["snowmelt_risk"].eq(False)
    dates = frame["date"]
    result = pd.DataFrame({
        "date": dates,
        "log_q": log_q,
        "dlog_q_1": log_q - log_q.shift(1),
        "dlog_q_3": (log_q - log_q.shift(3)) / 3.0,
        "dlog_q_7": (log_q - log_q.shift(7)) / 7.0,
        "dlog_q_14": (log_q - log_q.shift(14)) / 14.0,
        "log_q_minus_mean_7": log_q - log_q.rolling(7, min_periods=4).mean(),
        "log_q_volatility_7": log_q.diff().rolling(7, min_periods=4).std(),
        "log_q_volatility_14": log_q.diff().rolling(14, min_periods=7).std(),
        "dry_spell_age": dry_spell_age(current_dry),
        "pr_sum_7": pr.rolling(7, min_periods=4).sum(),
        "pr_sum_30": pr.rolling(30, min_periods=15).sum(),
        "pr_sum_90": pr.rolling(90, min_periods=45).sum(),
        "pet_sum_7": pet.rolling(7, min_periods=4).sum(),
        "pet_sum_30": pet.rolling(30, min_periods=15).sum(),
        "pet_sum_90": pet.rolling(90, min_periods=45).sum(),
        "tmean_7": temperature.rolling(7, min_periods=4).mean(),
        "tmean_30": temperature.rolling(30, min_periods=15).mean(),
        "antecedent_aridity_30": pet.rolling(30, min_periods=15).sum() / (
            pr.rolling(30, min_periods=15).sum() + 1.0
        ),
        "doy_sin": np.sin(2 * np.pi * dates.dt.dayofyear / 365.25),
        "doy_cos": np.cos(2 * np.pi * dates.dt.dayofyear / 365.25),
    })
    columns = [column for column in result if column != "date"]
    return result, columns


def contiguous_filter(mask: pd.Series, minimum: int) -> pd.Series:
    values = mask.fillna(False).to_numpy(bool)
    groups = np.cumsum(~values)
    counts = pd.Series(values).groupby(groups).transform("sum").to_numpy()
    return pd.Series(values & (counts >= minimum), index=mask.index)


def future_dry_mask(frame: pd.DataFrame, lead: int, config: dict) -> pd.Series:
    antecedent = int(config["antecedent_days"])
    offsets = range(-(antecedent - 1), lead + 1)
    precipitation = pd.concat(
        [frame["pr_mm"].shift(-offset) for offset in offsets], axis=1
    ).max(axis=1, skipna=False)
    melt = frame["snowmelt_risk"].astype("boolean")
    melt_window = pd.concat([melt.shift(-offset) for offset in offsets], axis=1)
    possible_melt = melt_window.eq(True).any(axis=1) | melt_window.isna().any(axis=1)
    return precipitation.le(config["precipitation_threshold_mm_day"]) & ~possible_melt


def conditional_cases(frame: pd.DataFrame, lead: int, scale: float, config: dict) -> pd.DataFrame:
    dates = frame["date"]
    q = frame["q_mm_day"] / scale
    future = q.shift(-lead)
    consecutive = (dates.shift(-lead) - dates).dt.days.eq(lead)
    valid = consecutive & q.gt(0) & future.ge(0)
    selected = contiguous_filter(
        valid & future_dry_mask(frame, lead, config), int(config["minimum_contiguous_days"])
    )
    features, columns = initialization_features(frame, scale, config)
    output = features.loc[selected].copy()
    output["future_date"] = dates.shift(-lead).loc[selected].to_numpy()
    output["q"] = q.loc[selected].to_numpy(float)
    output["q_next"] = future.loc[selected].to_numpy(float)
    output["dq"] = output["q_next"] - output["q"]
    return output[["date", "future_date", "q", "q_next", "dq", *columns]]


def current_dry_cases(frame: pd.DataFrame, lead: int, scale: float, config: dict) -> pd.DataFrame:
    """Cases known to be dry at initialization; future forcing is not screened."""
    dates = frame["date"]
    q = frame["q_mm_day"] / scale
    future = q.shift(-lead)
    consecutive = (dates.shift(-lead) - dates).dt.days.eq(lead)
    current_dry = frame["pr_mm"].le(config["precipitation_threshold_mm_day"]) & frame[
        "snowmelt_risk"
    ].eq(False)
    features, columns = initialization_features(frame, scale, config)
    selected = (
        consecutive & q.gt(0) & future.ge(0) & current_dry
        & features["dry_spell_age"].ge(int(config["antecedent_days"]))
    )
    output = features.loc[selected].copy()
    output["future_date"] = dates.shift(-lead).loc[selected].to_numpy()
    output["q"] = q.loc[selected].to_numpy(float)
    output["q_next"] = future.loc[selected].to_numpy(float)
    output["future_dry"] = future_dry_mask(frame, lead, config).loc[selected].to_numpy(bool)
    return output[["date", "future_date", "q", "q_next", "future_dry", *columns]]


def split_cases(cases: pd.DataFrame, config: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    fit = cases[cases["future_date"].le(pd.Timestamp(config["fit_end"]))].copy()
    calibration = cases[
        cases["date"].ge(pd.Timestamp(config["calibration_start"]))
        & cases["future_date"].le(pd.Timestamp(config["calibration_end"]))
    ].copy()
    evaluation = cases[
        cases["date"].ge(pd.Timestamp(config["evaluation_start"]))
        & cases["future_date"].le(pd.Timestamp(config["evaluation_end"]))
    ].copy()
    return fit, calibration, evaluation


def fit_location_scale(train: pd.DataFrame, bins: int = 10, minimum_bin: int = 30) -> dict:
    q = train["q"].to_numpy(float)
    dq = train["dq"].to_numpy(float)
    edges = np.unique(np.quantile(q, np.linspace(0, 1, bins + 1)))
    if len(edges) < 7:
        raise ValueError("too_few_unique_flow_bins")
    assignment = np.searchsorted(edges[1:-1], q, side="right")
    rows = []
    for group in range(len(edges) - 1):
        selected = assignment == group
        if selected.sum() < minimum_bin:
            continue
        variance = np.var(dq[selected], ddof=1)
        if np.isfinite(variance) and variance > 0:
            rows.append((np.exp(np.mean(np.log(q[selected]))), np.mean(dq[selected]), variance,
                         int(selected.sum())))
    if len(rows) < 6:
        raise ValueError("too_few_valid_location_scale_bins")
    summary = np.asarray(rows, dtype=float)
    exponent_fit = stats.linregress(np.log(summary[:, 0]), np.log(summary[:, 2]))
    exponent = float(np.clip(exponent_fit.slope, 0, 4))
    coefficient = float(np.exp(np.average(
        np.log(summary[:, 2]) - exponent * np.log(summary[:, 0]), weights=summary[:, 3]
    )))
    order = np.argsort(summary[:, 0])
    centers, means = summary[order, 0], summary[order, 1]
    q_model = np.clip(q, centers[0], centers[-1])
    mean = np.interp(np.log(q_model), np.log(centers), means)
    sigma = np.sqrt(coefficient * q_model**exponent)
    residual = (dq - mean) / sigma
    residual = residual[np.isfinite(residual)]
    if len(residual) < 500 or np.std(residual) <= 0:
        raise ValueError("insufficient_standardized_residuals")
    residual_edges = np.unique(np.quantile(q, np.linspace(0, 1, 6)))
    residual_assignment = np.searchsorted(residual_edges[1:-1], q, side="right")
    groups = {}
    for group in range(len(residual_edges) - 1):
        values = ((dq - mean) / sigma)[residual_assignment == group]
        values = np.sort(values[np.isfinite(values)])
        if len(values) >= minimum_bin:
            groups[group] = values
    return {
        "edges": edges, "centers": centers, "means": means,
        "q_low": centers[0], "q_high": centers[-1], "exponent": exponent,
        "coefficient": coefficient, "residual": np.sort(residual),
        "residual_edges": residual_edges, "residual_groups": groups,
    }


def location_scale(q: np.ndarray, model: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    q_model = np.clip(np.asarray(q, float), model["q_low"], model["q_high"])
    mean = np.interp(np.log(q_model), np.log(model["centers"]), model["means"])
    sigma = np.sqrt(model["coefficient"] * q_model**model["exponent"])
    return q_model, mean, sigma


def empirical_members(q: np.ndarray, model: dict, members: int) -> np.ndarray:
    probabilities = (np.arange(members) + 0.5) / members
    q_model, mean, sigma = location_scale(q, model)
    assignments = np.searchsorted(model["residual_edges"][1:-1], q_model, side="right")
    result = np.empty((len(q_model), members), float)
    for group in np.unique(assignments):
        selected = assignments == group
        residual = model["residual_groups"].get(int(group), model["residual"])
        innovations = np.quantile(residual, probabilities)
        result[selected] = q[selected, None] + mean[selected, None] + sigma[selected, None] * innovations
    return np.maximum(result, 0)


def student_members(q: np.ndarray, model: dict, members: int) -> np.ndarray:
    probabilities = (np.arange(members) + 0.5) / members
    df, loc, scale = stats.t.fit(model["residual"])
    innovations = stats.t.ppf(probabilities, df, loc=loc, scale=max(scale, 1e-8))
    _, mean, sigma = location_scale(q, model)
    return np.maximum(q[:, None] + mean[:, None] + sigma[:, None] * innovations, 0)


def gpd_splice_quantiles(residual: np.ndarray, probabilities: np.ndarray,
                         tail_probability: float, minimum: int) -> tuple[np.ndarray, dict]:
    residual = np.sort(np.asarray(residual, float))
    threshold = float(np.quantile(residual, tail_probability))
    exceedance = threshold - residual[residual < threshold]
    diagnostics = {"gpd_shape": np.nan, "gpd_scale": np.nan, "gpd_exceedances": len(exceedance)}
    if len(exceedance) < minimum or np.allclose(exceedance, exceedance[0]):
        return np.quantile(residual, probabilities), diagnostics
    shape, _, scale = stats.genpareto.fit(exceedance, floc=0)
    if not np.isfinite(shape) or not np.isfinite(scale) or scale <= 0:
        return np.quantile(residual, probabilities), diagnostics
    output = np.empty_like(probabilities)
    tail = probabilities < tail_probability
    survival_probability = np.clip(probabilities[tail] / tail_probability, 1e-10, 1 - 1e-10)
    output[tail] = threshold - stats.genpareto.ppf(
        1 - survival_probability, shape, loc=0, scale=scale
    )
    body = residual[residual >= threshold]
    body_probability = (probabilities[~tail] - tail_probability) / (1 - tail_probability)
    output[~tail] = np.quantile(body, np.clip(body_probability, 0, 1))
    diagnostics.update({"gpd_shape": float(shape), "gpd_scale": float(scale)})
    return output, diagnostics


def gpd_members(q: np.ndarray, model: dict, members: int, config: dict) -> tuple[np.ndarray, dict]:
    probabilities = (np.arange(members) + 0.5) / members
    innovations, diagnostics = gpd_splice_quantiles(
        model["residual"], probabilities, float(config["gpd_tail_probability"]),
        int(config["gpd_minimum_exceedances"]),
    )
    _, mean, sigma = location_scale(q, model)
    return np.maximum(q[:, None] + mean[:, None] + sigma[:, None] * innovations, 0), diagnostics


def _asymmetric_laplace_logpdf(x: np.ndarray, kappa: float, loc: float, scale: float) -> np.ndarray:
    return stats.laplace_asymmetric.logpdf(x, kappa, loc=loc, scale=scale)


def fit_asymmetric_laplace_mixture(values: np.ndarray, seed: int) -> dict:
    """Fit a two-component asymmetric-Laplace mixture by penalized maximum likelihood."""
    rng = np.random.default_rng(seed)
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    if len(values) > 5000:
        values = rng.choice(values, 5000, replace=False)
    q25, median, q75 = np.quantile(values, [0.25, 0.5, 0.75])
    spread = max((q75 - q25) / 1.35, 0.05)

    def unpack(theta):
        weight = expit(theta[0])
        k1, k2 = np.exp(theta[[1, 4]])
        loc1, loc2 = theta[[2, 5]]
        scale1, scale2 = np.exp(theta[[3, 6]])
        return weight, k1, loc1, scale1, k2, loc2, scale2

    def objective(theta):
        weight, k1, loc1, scale1, k2, loc2, scale2 = unpack(theta)
        components = np.vstack([
            np.log(weight + 1e-12) + _asymmetric_laplace_logpdf(values, k1, loc1, scale1),
            np.log(1 - weight + 1e-12) + _asymmetric_laplace_logpdf(values, k2, loc2, scale2),
        ])
        penalty = 0.002 * float(np.sum(theta[[1, 3, 4, 6]] ** 2))
        return float(-np.sum(logsumexp(components, axis=0)) + penalty)

    starts = [
        np.array([0.0, 0.0, q25, np.log(spread), 0.0, q75, np.log(spread)]),
        np.array([1.0, np.log(0.7), q25, np.log(spread), np.log(1.5), q75, np.log(spread)]),
    ]
    location_low = float(np.quantile(values, 0.001) - 2 * np.std(values))
    location_high = float(np.quantile(values, 0.999) + 2 * np.std(values))
    bounds = [
        (-6, 6), (-3, 3), (location_low, location_high), (-5, 3),
        (-3, 3), (location_low, location_high), (-5, 3),
    ]
    best = None
    for start in starts:
        fit = optimize.minimize(
            objective, start, method="L-BFGS-B", bounds=bounds,
            options={"maxiter": 250},
        )
        if fit.success and (best is None or fit.fun < best.fun):
            best = fit
    if best is None:
        raise ValueError("asymmetric_laplace_mixture_fit_failed")
    values_out = unpack(best.x)
    return dict(zip(["weight", "kappa1", "loc1", "scale1", "kappa2", "loc2", "scale2"], values_out))


def asymmetric_laplace_mixture_members(q: np.ndarray, model: dict, members: int,
                                        seed: int) -> tuple[np.ndarray, dict]:
    fitted = fit_asymmetric_laplace_mixture(model["residual"], seed)
    rng = np.random.default_rng(seed + 37)
    draws = max(20000, members * 100)
    first = rng.random(draws) < fitted["weight"]
    innovations = np.empty(draws)
    innovations[first] = stats.laplace_asymmetric.rvs(
        fitted["kappa1"], loc=fitted["loc1"], scale=fitted["scale1"],
        size=first.sum(), random_state=rng,
    )
    innovations[~first] = stats.laplace_asymmetric.rvs(
        fitted["kappa2"], loc=fitted["loc2"], scale=fitted["scale2"],
        size=(~first).sum(), random_state=rng,
    )
    probabilities = (np.arange(members) + 0.5) / members
    innovation_quantiles = np.quantile(innovations, probabilities)
    _, mean, sigma = location_scale(q, model)
    ensemble = q[:, None] + mean[:, None] + sigma[:, None] * innovation_quantiles
    return np.maximum(ensemble, 0), fitted


def qbin_probability(train_q: np.ndarray, train_y: np.ndarray, test_q: np.ndarray,
                     bins: int = 10) -> np.ndarray:
    edges = np.unique(np.quantile(train_q, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return np.full(len(test_q), (train_y.sum() + 0.5) / (len(train_y) + 1))
    train_group = np.searchsorted(edges[1:-1], train_q, side="right")
    test_group = np.searchsorted(edges[1:-1], test_q, side="right")
    global_p = (train_y.sum() + 0.5) / (len(train_y) + 1)
    group_p = {}
    for group in range(len(edges) - 1):
        selected = train_group == group
        group_p[group] = ((train_y[selected].sum() + 0.5) / (selected.sum() + 1)
                          if selected.any() else global_p)
    return np.asarray([group_p[int(group)] for group in test_group])


def fit_logistic(train: pd.DataFrame, target: np.ndarray, features: list[str]):
    model = make_pipeline(
        SimpleImputer(strategy="median"), StandardScaler(),
        LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs"),
    )
    model.fit(train[features], target)
    return model


def idr_probability(train_q: np.ndarray, train_y: np.ndarray, test_q: np.ndarray) -> np.ndarray:
    model = IsotonicRegression(increasing=False, y_min=0, y_max=1, out_of_bounds="clip")
    model.fit(np.log(np.maximum(train_q, 1e-10)), train_y.astype(float))
    return model.predict(np.log(np.maximum(test_q, 1e-10)))


def quantile_forest_members(train: pd.DataFrame, test: pd.DataFrame, features: list[str],
                            config: dict, seed: int) -> np.ndarray:
    """Quantile-regression-forest analogue using empirical outcomes in terminal leaves."""
    imputer = SimpleImputer(strategy="median")
    x_train = imputer.fit_transform(train[features])
    x_test = imputer.transform(test[features])
    y_train = train["q_next"].to_numpy(float)
    forest = RandomForestRegressor(
        n_estimators=int(config["random_forest_trees"]),
        min_samples_leaf=int(config["random_forest_min_leaf"]), max_features=0.8,
        n_jobs=1, random_state=seed,
    )
    forest.fit(x_train, y_train)
    result = np.empty((len(test), len(forest.estimators_)), dtype=float)
    rng = np.random.default_rng(seed + 51)
    for column, tree in enumerate(forest.estimators_):
        train_leaf = tree.apply(x_train)
        test_leaf = tree.apply(x_test)
        order = np.argsort(train_leaf)
        sorted_leaf = train_leaf[order]
        sorted_y = y_train[order]
        unique, starts = np.unique(sorted_leaf, return_index=True)
        stops = np.r_[starts[1:], len(sorted_leaf)]
        lookup = {int(leaf): sorted_y[start:stop] for leaf, start, stop in zip(unique, starts, stops)}
        for row, leaf in enumerate(test_leaf):
            candidates = lookup[int(leaf)]
            result[row, column] = candidates[rng.integers(0, len(candidates))]
    return np.maximum(result, 0)


def ensemble_probability(ensemble: np.ndarray, threshold: float) -> np.ndarray:
    return (np.sum(ensemble <= threshold, axis=1) + 0.5) / (ensemble.shape[1] + 1)


def ensemble_crps(ensemble: np.ndarray, observation: np.ndarray) -> np.ndarray:
    values = np.sort(ensemble, axis=1)
    members = values.shape[1]
    first = np.mean(np.abs(values - observation[:, None]), axis=1)
    coefficient = 2 * np.arange(1, members + 1) - members - 1
    second = np.sum(values * coefficient[None, :], axis=1) / members**2
    return first - second


def lower_tail_twcrps(ensemble: np.ndarray, observation: np.ndarray,
                      threshold: float, grid_points: int = 80) -> np.ndarray:
    """Threshold-weighted CRPS with weight one below the low-flow threshold."""
    maximum = max(float(threshold), 1e-10)
    grid = np.linspace(0, maximum, grid_points)
    cdf = np.mean(ensemble[:, :, None] <= grid[None, None, :], axis=1)
    observed_cdf = observation[:, None] <= grid[None, :]
    return np.trapezoid((cdf - observed_cdf) ** 2, grid, axis=1)


def reliability_error(probability: np.ndarray, target: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0, 1, bins + 1)
    assignment = np.clip(np.digitize(probability, edges[1:-1]), 0, bins - 1)
    total = 0.0
    for group in range(bins):
        selected = assignment == group
        if selected.any():
            total += selected.mean() * abs(probability[selected].mean() - target[selected].mean())
    return float(total)


def score_probability(probability: np.ndarray, target: np.ndarray, clip: float) -> dict:
    p = np.clip(np.asarray(probability, float), clip, 1 - clip)
    y = np.asarray(target, bool)
    result = {
        "n": len(y), "events": int(y.sum()), "event_rate": float(y.mean()),
        "brier": float(np.mean((p - y.astype(float)) ** 2)),
        "log_loss": float(-np.mean(y * np.log(p) + (~y) * np.log(1 - p))),
        "probability_bias": float(np.mean(p) - np.mean(y)),
        "reliability_error": reliability_error(p, y),
        "roc_auc": np.nan, "average_precision": np.nan,
    }
    if y.any() and (~y).any():
        result["roc_auc"] = float(roc_auc_score(y, p))
        result["average_precision"] = float(average_precision_score(y, p))
    return result


def score_ensemble(ensemble: np.ndarray, observation: np.ndarray, threshold: float) -> dict:
    low, high = np.quantile(ensemble, [0.05, 0.95], axis=1)
    return {
        "crps": float(np.mean(ensemble_crps(ensemble, observation))),
        "twcrps_below_threshold": float(np.mean(lower_tail_twcrps(ensemble, observation, threshold))),
        "interval_90_coverage": float(np.mean((observation >= low) & (observation <= high))),
        "interval_90_width": float(np.mean(high - low)),
    }


def water_year(dates: Iterable) -> np.ndarray:
    dates = pd.to_datetime(pd.Series(dates))
    return (dates.dt.year + dates.dt.month.ge(10).astype(int)).to_numpy(int)
