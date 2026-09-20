"""Shared data, feature, scoring, and I/O utilities for the final workflow."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


BENCHMARK_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = BENCHMARK_ROOT.parent
RESULTS_ROOT = BENCHMARK_ROOT / "results"
DAILY_ROOT = PROJECT_ROOT / "continental_run" / "09_snow_proxy" / "daily"
INVENTORY_FILE = (
    PROJECT_ROOT / "continental_run" / "09_snow_proxy" / "snow_daily_manifest.csv"
)
ATTRIBUTES_FILE = (
    PROJECT_ROOT / "continental_run" / "06_basin_index" / "basin_index.csv"
)
DEVELOPMENT_PANEL = BENCHMARK_ROOT / "development_panel.csv"


def load_benchmark_config() -> dict:
    return json.loads((BENCHMARK_ROOT / "config.json").read_text(encoding="utf-8"))


def ensure_dirs() -> None:
    for name in ("00_audit", "01_panel"):
        (RESULTS_ROOT / name).mkdir(parents=True, exist_ok=True)


def atomic_csv(frame: pd.DataFrame, path: Path, **kwargs) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = ".csv.gz" if str(path).endswith(".gz") else ".csv"
    fd, temporary = tempfile.mkstemp(
        prefix=path.stem + ".", suffix=suffix, dir=path.parent
    )
    os.close(fd)
    temporary_path = Path(temporary)
    try:
        frame.to_csv(temporary_path, index=False, **kwargs)
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def atomic_json(value: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=path.stem + ".", suffix=".json", dir=path.parent
    )
    os.close(fd)
    temporary_path = Path(temporary)
    try:
        temporary_path.write_text(
            json.dumps(value, indent=2) + "\n", encoding="utf-8"
        )
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def normalize_gage(values: pd.Series) -> pd.Series:
    return values.astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(8)


def eligible_inventory() -> pd.DataFrame:
    """Return the frozen 192-basin development panel used by the study."""
    output = pd.read_csv(DEVELOPMENT_PANEL, dtype={"GAGE_ID": str}, low_memory=False)
    output["GAGE_ID"] = normalize_gage(output["GAGE_ID"])
    output["file"] = output["GAGE_ID"].map(
        lambda gage: str(DAILY_ROOT / f"{gage}.csv.gz")
    )
    return output.sort_values("GAGE_ID").reset_index(drop=True)


def all_accepted_inventory() -> pd.DataFrame:
    """Return all 5,227 Tier-1/Tier-2 basins accepted by data screening."""
    inventory = pd.read_csv(INVENTORY_FILE, dtype={"GAGE_ID": str})
    attributes = pd.read_csv(
        ATTRIBUTES_FILE,
        usecols=[
            "GAGE_ID",
            "AGGECOREGION",
            "quality_tier",
            "is_reference",
            "record_years",
            "BFI_AVE",
            "aridity",
            "snow_fraction",
            "log_area",
        ],
        dtype={"GAGE_ID": str},
        low_memory=False,
    )
    inventory["GAGE_ID"] = normalize_gage(inventory["GAGE_ID"])
    attributes["GAGE_ID"] = normalize_gage(attributes["GAGE_ID"])
    output = inventory[["GAGE_ID", "file"]].merge(
        attributes, on="GAGE_ID", how="left"
    )
    output["spatial_group"] = output["AGGECOREGION"].fillna("Unknown").astype(str)
    output["file"] = output["GAGE_ID"].map(
        lambda gage: str(DAILY_ROOT / f"{gage}.csv.gz")
    )
    return output.sort_values("GAGE_ID").reset_index(drop=True)


def load_daily(path: str | Path) -> pd.DataFrame:
    columns = [
        "GAGE_ID",
        "date",
        "q_mm_day",
        "pr_mm",
        "pet_mm",
        "tmean_c",
        "snowmelt_risk",
    ]
    frame = pd.read_csv(path, usecols=columns, parse_dates=["date"], low_memory=False)
    frame = frame.sort_values("date").drop_duplicates("date").reset_index(drop=True)
    for name in ("q_mm_day", "pr_mm", "pet_mm", "tmean_c"):
        frame[name] = pd.to_numeric(frame[name], errors="coerce")
    if frame["snowmelt_risk"].dtype != bool:
        frame["snowmelt_risk"] = (
            frame["snowmelt_risk"]
            .astype(str)
            .str.lower()
            .map({"true": True, "false": False})
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


def initialization_features(
    frame: pd.DataFrame, scale: float, config: dict
) -> tuple[pd.DataFrame, list[str]]:
    q = frame["q_mm_day"] / scale
    log_q = np.log(np.maximum(q, 1e-10))
    precipitation = frame["pr_mm"]
    pet = frame["pet_mm"]
    temperature = frame["tmean_c"]
    current_dry = precipitation.le(config["precipitation_threshold_mm_day"]) & frame[
        "snowmelt_risk"
    ].eq(False)
    dates = frame["date"]
    result = pd.DataFrame(
        {
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
            "pr_sum_7": precipitation.rolling(7, min_periods=4).sum(),
            "pr_sum_30": precipitation.rolling(30, min_periods=15).sum(),
            "pr_sum_90": precipitation.rolling(90, min_periods=45).sum(),
            "pet_sum_7": pet.rolling(7, min_periods=4).sum(),
            "pet_sum_30": pet.rolling(30, min_periods=15).sum(),
            "pet_sum_90": pet.rolling(90, min_periods=45).sum(),
            "tmean_7": temperature.rolling(7, min_periods=4).mean(),
            "tmean_30": temperature.rolling(30, min_periods=15).mean(),
            "antecedent_aridity_30": pet.rolling(30, min_periods=15).sum()
            / (precipitation.rolling(30, min_periods=15).sum() + 1.0),
            "doy_sin": np.sin(2 * np.pi * dates.dt.dayofyear / 365.25),
            "doy_cos": np.cos(2 * np.pi * dates.dt.dayofyear / 365.25),
        }
    )
    columns = [column for column in result if column != "date"]
    return result, columns


def fit_logistic(train: pd.DataFrame, target: np.ndarray, features: list[str]):
    model = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs"),
    )
    model.fit(train[features], target)
    return model


def ensemble_crps(ensemble: np.ndarray, observation: np.ndarray) -> np.ndarray:
    values = np.sort(ensemble, axis=1)
    members = values.shape[1]
    first = np.mean(np.abs(values - observation[:, None]), axis=1)
    coefficient = 2 * np.arange(1, members + 1) - members - 1
    second = np.sum(values * coefficient[None, :], axis=1) / members**2
    return first - second


def water_year(dates: Iterable) -> np.ndarray:
    dates = pd.to_datetime(pd.Series(dates))
    return (dates.dt.year + dates.dt.month.ge(10).astype(int)).to_numpy(int)
