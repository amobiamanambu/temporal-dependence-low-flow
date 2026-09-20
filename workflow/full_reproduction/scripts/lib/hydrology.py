"""Hydrologic unit conversions and the temperature-index snow proxy."""

from __future__ import annotations

import numpy as np
import pandas as pd


CFS_TO_MM_DAY_PER_KM2 = 2.446575545


def cfs_to_mm_day(q_cfs: pd.Series | np.ndarray, area_km2: float) -> np.ndarray:
    if not np.isfinite(area_km2) or area_km2 <= 0:
        return np.full(len(q_cfs), np.nan)
    return np.asarray(q_cfs, dtype=float) * CFS_TO_MM_DAY_PER_KM2 / area_km2


def precipitation_snow_fraction(
    tmean_c: pd.Series | np.ndarray, snow_temperature_c: float, rain_temperature_c: float
) -> np.ndarray:
    temperature = np.asarray(tmean_c, dtype=float)
    fraction = (rain_temperature_c - temperature) / (rain_temperature_c - snow_temperature_c)
    return np.clip(fraction, 0.0, 1.0)


def add_temperature_index_snow_proxy(frame: pd.DataFrame, settings: dict) -> pd.DataFrame:
    """Add a transparent degree-day snow-storage proxy.

    This is a forcing-screening proxy, not a reconstruction of SWE. Missing forcing
    breaks the storage recursion and restarts it at zero on the next valid day.
    """
    out = frame.sort_values("date").copy()
    phase = precipitation_snow_fraction(
        out["tmean_c"], settings["snow_temperature_c"], settings["rain_temperature_c"]
    )
    pr = out["pr_mm"].to_numpy(dtype=float)
    temperature = out["tmean_c"].to_numpy(dtype=float)
    snowfall = pr * phase
    potential_melt = settings["degree_day_factor_mm_c_day"] * np.maximum(
        temperature - settings["melt_temperature_c"], 0.0
    )
    storage_before = np.full(len(out), np.nan)
    storage_after = np.full(len(out), np.nan)
    melt = np.full(len(out), np.nan)
    storage = 0.0
    for index in range(len(out)):
        if not (np.isfinite(pr[index]) and np.isfinite(temperature[index])):
            storage = 0.0
            continue
        storage_before[index] = storage
        available = max(0.0, storage + max(0.0, snowfall[index]))
        melt[index] = min(available, max(0.0, potential_melt[index]))
        storage = max(0.0, available - melt[index])
        storage_after[index] = storage
    out["snow_fraction"] = phase
    out["snowfall_proxy_mm"] = snowfall
    out["snow_storage_before_mm"] = storage_before
    out["snowmelt_proxy_mm"] = melt
    out["snow_storage_after_mm"] = storage_after
    out["snowmelt_risk"] = (
        (out["snow_storage_before_mm"] >= settings["minimum_snow_storage_mm"])
        & (out["snowmelt_proxy_mm"] > 0)
    )
    return out
