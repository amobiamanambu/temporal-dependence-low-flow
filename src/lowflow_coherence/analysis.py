"""Input validation and summaries for the compact real-data example."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .metrics import brier_score, relative_skill_percent


REQUIRED_FORECAST_COLUMNS = {
    "GAGE_ID",
    "issue_date",
    "lead_days",
    "lead_is_scorable",
    "observed_event",
    "forecast_event_probability",
}


def normalize_gage_id(value: object) -> str:
    """Return an eight-character USGS station identifier."""
    text = str(value).strip()
    if text.endswith(".0") and text[:-2].isdigit():
        text = text[:-2]
    digits = "".join(character for character in text if character.isdigit())
    return digits.zfill(8) if digits else ""


def load_sample_forecasts(path: str | Path) -> pd.DataFrame:
    """Read and validate the compact forecast table."""
    frame = pd.read_csv(path, dtype={"GAGE_ID": "string"})
    missing = sorted(REQUIRED_FORECAST_COLUMNS.difference(frame.columns))
    if missing:
        raise ValueError(f"sample forecast file is missing columns: {missing}")
    frame["GAGE_ID"] = frame["GAGE_ID"].map(normalize_gage_id)
    frame["issue_date"] = pd.to_datetime(frame["issue_date"], errors="raise")
    frame["lead_days"] = pd.to_numeric(frame["lead_days"], errors="raise").astype(int)
    frame["forecast_event_probability"] = pd.to_numeric(
        frame["forecast_event_probability"], errors="raise"
    )
    if frame["GAGE_ID"].eq("").any():
        raise ValueError("one or more GAGE_ID values could not be normalized")
    if not frame["forecast_event_probability"].between(0, 1).all():
        raise ValueError("forecast_event_probability must lie in [0, 1]")
    frame["lead_is_scorable"] = _as_boolean(frame["lead_is_scorable"])
    frame["observed_event"] = _as_boolean(frame["observed_event"])
    return frame.sort_values(["GAGE_ID", "issue_date", "lead_days"]).reset_index(drop=True)


def _as_boolean(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype(bool)
    lookup = {"true": True, "false": False, "1": True, "0": False}
    converted = series.astype(str).str.strip().str.lower().map(lookup)
    if converted.isna().any():
        raise ValueError(f"could not parse boolean values in {series.name}")
    return converted.astype(bool)


def horizon_verification(frame: pd.DataFrame) -> pd.DataFrame:
    """Summarize pooled Q10 occurrence verification by forecast window."""
    scored = frame.loc[frame["lead_is_scorable"]].copy()
    records: list[dict[str, float | int]] = []
    for lead, group in scored.groupby("lead_days", sort=True):
        observed = group["observed_event"].astype(float).to_numpy()
        probability = group["forecast_event_probability"].to_numpy(dtype=float)
        records.append(
            {
                "lead_days": int(lead),
                "gages": int(group["GAGE_ID"].nunique()),
                "forecast_initializations": int(len(group)),
                "observed_event_frequency": float(observed.mean()),
                "mean_forecast_probability": float(probability.mean()),
                "calibration_bias": float(probability.mean() - observed.mean()),
                "brier_score": brier_score(observed, probability),
            }
        )
    return pd.DataFrame.from_records(records)


def basin_verification(frame: pd.DataFrame) -> pd.DataFrame:
    """Summarize Q10 occurrence verification separately by gage and window."""
    scored = frame.loc[frame["lead_is_scorable"]].copy()
    records: list[dict[str, float | int | str]] = []
    for (gage, lead), group in scored.groupby(["GAGE_ID", "lead_days"], sort=True):
        observed = group["observed_event"].astype(float).to_numpy()
        probability = group["forecast_event_probability"].to_numpy(dtype=float)
        records.append(
            {
                "GAGE_ID": str(gage),
                "lead_days": int(lead),
                "forecast_initializations": int(len(group)),
                "observed_event_frequency": float(observed.mean()),
                "mean_forecast_probability": float(probability.mean()),
                "calibration_bias": float(probability.mean() - observed.mean()),
                "brier_score": brier_score(observed, probability),
            }
        )
    return pd.DataFrame.from_records(records)


def reliability_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """Return fixed-bin reliability points for each forecast window."""
    scored = frame.loc[frame["lead_is_scorable"]].copy()
    edges = np.array([-1e-12, 0.05, 0.10, 0.20, 0.40, 0.60, 0.80, 1.0 + 1e-12])
    scored["probability_bin"] = pd.cut(
        scored["forecast_event_probability"], edges, labels=False, include_lowest=True
    )
    grouped = (
        scored.groupby(["lead_days", "probability_bin"], observed=True)
        .agg(
            forecast_initializations=("observed_event", "size"),
            mean_forecast_probability=("forecast_event_probability", "mean"),
            observed_event_frequency=("observed_event", "mean"),
        )
        .reset_index()
    )
    grouped["probability_bin"] = grouped["probability_bin"].astype(int)
    return grouped


def verify_same_marginal_scores(path: str | Path) -> pd.DataFrame:
    """Recalculate reported relative score reductions from archived scores."""
    frame = pd.read_csv(path)
    required = {"candidate_score", "reference_score", "relative_skill_percent"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"same-marginal table is missing columns: {missing}")
    recalculated = [
        relative_skill_percent(candidate, reference)
        for candidate, reference in zip(frame["candidate_score"], frame["reference_score"])
    ]
    result = frame.copy()
    result["recalculated_relative_skill_percent"] = recalculated
    result["absolute_recalculation_difference"] = np.abs(
        result["recalculated_relative_skill_percent"] - result["relative_skill_percent"]
    )
    return result
