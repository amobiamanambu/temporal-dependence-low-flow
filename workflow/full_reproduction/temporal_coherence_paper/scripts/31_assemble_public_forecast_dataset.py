#!/usr/bin/env python3
"""Assemble and validate the versioned 30--90-day public forecast dataset.

Run this only after Stage 30 is complete:

    python temporal_coherence_paper/scripts/31_assemble_public_forecast_dataset.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RELEASE_ROOT = PROJECT_ROOT / "temporal_coherence_paper" / "data_release" / "v1"
WORK_ROOT = RELEASE_ROOT / "work" / "by_basin"
DATA_ROOT = RELEASE_ROOT / "data"
METRICS_ROOT = (
    PROJECT_ROOT / "lowflow_forecast_benchmark" / "results" / "09_extended_forecast"
)
ATTRIBUTES = PROJECT_ROOT / "continental_run" / "06_basin_index" / "basin_index.csv"
LEADS = (30, 45, 60, 90)
VERSION = "1.0.0"
BRIER_REPRODUCTION_TOLERANCE = 1e-5


DICTIONARY = [
    ("data_version", "string", "Release version."),
    ("GAGE_ID", "string", "Eight-character USGS station identifier; leading zeros retained."),
    ("issue_date", "date", "Weekly retrospective forecast initialization date."),
    ("valid_end_date", "date", "Last date in the forecast window."),
    ("lead_days", "integer", "Forecast-window length in days (30, 45, 60, or 90)."),
    ("threshold_name", "string", "Low-flow threshold label; Q10 is the training-period 10th percentile."),
    ("target", "string", "Forecast target; onset means streamflow at forecast initialization is above Q10."),
    ("model", "string", "Frozen basin-specific 101-member hydrograph-state analog."),
    ("spatial_group", "string", "Aggregated GAGES-II ecoregion."),
    ("quality_tier", "string", "Continental workflow record-quality tier."),
    ("is_reference", "boolean", "GAGES-II reference-basin indicator."),
    ("record_years", "number", "Valid discharge record length in years within 1980--2025."),
    ("latitude", "degrees north", "USGS gage latitude from GAGES-II."),
    ("longitude", "degrees east", "USGS gage longitude from GAGES-II."),
    ("initial_flow_mm_day", "mm d-1", "Observed runoff depth on the forecast initialization date."),
    ("threshold_q10_mm_day", "mm d-1", "Basin Q10 threshold fixed using observations through 2015."),
    ("dry_spell_age_days", "days", "Consecutive dry-screen days available at initialization."),
    ("ensemble_members", "count", "Number of forecast members."),
    ("lead_is_scorable", "boolean", "True when held-out event and non-event counts each meet the frozen minimum."),
    ("observed_event", "boolean", "Whether Q10 was crossed at least once in the window."),
    ("forecast_event_probability", "probability", "Ensemble probability of at least one Q10 crossing."),
    ("observed_endpoint_low", "boolean", "Whether flow was at or below Q10 on the final window day."),
    ("forecast_endpoint_probability", "probability", "Ensemble probability of endpoint flow at or below Q10."),
    ("observed_onset_day", "days", "First observed Q10 crossing after forecast initialization; missing if no crossing."),
    ("observed_onset_day_censored", "days", "First crossing, or lead+1 when no crossing occurred."),
    ("forecast_onset_p05", "days", "5th percentile of censored ensemble onset day."),
    ("forecast_onset_p50", "days", "Median censored ensemble onset day."),
    ("forecast_onset_p95", "days", "95th percentile of censored ensemble onset day."),
    ("observed_duration_days", "days", "Observed total low-flow-day count within the window; the historical field name is retained."),
    ("forecast_duration_p05", "days", "5th percentile of ensemble total low-flow-day count; the historical field name is retained."),
    ("forecast_duration_p50", "days", "Median ensemble total low-flow-day count; the historical field name is retained."),
    ("forecast_duration_p95", "days", "95th percentile of ensemble total low-flow-day count; the historical field name is retained."),
    ("observed_deficit_mm", "mm", "Observed cumulative runoff-depth deficit below Q10."),
    ("forecast_deficit_mm_p05", "mm", "5th percentile of ensemble cumulative deficit."),
    ("forecast_deficit_mm_p50", "mm", "Median ensemble cumulative deficit."),
    ("forecast_deficit_mm_p95", "mm", "95th percentile of ensemble cumulative deficit."),
    ("observed_minimum_flow_mm_day", "mm d-1", "Observed minimum runoff depth in the window."),
    ("forecast_minimum_flow_mm_day_p05", "mm d-1", "5th percentile of ensemble minimum flow."),
    ("forecast_minimum_flow_mm_day_p50", "mm d-1", "Median ensemble minimum flow."),
    ("forecast_minimum_flow_mm_day_p95", "mm d-1", "95th percentile of ensemble minimum flow."),
]


def normalize_gage(values: pd.Series) -> pd.Series:
    return values.astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(8)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_csv(frame: pd.DataFrame, path: Path, compression: str | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    frame.to_csv(temporary, index=False, compression=compression)
    temporary.replace(path)


def assemble_lead(files: list[Path], lead: int, destination: Path) -> int:
    writer: pq.ParquetWriter | None = None
    rows = 0
    temporary = destination.with_name(f".{destination.name}.tmp")
    try:
        for path in files:
            table = pq.read_table(path, filters=[("lead_days", "=", int(lead))])
            if table.num_rows == 0:
                continue
            if writer is None:
                writer = pq.ParquetWriter(
                    temporary,
                    table.schema,
                    compression="zstd",
                    use_dictionary=True,
                )
            writer.write_table(table)
            rows += table.num_rows
    finally:
        if writer is not None:
            writer.close()
    if rows == 0:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"No Stage 30 records were available for lead {lead}")
    temporary.replace(destination)
    return rows


def archived_validation_metrics() -> pd.DataFrame:
    frames = []
    for path in sorted(METRICS_ROOT.glob("confirmation_shard_*/by_basin/*_metrics.csv.gz")):
        frame = pd.read_csv(path, dtype={"GAGE_ID": str})
        selected = frame[
            frame["threshold_name"].eq("Q10")
            & frame["target"].eq("onset")
            & frame["lead_days"].isin(LEADS)
            & frame["model"].isin([
                "hydrograph_analog",
                "hydrograph_marginal_shuffle",
                "seasonal_climatology_path",
                "constant_persistence_path",
            ])
        ]
        if not selected.empty:
            frames.append(selected)
    if not frames:
        raise RuntimeError("No archived validation metrics were found")
    output = pd.concat(frames, ignore_index=True)
    output["GAGE_ID"] = normalize_gage(output["GAGE_ID"])
    return output.sort_values(["GAGE_ID", "lead_days", "model"]).reset_index(drop=True)


def validate_predictions(paths: list[Path], metrics: pd.DataFrame) -> dict:
    """Run release-level structural and score-reproduction checks."""
    reports = {}
    required = [
        "GAGE_ID", "issue_date", "valid_end_date", "lead_days",
        "lead_is_scorable", "observed_event", "forecast_event_probability",
        "observed_endpoint_low", "forecast_endpoint_probability",
        "observed_onset_day_censored", "forecast_onset_p05",
        "forecast_onset_p50", "forecast_onset_p95",
        "observed_duration_days", "forecast_duration_p05",
        "forecast_duration_p50", "forecast_duration_p95",
        "observed_deficit_mm", "forecast_deficit_mm_p05",
        "forecast_deficit_mm_p50", "forecast_deficit_mm_p95",
        "observed_minimum_flow_mm_day", "forecast_minimum_flow_mm_day_p05",
        "forecast_minimum_flow_mm_day_p50", "forecast_minimum_flow_mm_day_p95",
    ]
    critical = [
        "GAGE_ID", "issue_date", "valid_end_date", "lead_days",
        "lead_is_scorable", "observed_event", "forecast_event_probability",
        "observed_endpoint_low", "forecast_endpoint_probability",
        "observed_onset_day_censored", "observed_duration_days",
        "observed_deficit_mm", "observed_minimum_flow_mm_day",
    ]
    triplets = [
        ("forecast_onset_p05", "forecast_onset_p50", "forecast_onset_p95"),
        ("forecast_duration_p05", "forecast_duration_p50", "forecast_duration_p95"),
        ("forecast_deficit_mm_p05", "forecast_deficit_mm_p50", "forecast_deficit_mm_p95"),
        (
            "forecast_minimum_flow_mm_day_p05",
            "forecast_minimum_flow_mm_day_p50",
            "forecast_minimum_flow_mm_day_p95",
        ),
    ]
    all_passed = True
    for path in paths:
        frame = pd.read_parquet(path, columns=required)
        lead = int(frame["lead_days"].iloc[0])
        issue = pd.to_datetime(frame["issue_date"])
        valid = pd.to_datetime(frame["valid_end_date"])
        duplicate_rows = int(frame.duplicated(["GAGE_ID", "issue_date", "lead_days"]).sum())
        lead_value_errors = int((frame["lead_days"] != lead).sum())
        valid_date_errors = int(((valid - issue).dt.days != lead).sum())
        probability_errors = int(
            (~frame["forecast_event_probability"].between(0, 1)).sum()
            + (~frame["forecast_endpoint_probability"].between(0, 1)).sum()
        )
        critical_nulls = int(frame[critical].isna().sum().sum())
        quantile_order_errors = 0
        for lower, median, upper in triplets:
            quantile_order_errors += int(
                ((frame[lower] > frame[median]) | (frame[median] > frame[upper])).sum()
            )

        scorable = frame["lead_is_scorable"].fillna(False).astype(bool)
        scored = frame.loc[scorable]
        released_brier = float(
            ((scored["forecast_event_probability"] - scored["observed_event"].astype(float)) ** 2).mean()
        )
        archived = metrics[
            metrics["lead_days"].eq(lead)
            & metrics["model"].eq("hydrograph_analog")
        ]
        archived_brier = float(
            (archived["event_brier"] * archived["n"]).sum() / archived["n"].sum()
        )
        brier_difference = abs(released_brier - archived_brier)
        passed = (
            duplicate_rows == 0
            and lead_value_errors == 0
            and valid_date_errors == 0
            and probability_errors == 0
            and critical_nulls == 0
            and quantile_order_errors == 0
            and brier_difference <= BRIER_REPRODUCTION_TOLERANCE
        )
        all_passed = all_passed and passed
        reports[str(lead)] = {
            "rows": int(len(frame)),
            "basins": int(frame["GAGE_ID"].nunique()),
            "scorable_rows": int(scorable.sum()),
            "scorable_basins": int(scored["GAGE_ID"].nunique()),
            "duplicate_issue_keys": duplicate_rows,
            "lead_value_errors": lead_value_errors,
            "valid_end_date_errors": valid_date_errors,
            "probability_range_errors": probability_errors,
            "critical_null_values": critical_nulls,
            "quantile_order_errors": quantile_order_errors,
            "released_occurrence_brier": released_brier,
            "archived_occurrence_brier": archived_brier,
            "absolute_brier_difference": brier_difference,
            "passed": passed,
        }
    return {
        "data_version": VERSION,
        "all_checks_passed": all_passed,
        "brier_reproduction_tolerance": BRIER_REPRODUCTION_TOLERANCE,
        "checks": reports,
    }


def readme_text(counts: dict[int, int], basins: int) -> str:
    count_lines = "\n".join(f"- {lead} days: {counts[lead]:,} forecasts" for lead in LEADS)
    return f"""# Temporal Coherence Low-Flow Forecast Dataset, version {VERSION}

## Contents

This package contains retrospective probabilistic Q10 low-flow forecasts for {basins:,} U.S. catchments. Forecasts were initialized weekly during 2016--2025 from observed dry states and were generated with the frozen 101-member hydrograph-state analog described in the accompanying manuscript.

{count_lines}

The four `forecast_predictions_q10_lead_*.parquet` files contain forecast-level predictions and observations. `basin_validation_metrics.csv.gz` contains the archived basin-level proper scores for the coherent forecast and three benchmarks. `gage_metadata.csv` contains station metadata. `forecast_predictions_sample.csv.gz` is a small human-readable extract; it is not a separate analysis product.

## Scientific scope

These are hindcasts from a controlled research experiment, not operational forecasts. Training, scaling, and thresholds use data ending no later than 2015; evaluation uses 2016--2025. Forecast initialization dates are Wednesdays that meet the frozen observed dry-state screen. No future meteorological observation is used to select cases or construct the eligible forecast.

The Q10 threshold is basin specific and equals the 10th percentile of normalized daily flow through 2015. Forecast occurrence means that at least one day in the stated window is at or below Q10. Onset, duration, cumulative deficit, and minimum flow are computed member by member so that temporal coherence is retained.

## Recommended use

Use `lead_is_scorable = true` when reproducing paper-level verification because the frozen design required at least 10 held-out events and 10 non-events for a basin-lead comparison. Preserve `GAGE_ID` as text. Join forecasts across leads with `GAGE_ID` and `issue_date`.

## Provenance

Daily discharge: U.S. Geological Survey Water Data for the Nation, parameter 00060, daily-value statistic 00003. Basin attributes and reference status: GAGES-II. Meteorological initialization variables: GridMET. Code and the exact archived score files are retained with the project.

## License and citation

A repository DOI and reuse license must be selected by the authors at deposit. CC BY 4.0 is recommended for the derived forecast tables. Cite the accompanying manuscript and dataset DOI once assigned.
"""


def citation_text() -> str:
    return f"""cff-version: 1.2.0
message: "Please cite this dataset and its accompanying article."
title: "Temporal Coherence Low-Flow Forecast Dataset"
type: dataset
authors:
  - family-names: "Amanambu"
    given-names: "Amobichukwu C."
version: "{VERSION}"
date-released: "{datetime.now(timezone.utc).date().isoformat()}"
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    files = sorted(WORK_ROOT.glob("*_forecast.parquet"))
    if not files:
        raise RuntimeError("No Stage 30 basin forecast files were found")

    counts: dict[int, int] = {}
    outputs: list[Path] = []
    for lead in LEADS:
        destination = DATA_ROOT / f"forecast_predictions_q10_lead_{lead:03d}.parquet"
        if destination.exists() and not args.force:
            table = pq.ParquetFile(destination)
            counts[lead] = int(table.metadata.num_rows)
        else:
            counts[lead] = assemble_lead(files, lead, destination)
        outputs.append(destination)
        print(f"Lead {lead}: {counts[lead]:,} rows", flush=True)

    metrics = archived_validation_metrics()
    metrics_path = DATA_ROOT / "basin_validation_metrics.csv.gz"
    atomic_csv(metrics, metrics_path, compression="gzip")
    outputs.append(metrics_path)

    attributes = pd.read_csv(
        ATTRIBUTES,
        usecols=[
            "GAGE_ID", "STANAME", "STATE", "LAT_GAGE", "LNG_GAGE",
            "DRAIN_SQKM", "AGGECOREGION", "quality_tier", "is_reference",
            "record_years", "BFI_AVE", "aridity", "snow_fraction",
        ],
        dtype={"GAGE_ID": str},
    )
    attributes["GAGE_ID"] = normalize_gage(attributes["GAGE_ID"])
    release_gages = set()
    for path in files:
        release_gages.add(path.name.split("_", 1)[0])
    attributes = attributes[attributes["GAGE_ID"].isin(release_gages)]
    attributes = attributes.sort_values("GAGE_ID").reset_index(drop=True)
    metadata_path = DATA_ROOT / "gage_metadata.csv"
    atomic_csv(attributes, metadata_path)
    outputs.append(metadata_path)

    dictionary = pd.DataFrame(DICTIONARY, columns=["field", "units_or_type", "description"])
    dictionary_path = RELEASE_ROOT / "DATA_DICTIONARY.csv"
    atomic_csv(dictionary, dictionary_path)
    outputs.append(dictionary_path)

    sample_frames = []
    for lead in LEADS:
        path = DATA_ROOT / f"forecast_predictions_q10_lead_{lead:03d}.parquet"
        sample_frames.append(pd.read_parquet(path).head(250))
    sample = pd.concat(sample_frames, ignore_index=True)
    sample_path = DATA_ROOT / "forecast_predictions_sample.csv.gz"
    atomic_csv(sample, sample_path, compression="gzip")
    outputs.append(sample_path)

    readme_path = RELEASE_ROOT / "README.md"
    readme_path.write_text(readme_text(counts, len(release_gages)), encoding="utf-8")
    outputs.append(readme_path)

    citation_path = RELEASE_ROOT / "CITATION.cff"
    citation_path.write_text(citation_text(), encoding="utf-8")
    outputs.append(citation_path)

    validation = validate_predictions(outputs[:4], metrics)
    validation_path = RELEASE_ROOT / "VALIDATION_REPORT.json"
    validation_path.write_text(
        json.dumps(validation, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    outputs.append(validation_path)
    if not validation["all_checks_passed"]:
        raise RuntimeError(f"Release validation failed; inspect {validation_path}")

    manifest_rows = []
    for path in outputs:
        manifest_rows.append({
            "relative_path": str(path.relative_to(RELEASE_ROOT)),
            "bytes": int(path.stat().st_size),
            "sha256": sha256(path),
        })
    manifest = pd.DataFrame(manifest_rows)
    manifest_path = RELEASE_ROOT / "MANIFEST.csv"
    atomic_csv(manifest, manifest_path)

    receipt = {
        "stage": 31,
        "status": "complete",
        "data_version": VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "basins": int(len(release_gages)),
        "rows_by_lead": {str(k): int(v) for k, v in counts.items()},
        "prediction_files": [str(path.relative_to(RELEASE_ROOT)) for path in outputs[:4]],
        "sha256_manifest": str(manifest_path.relative_to(RELEASE_ROOT)),
    }
    (RELEASE_ROOT / "STAGE31_SUCCESS.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(receipt, indent=2), flush=True)


if __name__ == "__main__":
    main()
