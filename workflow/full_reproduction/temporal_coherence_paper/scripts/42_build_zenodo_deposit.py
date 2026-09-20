#!/usr/bin/env python3
"""Assemble and validate the Zenodo-ready supporting-data deposit.

Run after Stage 41 completes:

    python temporal_coherence_paper/scripts/42_build_zenodo_deposit.py

The resulting directory contains the exact files to upload individually to
Zenodo.  Raw USGS, GAGES-II, and GridMET inputs are not redistributed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import zipfile
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PAPER_ROOT = PROJECT_ROOT / "temporal_coherence_paper"
DEFAULT_WORK_ROOT = PAPER_ROOT / "zenodo_deposit" / "work_v1"
DEFAULT_RELEASE_ROOT = PAPER_ROOT / "zenodo_deposit" / "v1.0.0_upload"
GITHUB_ROOT = PROJECT_ROOT.parents[1]
GITHUB_URL = "https://github.com/amobiamanambu/temporal-dependence-low-flow"
FINAL_ROOT = PAPER_ROOT / "dependence_reconstruction" / "full"
FINAL_METRICS = FINAL_ROOT / "dependence_reconstruction_metrics.csv.gz"
CALIBRATION = PAPER_ROOT / "temporal_validation" / "basin_horizon_calibration.csv.gz"
ECOREGION_CALIBRATION = PAPER_ROOT / "temporal_validation" / "ecoregion_horizon_calibration.csv"
BASIN_ATTRIBUTES = PROJECT_ROOT / "continental_run" / "06_basin_index" / "basin_index.csv"
EXTENSION_FAILURES = (
    PROJECT_ROOT
    / "lowflow_forecast_benchmark"
    / "results"
    / "10_extension_120"
    / "extension_120_failures.csv"
)
CONFIGURATION = PROJECT_ROOT / "lowflow_forecast_benchmark" / "config.json"
RELEASE_LEADS = (30, 45, 60, 90, 105, 120)
VERSION = "1.0.0"
BRIER_TOLERANCE = 1e-6
TITLE = (
    "Temporal Dependence Improves Probabilistic Forecasts "
    "of Low-Flow Events Across CONUS Catchments"
)
DOI = "10.5281/zenodo.22770792"
DOI_URL = f"https://doi.org/{DOI}"
PRIVATE_PATH_TOKENS = {
    "accepted_figure_revisions",
    "figure_redesign_review",
    "horizon_skill_candidate",
    "manuscript",
    "private_authoring",
    "submission",
}
PRIVATE_AUTHORING_MARKERS = (
    "wrr.add_body(",
    "from docx import Document",
    "import docx",
)


FORECAST_DICTIONARY = [
    ("data_version", "string", "Version of this supporting-data release."),
    ("experiment", "string", "Frozen experiment and common-support definition."),
    ("GAGE_ID", "string", "USGS station identifier stored as text; leading zeros retained."),
    ("initialization_date", "date", "Weekly retrospective forecast initialization date."),
    ("valid_end_date", "date", "Last date in the forecast window."),
    ("lead_days", "integer days", "Forecast-window length: 30, 45, 60, 90, 105, or 120 days."),
    ("threshold_name", "string", "Low-flow threshold; Q10 is the training-period 10th percentile."),
    ("target", "string", "Onset experiment initialized above the basin Q10 threshold."),
    ("model", "string", "Frozen basin-specific 101-member hydrograph-state analog."),
    ("spatial_group", "string", "Aggregated GAGES-II ecoregion."),
    ("quality_tier", "string", "Continental workflow record-quality tier."),
    ("is_reference", "boolean", "GAGES-II reference-basin indicator."),
    ("record_years", "years", "Valid discharge record length during 1980--2025."),
    ("initial_flow_mm_day", "mm d-1", "Observed runoff depth at initialization."),
    ("threshold_q10_mm_day", "mm d-1", "Basin Q10 threshold estimated through 2015."),
    ("dry_spell_age_days", "days", "Consecutive screened dry days at initialization."),
    ("ensemble_members", "count", "Number of forecast trajectories."),
    ("lead_is_scorable", "boolean", "At least 10 held-out events and 10 non-events at this lead."),
    ("observed_event", "boolean", "Q10 crossed at least once in the forecast window."),
    ("forecast_event_probability", "probability", "Fraction of members with at least one Q10 crossing."),
    ("observed_endpoint_low", "boolean", "Observed flow at or below Q10 on the final day."),
    ("forecast_endpoint_probability", "probability", "Fraction of members at or below Q10 on the final day."),
    ("observed_onset_day", "days", "First observed Q10 crossing; missing if no crossing occurred."),
    ("observed_onset_day_censored", "days", "First crossing, or lead+1 if no crossing occurred."),
    ("forecast_onset_p05", "days", "5th percentile of censored ensemble onset day."),
    ("forecast_onset_p50", "days", "Median censored ensemble onset day."),
    ("forecast_onset_p95", "days", "95th percentile of censored ensemble onset day."),
    ("observed_duration_days", "days", "Observed total low-flow-day count within the window; historical field name retained."),
    ("forecast_duration_p05", "days", "5th percentile of ensemble total low-flow-day count; historical field name retained."),
    ("forecast_duration_p50", "days", "Median ensemble total low-flow-day count; historical field name retained."),
    ("forecast_duration_p95", "days", "95th percentile of ensemble total low-flow-day count; historical field name retained."),
    ("observed_deficit_mm", "mm", "Observed cumulative runoff-depth deficit below Q10."),
    ("forecast_deficit_mm_p05", "mm", "5th percentile of ensemble cumulative deficit."),
    ("forecast_deficit_mm_p50", "mm", "Median ensemble cumulative deficit."),
    ("forecast_deficit_mm_p95", "mm", "95th percentile of ensemble cumulative deficit."),
    ("observed_minimum_flow_mm_day", "mm d-1", "Observed minimum runoff depth in the window."),
    ("forecast_minimum_flow_mm_day_p05", "mm d-1", "5th percentile of ensemble minimum flow."),
    ("forecast_minimum_flow_mm_day_p50", "mm d-1", "Median ensemble minimum flow."),
    ("forecast_minimum_flow_mm_day_p95", "mm d-1", "95th percentile of ensemble minimum flow."),
]


ANALYSIS_FILES = [
    "reconstruction_comparisons.csv",
    "reconstruction_recovery_fraction.csv",
    "archived_stratified_basin_intervals.csv",
    "uncertainty_scheme_sensitivity.csv",
    "leave_one_water_year_out.csv",
    "sample_accounting.csv",
    "development_architecture_replication.csv",
    "STAGE36_SUCCESS.json",
    "STAGE39_SUCCESS.json",
]


FIGURE_SOURCE_DIRS = [PROJECT_ROOT / "figures" / "source_tables"]


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


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def build_zip(
    destination: Path,
    sources: list[tuple[Path, str]],
    excluded_names: set[str] | None = None,
    excluded_parts: set[str] | None = None,
) -> None:
    """Create a deterministic ZIP from files or directory trees."""
    excluded_names = excluded_names or set()
    excluded_parts = excluded_parts or set()
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for source, archive_root in sources:
            if source.is_dir():
                for path in sorted(source.rglob("*")):
                    if not path.is_file():
                        continue
                    if path.name in excluded_names:
                        continue
                    if any(part in excluded_parts for part in path.parts):
                        continue
                    if any(part in {".git", "__pycache__", ".cache", ".DS_Store"} for part in path.parts):
                        continue
                    relative = Path(archive_root) / path.relative_to(source)
                    archive.write(path, relative.as_posix())
            elif source.is_file():
                archive.write(source, (Path(archive_root) / source.name).as_posix())
    temporary.replace(destination)


def validate_public_source_snapshot(path: Path) -> None:
    """Fail if manuscript-authoring material entered the public code snapshot."""
    problems: list[str] = []
    with zipfile.ZipFile(path) as archive:
        for member in archive.infolist():
            member_path = Path(member.filename)
            if member.is_dir():
                continue
            if any(token in member.filename.lower() for token in PRIVATE_PATH_TOKENS):
                problems.append(member.filename)
                continue
            if member_path.suffix.lower() in {".doc", ".docx", ".odt"}:
                problems.append(member.filename)
                continue
            if (
                member_path.suffix.lower() == ".py"
                and member_path.name != Path(__file__).name
            ):
                content = archive.read(member).decode("utf-8", errors="replace")
                if any(marker in content for marker in PRIVATE_AUTHORING_MARKERS):
                    problems.append(member.filename)
    if problems:
        formatted = "\n".join(f"- {item}" for item in sorted(set(problems)))
        raise RuntimeError(
            "Private manuscript-authoring material found in source snapshot:\n"
            f"{formatted}"
        )


def assemble_lead(files: list[Path], lead: int, destination: Path) -> int:
    writer: pq.ParquetWriter | None = None
    rows = 0
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    try:
        for path in files:
            table = pq.read_table(path, filters=[("lead_days", "=", int(lead))])
            if table.num_rows == 0:
                continue
            if writer is None:
                writer = pq.ParquetWriter(
                    temporary, table.schema, compression="zstd", use_dictionary=True
                )
            writer.write_table(table)
            rows += table.num_rows
    finally:
        if writer is not None:
            writer.close()
    if rows == 0:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"No Stage 41 records were found for lead {lead}")
    temporary.replace(destination)
    return rows


def final_archived_brier() -> pd.DataFrame:
    metrics = pd.read_csv(FINAL_METRICS, dtype={"GAGE_ID": str})
    selected = metrics[
        metrics["aggregation"].eq("basin")
        & metrics["model"].eq("hydrograph_analog")
        & metrics["score"].eq("event_brier")
        & metrics["lead_days"].isin(RELEASE_LEADS)
    ][["GAGE_ID", "lead_days", "n", "events", "score_value"]].copy()
    selected["GAGE_ID"] = selected["GAGE_ID"].astype(str).str.zfill(8)
    return selected.rename(columns={"score_value": "archived_event_brier"})


def validate_forecasts(paths: list[Path]) -> dict:
    required = [
        "GAGE_ID", "initialization_date", "valid_end_date", "lead_days",
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
        "GAGE_ID", "initialization_date", "valid_end_date", "lead_days",
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
    archived = final_archived_brier()
    reports: dict[str, dict] = {}
    key_reference: pd.MultiIndex | None = None
    all_passed = True

    for path in paths:
        frame = pd.read_parquet(path, columns=required)
        frame["GAGE_ID"] = frame["GAGE_ID"].astype(str).str.zfill(8)
        leads = frame["lead_days"].dropna().unique()
        if len(leads) != 1:
            raise RuntimeError(f"Mixed leads in {path}")
        lead = int(leads[0])
        initialization = pd.to_datetime(frame["initialization_date"])
        valid = pd.to_datetime(frame["valid_end_date"])
        keys = pd.MultiIndex.from_frame(frame[["GAGE_ID", "initialization_date"]])
        keys = keys.sort_values()
        if key_reference is None:
            key_reference = keys
        common_support_errors = int(
            len(keys.symmetric_difference(key_reference))
        )

        duplicate_keys = int(frame.duplicated(["GAGE_ID", "initialization_date", "lead_days"]).sum())
        date_errors = int(((valid - initialization).dt.days != lead).sum())
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

        scored = frame[frame["lead_is_scorable"].fillna(False).astype(bool)].copy()
        scored["squared_error"] = (
            scored["forecast_event_probability"]
            - scored["observed_event"].astype(float)
        ) ** 2
        reproduced = scored.groupby("GAGE_ID", as_index=False).agg(
            released_n=("observed_event", "size"),
            released_events=("observed_event", "sum"),
            released_event_brier=("squared_error", "mean"),
        )
        release_ids = set(frame["GAGE_ID"])
        expected = archived[
            archived["lead_days"].eq(lead)
            & archived["GAGE_ID"].isin(release_ids)
        ].copy()
        comparison = expected.merge(reproduced, on="GAGE_ID", how="outer", indicator=True)
        missing_score_rows = int((comparison["_merge"] != "both").sum())
        complete = comparison[comparison["_merge"].eq("both")]
        count_mismatches = int(
            (
                (complete["n"].astype(int) != complete["released_n"].astype(int))
                | (complete["events"].astype(int) != complete["released_events"].astype(int))
            ).sum()
        )
        max_brier_error = float(
            np.max(np.abs(
                complete["archived_event_brier"] - complete["released_event_brier"]
            )) if len(complete) else 0.0
        )
        passed = bool(
            duplicate_keys == 0
            and date_errors == 0
            and probability_errors == 0
            and critical_nulls == 0
            and quantile_order_errors == 0
            and common_support_errors == 0
            and missing_score_rows == 0
            and count_mismatches == 0
            and max_brier_error <= BRIER_TOLERANCE
        )
        all_passed = all_passed and passed
        reports[str(lead)] = {
            "rows": int(len(frame)),
            "basins": int(frame["GAGE_ID"].nunique()),
            "scorable_rows": int(len(scored)),
            "scorable_basins": int(scored["GAGE_ID"].nunique()),
            "duplicate_initialization_keys": duplicate_keys,
            "valid_end_date_errors": date_errors,
            "probability_range_errors": probability_errors,
            "critical_null_values": critical_nulls,
            "quantile_order_errors": quantile_order_errors,
            "common_support_key_errors": common_support_errors,
            "missing_or_extra_archived_score_rows": missing_score_rows,
            "score_count_mismatches": count_mismatches,
            "maximum_absolute_brier_error": max_brier_error,
            "passed": passed,
        }
    return {
        "data_version": VERSION,
        "validated_utc": datetime.now(timezone.utc).isoformat(),
        "all_checks_passed": all_passed,
        "brier_reproduction_tolerance": BRIER_TOLERANCE,
        "checks": reports,
    }


def readme_text(
    counts: dict[int, int], basins: int, scorable_basins: dict[int, int]
) -> str:
    count_lines = "\n".join(
        f"- {lead} days: {counts[lead]:,} initialization records"
        for lead in RELEASE_LEADS
    )
    return f"""# {TITLE}

Version {VERSION}

DOI: {DOI_URL}

## Overview

This deposit contains the processed data used to evaluate temporal dependence
in retrospective probabilistic forecasts of Q10 low-flow events. The final
forecast cohort contains {basins:,} CONUS catchments and six forecast windows:

{count_lines}

The lead-specific Parquet files contain the observed outcomes and intact
hydrograph-analog forecasts for each initialization. `final_score_metrics.csv.gz`
contains basin- and water-year-level proper scores for the intact trajectories,
independent daily ordering, and seasonal rank reconstruction. The remaining
analysis tables contain the comparisons, uncertainty analyses, sample
accounting, temporal validation, and exact source data used for manuscript
figures.

## Analysis-population accounting

The parent quality-control workflow retained 5,227 Tier-1/Tier-2 basins. A
192-basin development sample was excluded before 5,035 basins entered
independent confirmation. The original 1--90-day analysis produced at least
one valid score for 3,835 basins, and the separately processed extension
produced at least one scorable threshold--target combination for 3,733 basins.
These are parallel results from the same 5,035-basin confirmation population:
3,731 basins occurred in both sets, 104 only in the original results, and two
only in the extension. The extension failures comprised 706 basins with too
few training initializations, 351 with too few evaluation initializations, and
245 without the required event/non-event balance.

Among the 3,733 extension-eligible basins, 3,402 had an estimable score for the
primary Q10-onset target at one or more windows and form this forecast-level
release. The other 331 basins supported a different threshold or target but
not primary Q10 onset.

The released files contain forecast records for all 3,402 basins at every
window. Manuscript-level scoring additionally required at least 10 held-out
events and 10 non-events within a basin and window. The resulting scorable
basin counts were {scorable_basins[30]:,}, {scorable_basins[45]:,},
{scorable_basins[60]:,}, {scorable_basins[90]:,},
{scorable_basins[105]:,}, and {scorable_basins[120]:,} at 30, 45, 60, 90, 105,
and 120 days, respectively. The `lead_is_scorable` field identifies those
records.

## Scientific scope

These are retrospective hindcasts, not operational USGS forecasts. The model
was trained using data ending in 2015 and evaluated during 2016--2025.
Initialization dates were selected from information available on the
initialization date. No future meteorological observations were used to select
cases or construct an eligible forecast.

The basin-specific Q10 threshold is the lower 10th percentile of normalized
daily flow in the training record (equivalent to Q90 in exceedance-probability
notation). An occurrence event means that streamflow crossed Q10 at least once
during the forecast window. Onset, total low-flow-day count, cumulative deficit,
and minimum flow were calculated member by member. The archived fields containing
the word `duration` store total days at or below Q10 and do not isolate one
continuous drought episode. All six released
forecast files use a common set of initialization dates with complete 120-day
observed paths.

Occurrence and final-day state were evaluated with the Brier score. Joint
onset or no event and conditional onset used a normalized ranked probability
score (RPS); low-flow-day count, deficit, and minimum flow used ensemble CRPS. For
backward compatibility, the archived score table retains the internal name
`timing_crps` for the joint-onset RPS. The name is historical; the stored score
is the RPS defined in the manuscript.

Full 101-member trajectories are not included because their continental size
is impractical. The archived final score table preserves results for all three
trajectory-ordering configurations, and the open-source workflow reconstructs
the member trajectories from the cited public inputs.

## Files

See `FILE_CATALOG.csv` for a file-by-file description and
`DATA_DICTIONARY.csv` for the forecast columns. `VALIDATION_REPORT.json`
documents structural checks and reproduction of the archived basin-level
occurrence Brier scores. `MANIFEST.csv` and `SHA256SUMS.txt` provide file sizes
and cryptographic checksums.

## Data sources

Raw inputs are not redistributed. Daily discharge came from USGS Water Data
for the Nation (parameter 00060, statistic 00003). Basin attributes and
reference status came from GAGES-II. Meteorological initialization variables
came from GridMET. Persistent source identifiers and access information are
provided in `PROVENANCE.md`.

## Code

The executable workflow is available at {GITHUB_URL}. A frozen source-code
snapshot is also included in this deposit.

## Reuse

The derived data are released under Creative Commons Attribution 4.0
International (CC BY 4.0). The source-code snapshot retains its MIT license.
Preserve `GAGE_ID` as text so leading zeros are not lost. Use
`lead_is_scorable = true` for manuscript-level score reproduction.
"""


def provenance_text() -> str:
    return """# Provenance

## U.S. Geological Survey daily discharge

- Product: USGS Water Data for the Nation, daily values.
- Parameter: 00060 (discharge).
- Statistic: 00003 (daily mean).
- Analysis period: 1980-01-01 through 2025-12-31.
- Access: https://waterdata.usgs.gov/nwis/dv/
- Web services: https://waterservices.usgs.gov/

## GAGES-II

- Product: Geospatial Attributes of Gages for Evaluating Streamflow, version II.
- Use: basin locations, attributes, ecoregions, and reference status.
- Access: https://water.usgs.gov/GIS/metadata/usgswrd/XML/gagesII_Sept2011.xml

## GridMET

- Product: Gridded Surface Meteorological Dataset.
- Use: precipitation, reference evapotranspiration, and temperature predictors available at initialization.
- Access: https://www.climatologylab.org/gridmet.html
- Repository: https://www.northwestknowledge.net/metdata/data/

Raw source data are not included in this deposit. Users should obtain current
copies from the authoritative providers and follow the workflow archived with
this release.
"""


def citation_text() -> str:
    return f"""cff-version: 1.2.0
message: "Please cite this dataset and the accompanying article."
title: "{TITLE}"
type: dataset
authors:
  - family-names: "Amanambu"
    given-names: "Amobichukwu C."
    affiliation: "The University of Alabama"
version: "{VERSION}"
date-released: "{date.today().isoformat()}"
doi: "{DOI}"
url: "{DOI_URL}"
repository-code: "{GITHUB_URL}"
license: CC-BY-4.0
keywords:
  - low flow
  - hydrological drought
  - probabilistic forecasting
  - ensemble trajectories
  - temporal dependence
  - streamflow
  - GAGES-II
  - CONUS
"""


def license_text() -> str:
    return """Creative Commons Attribution 4.0 International (CC BY 4.0)

Copyright (c) 2026 Amobichukwu C. Amanambu

You are free to share and adapt the derived data for any purpose, provided
appropriate credit is given, a link to the license is provided, and changes
are indicated. The legal code is available at:

https://creativecommons.org/licenses/by/4.0/legalcode

This license applies to the derived data in this deposit. The source-code ZIP
contains its own MIT license. USGS, GAGES-II, and GridMET source products are
not redistributed and remain subject to their providers' terms.
"""


def metadata_text() -> str:
    return f"""# Zenodo metadata to enter

## Upload type

Dataset

## Title

{TITLE}

## Reserved DOI

{DOI}

## Publication date

{date.today().isoformat()}

## Creator

- Family name: Amanambu
- Given name: Amobichukwu C.
- Affiliation: The University of Alabama
- ORCID: add only if the author has verified the identifier

## Description

Processed data for a continental retrospective experiment testing
how temporal dependence affects probabilistic forecasts of low-flow
occurrence, onset, low-flow-day count, and deficit. The release contains
initialization-level Q10
forecasts at 30, 45, 60, 90, 105, and 120 days for the final 3,402-basin Q10
forecast cohort, basin- and water-year-level scores for three trajectory-ordering
configurations, uncertainty analyses, station metadata, manuscript source
tables, and a frozen code snapshot. Training ended in 2015 and evaluation used
2016--2025. These products are research hindcasts and are not operational USGS
forecasts. Raw USGS, GAGES-II, and GridMET inputs are cited but not
redistributed.

## Keywords

low flow; hydrological drought; probabilistic forecasting; ensemble
trajectories; temporal dependence; streamflow; GAGES-II; CONUS

## License

Creative Commons Attribution 4.0 International

## Related identifier

- URL: {GITHUB_URL}
- Relation: Is supplemented by this upload / software source

After the article receives a DOI, add it as `Is supplement to` in a new Zenodo
metadata version. Do not invent or enter an article DOI before it exists.
"""


def upload_instructions_text() -> str:
    return f"""# Manual Zenodo upload checklist

1. Sign in at https://zenodo.org/ and open the existing unpublished draft.
2. Confirm that its reserved DOI is **{DOI}**. Do not request a second DOI and
   do not delete the draft.
3. Select **Upload files** and upload all 26 files in this directory. Do not
   upload the separate `work_v1` directory. The complete deposit is about
   348 MB, below Zenodo's current limit of 100 files and 50 GB per upload.
4. Select resource type **Dataset** and set file visibility to **Public**.
5. Copy the title, creator, description, keywords, license, and related GitHub
   identifier from `ZENODO_METADATA.md`.
6. Select **Save draft**, resolve any validation messages, and then use
   **Preview** to inspect the unpublished record.
7. Confirm that the DOI appears in the manuscript Data Availability statement,
   GitHub README, GitHub `CITATION.cff`, and this deposit's `CITATION.cff`.
8. Confirm that the Zenodo draft lists every file in `FILE_CATALOG.csv` and
   that the displayed file sizes agree with `MANIFEST.csv`.
9. Publish the Zenodo record only after all metadata and file checks pass.
10. Download one deposited Parquet file and verify its SHA-256 value against
   `SHA256SUMS.txt`.

Zenodo permits metadata edits after publication, but file additions,
replacements, or removals by the depositor are limited to 45 days. Check the
files carefully before selecting **Publish**.

Do not upload raw USGS discharge batches, GridMET NetCDF files, shapefiles,
temporary basin files, caches, or unpublished manuscript drafts.
"""


def file_catalog(rows_by_lead: dict[int, int]) -> pd.DataFrame:
    rows = []
    for lead in RELEASE_LEADS:
        rows.append({
            "file": f"forecast_predictions_q10_lead_{lead:03d}.parquet",
            "category": "forecast-level data",
            "description": (
                f"Observed outcomes and intact probabilistic forecasts for the {lead}-day window; "
                f"{rows_by_lead[lead]:,} initialization records."
            ),
        })
    rows.extend([
        {"file": "final_score_metrics.csv.gz", "category": "analysis data", "description": "Basin- and water-year-level proper scores for intact, independently ordered, and seasonally reconstructed trajectories. The legacy timing_crps field name stores the normalized onset RPS."},
        {"file": "basin_horizon_calibration.csv.gz", "category": "analysis data", "description": "Basin-level forecast probability, observed frequency, bias, and Brier score by forecast window."},
        {"file": "ecoregion_horizon_calibration.csv", "category": "analysis data", "description": "Ecoregion calibration summaries and bootstrap intervals by forecast window."},
        {"file": "gage_metadata.csv", "category": "metadata", "description": "Station coordinates, basin attributes, quality tier, reference status, and record length."},
        {"file": "forecast_predictions_sample.csv.gz", "category": "example", "description": "Small CSV extract illustrating the Parquet schema; not a separate analysis product."},
        {"file": "analysis_products.zip", "category": "analysis data", "description": "Final comparison, uncertainty, temporal sensitivity, sample-accounting, and exclusion tables."},
        {"file": "manuscript_figure_source_tables.zip", "category": "figure data", "description": "Tabular source data retained by the final figure-generation workflow."},
        {"file": "source_code_snapshot.zip", "category": "software", "description": "Frozen snapshot of the public analysis, validation, data-release, and figure repository; manuscript-authoring utilities are excluded; code is MIT licensed."},
        {"file": "analysis_configuration.json", "category": "configuration", "description": "Frozen low-flow forecasting configuration used by the workflow."},
        {"file": "DATA_DICTIONARY.csv", "category": "documentation", "description": "Field names, units or types, and definitions for forecast tables."},
        {"file": "PROVENANCE.md", "category": "documentation", "description": "Authoritative source products, access links, and non-redistribution statement."},
        {"file": "README.md", "category": "documentation", "description": "Dataset overview, scope, contents, use, and limitations."},
        {"file": "LICENSE.txt", "category": "license", "description": "CC BY 4.0 notice for derived data and license boundary for source inputs and code."},
        {"file": "CITATION.cff", "category": "citation", "description": "Machine-readable dataset citation metadata."},
        {"file": "ZENODO_METADATA.md", "category": "deposit metadata", "description": "Copy-ready metadata for the Zenodo web form."},
        {"file": "UPLOAD_INSTRUCTIONS.md", "category": "deposit documentation", "description": "Manual deposit and verification sequence."},
        {"file": "VALIDATION_REPORT.json", "category": "quality assurance", "description": "Structural, common-support, and archived-score reproduction checks."},
        {"file": "MANIFEST.csv", "category": "quality assurance", "description": "File sizes and SHA-256 checksums for the deposit."},
        {"file": "SHA256SUMS.txt", "category": "quality assurance", "description": "Standard checksum list for download verification."},
        {"file": "FILE_CATALOG.csv", "category": "documentation", "description": "This file-by-file description."},
    ])
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-root", type=Path, default=DEFAULT_WORK_ROOT)
    parser.add_argument("--release-root", type=Path, default=DEFAULT_RELEASE_ROOT)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    work_root = args.work_root.resolve()
    release_root = args.release_root.resolve()
    by_basin = work_root / "by_basin"
    receipt_path = work_root / "STAGE41_SUCCESS.json"
    if not receipt_path.exists():
        raise FileNotFoundError(f"Stage 41 receipt is missing: {receipt_path}")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("status") != "complete" or receipt.get("leads_days") != list(RELEASE_LEADS):
        raise RuntimeError("Stage 41 receipt does not describe the final six-window export")
    files = sorted(by_basin.glob("*_forecast.parquet"))
    if len(files) != int(receipt["completed_basins"]):
        raise RuntimeError("Stage 41 basin-file count does not match its receipt")
    if int(receipt.get("excluded_basins", 0)) != 0:
        raise RuntimeError("Final sample contains Stage 41 exclusions; inspect before release")

    release_root.mkdir(parents=True, exist_ok=True)
    forecast_paths: list[Path] = []
    counts: dict[int, int] = {}
    for lead in RELEASE_LEADS:
        destination = release_root / f"forecast_predictions_q10_lead_{lead:03d}.parquet"
        if destination.exists() and not args.force:
            counts[lead] = int(pq.ParquetFile(destination).metadata.num_rows)
        else:
            counts[lead] = assemble_lead(files, lead, destination)
        forecast_paths.append(destination)
        print(f"Lead {lead}: {counts[lead]:,} records", flush=True)

    release_gages = {
        path.name.removesuffix("_forecast.parquet") for path in files
    }
    attributes = pd.read_csv(
        BASIN_ATTRIBUTES,
        usecols=[
            "GAGE_ID", "STANAME", "STATE", "LAT_GAGE", "LNG_GAGE",
            "DRAIN_SQKM", "AGGECOREGION", "quality_tier", "is_reference",
            "record_years", "BFI_AVE", "aridity", "snow_fraction",
        ],
        dtype={"GAGE_ID": str},
    )
    attributes["GAGE_ID"] = attributes["GAGE_ID"].astype(str).str.zfill(8)
    attributes = attributes[attributes["GAGE_ID"].isin(release_gages)]
    attributes = attributes.sort_values("GAGE_ID").reset_index(drop=True)
    if attributes["GAGE_ID"].nunique() != len(release_gages):
        raise RuntimeError("Gage metadata does not cover every released basin")
    atomic_csv(attributes, release_root / "gage_metadata.csv")

    shutil.copy2(FINAL_METRICS, release_root / "final_score_metrics.csv.gz")
    shutil.copy2(CALIBRATION, release_root / "basin_horizon_calibration.csv.gz")
    shutil.copy2(ECOREGION_CALIBRATION, release_root / "ecoregion_horizon_calibration.csv")
    shutil.copy2(CONFIGURATION, release_root / "analysis_configuration.json")

    sample_frames = []
    for path in forecast_paths:
        frame = pd.read_parquet(path)
        selected_gages = sorted(frame["GAGE_ID"].astype(str).unique())[:5]
        sample_frames.append(
            frame[frame["GAGE_ID"].astype(str).isin(selected_gages)]
            .groupby("GAGE_ID", group_keys=False)
            .head(20)
        )
    sample = pd.concat(sample_frames, ignore_index=True)
    atomic_csv(
        sample,
        release_root / "forecast_predictions_sample.csv.gz",
        compression="gzip",
    )

    analysis_sources = [
        (FINAL_ROOT / name, "analysis")
        for name in ANALYSIS_FILES
        if (FINAL_ROOT / name).exists()
    ]
    if EXTENSION_FAILURES.exists():
        analysis_sources.append((EXTENSION_FAILURES, "analysis"))
    build_zip(release_root / "analysis_products.zip", analysis_sources)

    figure_sources = [
        (directory, directory.parent.name)
        for directory in FIGURE_SOURCE_DIRS
        if directory.exists()
    ]
    build_zip(
        release_root / "manuscript_figure_source_tables.zip",
        figure_sources,
        excluded_names=set(),
    )
    source_snapshot = release_root / "source_code_snapshot.zip"
    build_zip(
        source_snapshot,
        [(GITHUB_ROOT, "temporal-dependence-low-flow")],
        excluded_names=set(),
        excluded_parts={
            "continental_run",
            "results",
            "temporal_validation",
            "dependence_reconstruction",
            "zenodo_deposit",
            "external",
            "derived",
            "output",
            "source_tables",
        },
    )
    validate_public_source_snapshot(source_snapshot)

    dictionary = pd.DataFrame(
        FORECAST_DICTIONARY, columns=["field", "units_or_type", "description"]
    )
    atomic_csv(dictionary, release_root / "DATA_DICTIONARY.csv")
    atomic_csv(file_catalog(counts), release_root / "FILE_CATALOG.csv")
    scorable_basins = (
        final_archived_brier()
        .groupby("lead_days")["GAGE_ID"]
        .nunique()
        .astype(int)
        .to_dict()
    )
    write_text(
        release_root / "README.md",
        readme_text(counts, len(release_gages), scorable_basins),
    )
    write_text(release_root / "PROVENANCE.md", provenance_text())
    write_text(release_root / "CITATION.cff", citation_text())
    write_text(release_root / "LICENSE.txt", license_text())
    write_text(release_root / "ZENODO_METADATA.md", metadata_text())
    write_text(release_root / "UPLOAD_INSTRUCTIONS.md", upload_instructions_text())

    validation = validate_forecasts(forecast_paths)
    validation["stage_41_receipt"] = receipt
    validation["release_basins"] = len(release_gages)
    validation_path = release_root / "VALIDATION_REPORT.json"
    write_text(validation_path, json.dumps(validation, indent=2, sort_keys=True) + "\n")
    if not validation["all_checks_passed"]:
        raise RuntimeError(f"Release validation failed; inspect {validation_path}")

    excluded_from_manifest = {"MANIFEST.csv", "SHA256SUMS.txt"}
    payloads = [
        path for path in sorted(release_root.iterdir())
        if path.is_file() and path.name not in excluded_from_manifest
    ]
    manifest = pd.DataFrame([
        {"file": path.name, "bytes": int(path.stat().st_size), "sha256": sha256(path)}
        for path in payloads
    ])
    atomic_csv(manifest, release_root / "MANIFEST.csv")
    checksum_text = "".join(
        f"{row.sha256}  {row.file}\n" for row in manifest.itertuples(index=False)
    )
    write_text(release_root / "SHA256SUMS.txt", checksum_text)

    final_receipt = {
        "stage": 42,
        "status": "complete",
        "data_version": VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "release_directory": str(release_root),
        "basins": len(release_gages),
        "rows_by_lead": {str(key): int(value) for key, value in counts.items()},
        "files_to_upload": int(len(list(release_root.glob("*")))),
        "all_validation_checks_passed": True,
        "manifest": "MANIFEST.csv",
    }
    write_text(
        PAPER_ROOT / "zenodo_deposit" / "STAGE42_SUCCESS.json",
        json.dumps(final_receipt, indent=2, sort_keys=True) + "\n",
    )
    print(json.dumps(final_receipt, indent=2), flush=True)


if __name__ == "__main__":
    main()
