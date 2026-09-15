# Full continental reproduction

## Before starting

The full run is substantially larger than the example. It requires the raw
USGS, GAGES-II, and GridMET inputs described in `data_sources.md`, hundreds of
gigabytes of free disk space for safe intermediate processing, and an
environment that supports geospatial Python libraries.

Install the extended environment:

```bash
python -m pip install -r requirements-full.txt
```

Run all commands below from `workflow/full_reproduction/`. Do not run later
stages until the preceding stage has written its success receipt.

## A. Retrieve and prepare provider data

```bash
python upstream/01_download_usgs_daily_discharge.py
python upstream/02_download_gridmet_pr_pet.py
python upstream/03_concatenate_gridmet_pr_pet.py
python upstream/04_extract_gridmet_pr_pet.py
python upstream/05_quality_control_discharge.py
```

The concatenation/extraction scripts reproduce the original precipitation and
PET preparation and are memory intensive. The production temperature workflow
below streams annual files in chunks.

## B. Build matched daily catchment records

```bash
python scripts/00_check_environment.py
python scripts/01_download_gridmet_temperature.py
python scripts/02_validate_gridmet_temperature.py
python scripts/03_extract_basin_temperature.py
python scripts/04_build_climate_database.py
python scripts/05_ingest_usgs_discharge.py
python scripts/06_build_basin_index.py
python scripts/07_match_daily_forcings.py
python scripts/08_quality_control_matched_data.py
python scripts/09_build_snowmelt_proxy.py
python scripts/10_build_transition_inventory.py
python scripts/11_select_validation_basins.py
```

Stages 08–10 apply the frozen observation-quality and forcing screens. Their
outputs are the daily basin files consumed by the forecast experiment.

The original development-sample selection also used the archived Stage-18
eligibility table from the parent dry-period analysis. Reproduce that
computational precursor before running the candidate screen:

```bash
python scripts/12_validate_estimators.py --workers 4
python scripts/13_run_continental_irreversibility.py --workers 4
python scripts/14_run_continental_recession.py --workers 4
python scripts/15_assemble_continental_results.py
python scripts/16_test_hypotheses.py
python scripts/17_detect_temporal_change.py
python scripts/18_test_operational_value.py --workers 4
```

Only the resulting eligibility information enters the forecast-model
development workflow; the stochastic-process results are not claims of the
temporal-dependence paper.

## C. Reproduce model screening and 1–90-day confirmation

The complete candidate-screen code is retained because it documents how the
final hydrograph-state analog was selected. Follow the numbered scripts and
the detailed README in `lowflow_forecast_benchmark/`. The final extended-range
confirmation is:

```bash
python lowflow_forecast_benchmark/scripts/16_audit_weather_forecast_archives.py
python lowflow_forecast_benchmark/scripts/17_test_extended_trajectories.py --workers 4
python lowflow_forecast_benchmark/scripts/18_compare_extended_trajectories.py
python lowflow_forecast_benchmark/scripts/19_run_extended_confirmation.py --shard-index 0 --shard-count 4
python lowflow_forecast_benchmark/scripts/19_run_extended_confirmation.py --shard-index 1 --shard-count 4
python lowflow_forecast_benchmark/scripts/19_run_extended_confirmation.py --shard-index 2 --shard-count 4
python lowflow_forecast_benchmark/scripts/19_run_extended_confirmation.py --shard-index 3 --shard-count 4
python lowflow_forecast_benchmark/scripts/20_assess_extended_confirmation.py
python lowflow_forecast_benchmark/scripts/21_test_extended_sensitivities.py --workers 4
python lowflow_forecast_benchmark/scripts/22_assess_extended_sensitivities.py
python lowflow_forecast_benchmark/scripts/23_plot_extended_results.py
```

The confirmation script also supports independent shards; see its `--help`
output before using a cluster or multiple terminals.

## D. Reproduce the separately tested 105/120-day extension

```bash
python lowflow_forecast_benchmark/scripts/24_run_120day_extension.py --shard-index 0 --shard-count 4
python lowflow_forecast_benchmark/scripts/24_run_120day_extension.py --shard-index 1 --shard-count 4
python lowflow_forecast_benchmark/scripts/24_run_120day_extension.py --shard-index 2 --shard-count 4
python lowflow_forecast_benchmark/scripts/24_run_120day_extension.py --shard-index 3 --shard-count 4
python lowflow_forecast_benchmark/scripts/25_assess_120day_extension.py
python lowflow_forecast_benchmark/scripts/26_test_120day_sensitivities.py --workers 4
python lowflow_forecast_benchmark/scripts/27_assess_120day_sensitivities.py
```

## E. Assemble derived forecasts and publication outputs

```bash
python temporal_coherence_paper/scripts/30_export_public_forecast_dataset.py --workers 4
python temporal_coherence_paper/scripts/31_assemble_public_forecast_dataset.py
python temporal_coherence_paper/scripts/34_prepare_observed_forecast_temporal_data.py --workers 4
python temporal_coherence_paper/scripts/36_run_reviewer_strengthening.py --workers 4
python temporal_coherence_paper/scripts/39_assess_uncertainty_sensitivity.py
python temporal_coherence_paper/scripts/41_export_zenodo_forecasts.py --workers 4
python temporal_coherence_paper/scripts/42_build_zenodo_deposit.py
python temporal_coherence_paper/scripts/43_build_accepted_figure_suite.py
python temporal_coherence_paper/scripts/42_build_zenodo_deposit.py --force
```

Stages 30 and 31 retain the earlier 30--90-day public-data build for provenance.
Stages 41 and 42 create the final Zenodo deposit for the 3,402-basin primary
Q10 forecast cohort at 30--120 days. All 3,402 basins have released records at
all six windows; manuscript scoring uses the smaller window-specific subsets
identified by `lead_is_scorable`.

Stage 43 regenerates the common figure sources and every accepted presentation
variant in publication order. The final Stage 42 pass refreshes the source-table
archive and checksums after the figures have been rebuilt.

Manuscript-authoring, DOCX-integration, submission-audit, and peer-review-package
utilities are intentionally excluded from the public code archive. They do not
change scientific results. The author-approved figure variants are stored in
the named figure subdirectories retained with the workflow.

## Reproducibility safeguards

- Configuration dates, thresholds, ensemble size, and random seed are frozen
  in JSON files.
- Long basin jobs use atomic output writes and can resume.
- Production stages write success receipts or decision files.
- Station identifiers are stored as strings so leading zeros are retained.
- Development, calibration, and final evaluation periods are chronological.
- The discarded AORC experiment is not included because it was not used by the
  paper and the author explicitly chose the GridMET-based workflow.
