# Full continental reproduction

## Requirements

The continental workflow requires the USGS, GAGES-II, and GridMET inputs listed
in `data_sources.md`, substantial storage for intermediate daily files, and a
Python environment with the geospatial packages in `requirements-full.txt`.

```bash
python -m pip install -r requirements-full.txt
```

Run all commands below from `workflow/full_reproduction/`. Continue only after
each stage writes its success receipt.

## A. Retrieve and prepare provider data

```bash
python upstream/01_download_usgs_daily_discharge.py
python upstream/02_download_gridmet_pr_pet.py
python upstream/03_concatenate_gridmet_pr_pet.py
python upstream/04_extract_gridmet_pr_pet.py
python upstream/05_quality_control_discharge.py
```

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
```

Stage 09 supplies the daily discharge, precipitation, potential
evapotranspiration, temperature, and snowmelt-screen fields used by the
forecast experiment. No result from a separate analysis project is required.

## C. Reproduce trajectory selection and 1–90-day confirmation

```bash
python lowflow_forecast_benchmark/scripts/00_audit_inputs.py
python lowflow_forecast_benchmark/scripts/01_select_screening_panel.py
python lowflow_forecast_benchmark/scripts/17_test_extended_trajectories.py --workers 4
python lowflow_forecast_benchmark/scripts/18_compare_extended_trajectories.py
python lowflow_forecast_benchmark/scripts/19_run_extended_confirmation.py --shard-index 0 --shard-count 4
python lowflow_forecast_benchmark/scripts/19_run_extended_confirmation.py --shard-index 1 --shard-count 4
python lowflow_forecast_benchmark/scripts/19_run_extended_confirmation.py --shard-index 2 --shard-count 4
python lowflow_forecast_benchmark/scripts/19_run_extended_confirmation.py --shard-index 3 --shard-count 4
python lowflow_forecast_benchmark/scripts/20_assess_extended_confirmation.py
python lowflow_forecast_benchmark/scripts/21_test_extended_sensitivities.py --workers 4
python lowflow_forecast_benchmark/scripts/22_assess_extended_sensitivities.py
```

The four confirmation commands are independent shards. Use the same shard
count for every invocation and do not assess the results until all four finish.

## D. Reproduce the 105/120-day extension

```bash
python lowflow_forecast_benchmark/scripts/24_run_120day_extension.py --shard-index 0 --shard-count 4
python lowflow_forecast_benchmark/scripts/24_run_120day_extension.py --shard-index 1 --shard-count 4
python lowflow_forecast_benchmark/scripts/24_run_120day_extension.py --shard-index 2 --shard-count 4
python lowflow_forecast_benchmark/scripts/24_run_120day_extension.py --shard-index 3 --shard-count 4
python lowflow_forecast_benchmark/scripts/25_assess_120day_extension.py
python lowflow_forecast_benchmark/scripts/26_test_120day_sensitivities.py --workers 4
python lowflow_forecast_benchmark/scripts/27_assess_120day_sensitivities.py
```

## E. Reconstruct dependence, validate, and build release products

```bash
python temporal_coherence_paper/scripts/34_prepare_observed_forecast_temporal_data.py --workers 4
python temporal_coherence_paper/scripts/36_run_dependence_reconstruction.py --workers 4
python temporal_coherence_paper/scripts/39_assess_uncertainty_sensitivity.py
python temporal_coherence_paper/scripts/41_export_zenodo_forecasts.py --workers 4
python temporal_coherence_paper/scripts/42_build_zenodo_deposit.py
python temporal_coherence_paper/scripts/43_build_figure_suite.py
python temporal_coherence_paper/scripts/42_build_zenodo_deposit.py --force
```

Stages 41–42 reproduce the final 3,402-basin, Q10, six-window Zenodo release.
Stage 43 rebuilds figure-source tables and Figures 1–11. The final Stage 42
pass refreshes checksums after the figure sources have been generated.

## Safeguards

- Configuration dates, thresholds, ensemble size, and random seed are frozen.
- Station identifiers remain strings so leading zeros are retained.
- Development, calibration, and held-out evaluation periods are chronological.
- Long-running basin stages use atomic writes and restart from completed files.
- Raw provider data, discarded experiments, and private authoring material are
  not packaged in the public release.
