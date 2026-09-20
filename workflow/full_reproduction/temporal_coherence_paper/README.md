# Dependence reconstruction and release stages

This directory contains only the post-forecast analyses and release utilities
used by the temporal-dependence study:

- `32_generate_figure_sources.py` creates common figure-source tables;
- `34_prepare_observed_forecast_temporal_data.py` prepares temporal validation
  summaries;
- `36_run_dependence_reconstruction.py` compares intact, independently ordered,
  and seasonally reconstructed trajectories;
- `39_assess_uncertainty_sensitivity.py` compares uncertainty schemes;
- `41_export_zenodo_forecasts.py` exports forecast-level release data;
- `42_build_zenodo_deposit.py` validates and assembles the Zenodo package;
- `43_build_figure_suite.py` runs the numbered figure entry points.

The folder contains no article text, document builders, editorial
correspondence, or submission utilities.
