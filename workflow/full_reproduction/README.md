# Production workflow archive

This directory preserves the directory relationships expected by the scripts
used in the continental analysis. Run commands from this directory so relative
paths resolve correctly.

- `upstream/`: provider-data download, climate extraction, and discharge QC.
- `scripts/`: matched discharge–meteorology preparation (Stages 00–11).
- `lowflow_forecast_benchmark/`: candidate screening, frozen confirmation,
  same-marginal comparisons, and 105/120-day extension.
- `temporal_coherence_paper/`: derived-data export, temporal validation, data-release utilities, and publication
  figure code.

The folder intentionally excludes generated results, raw data, Python caches,
abandoned figure designs, manuscript backups, and the unused AORC experiment.
Manuscript text and manuscript-authoring or submission-package utilities are
also intentionally excluded.
See `../../docs/full_reproduction.md` for the run order and required inputs.
