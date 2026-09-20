# Full continental workflow

This directory contains the production code for the temporal-dependence
low-flow study. Run commands from this directory so that relative paths resolve
correctly.

- `upstream/`: retrieval and quality control of provider data.
- `scripts/`: preparation of matched daily discharge and meteorological data
  through the snowmelt screen (Stages 00–09).
- `lowflow_forecast_benchmark/`: the frozen development panel, trajectory
  comparison, continental confirmation, and 105/120-day extension.
- `temporal_coherence_paper/`: dependence reconstruction, validation,
  uncertainty, and data-release code.
- `figures/`: one documented entry point for every article figure.

Generated results, raw provider data, caches, discarded experiments, private
article text, authoring utilities, and submission files are intentionally
excluded. See `../../docs/full_reproduction.md` for the exact run order.
