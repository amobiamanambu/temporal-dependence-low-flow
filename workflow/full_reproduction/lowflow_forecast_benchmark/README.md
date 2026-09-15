# Low-flow forecast experiments

This directory contains the complete sequence used to screen candidate methods,
freeze the selected configuration, confirm it outside the development panel,
and extend the temporal-dependence experiment to 120 days.

## Chronological design

- fitting: 1980-01-01 through 2009-12-31;
- configuration and calibration: 2010-01-01 through 2015-12-31;
- held-out evaluation: 2016-01-01 through 2025-12-31.

The early candidate suite contains both conditional dry-spell experiments and
future-unrestricted forecasts. Their results are never pooled. The final
1–120-day analysis initializes on an observed dry state but does not use future
meteorological observations to select cases or construct the forecast.

## Script map

- `00`–`01`: audit inputs and select the 192-basin development panel.
- `02`–`08`: endpoint candidates, tail models, calibration, and regional
  innovation borrowing.
- `09`: coherent historical paths and an exact same-marginal shuffle.
- `10`–`11`: future-unrestricted occurrence, first-onset, duration, and deficit.
- `12`: freeze the candidate-screen decision.
- `13`–`15`: out-of-panel confirmation and diagnostic figures.
- `16`–`18`: audit and freeze the 1–90-day trajectory experiment.
- `19`–`23`: continental 1–90-day confirmation and sensitivities.
- `24`–`27`: separately tested 105/120-day extension and sensitivities.

The `scripts/lib/` modules hold the shared data loading, trajectory,
postprocessing, and proper-score implementations. `config.json` records the
dates, thresholds, minimum sample sizes, ensemble size, and random seed.

## Execution

Run from the parent `workflow/full_reproduction/` directory. To avoid numerical
library oversubscription when using four basin workers:

```bash
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
python lowflow_forecast_benchmark/scripts/run_candidate_suite.py --workers 4
```

Then follow the numbered order documented in `../../../docs/full_reproduction.md`.
All expensive basin stages write atomic files and retain completed basins when
restarted.

## Scope

The repository retains all candidate methods because the selection process is
part of the scientific provenance. The abandoned AORC experiment is excluded:
it did not contribute to the paper's methods, results, or conclusions.
