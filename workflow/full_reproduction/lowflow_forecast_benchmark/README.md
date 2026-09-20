# Low-flow trajectory experiment

This directory contains the frozen forecasting workflow reported in the
article. The chronology is:

- fitting: 1980-01-01 through 2009-12-31;
- configuration and calibration: 2010-01-01 through 2015-12-31;
- held-out evaluation: 2016-01-01 through 2025-12-31.

The final analysis initializes forecasts from an observed dry state. Future
meteorological observations are not used to select cases or construct a
forecast.

## Script map

- `00` audits the required inputs.
- `01` materializes the frozen 192-basin development panel stored in
  `development_panel.csv`.
- `17`–`18` compare five trajectory constructions on that panel and freeze the
  selected hydrograph-state analog.
- `19`–`20` run and assess the out-of-panel 1–90-day confirmation.
- `21`–`22` evaluate the associated sensitivity experiments.
- `24`–`25` run and assess the separately tested 105/120-day extension.
- `26`–`27` evaluate extension sensitivities.

Shared implementations are in `scripts/lib/`. `config.json` freezes the date
splits, thresholds, ensemble size, minimum sample sizes, and random seed.

Run from `workflow/full_reproduction/`. Expensive basin stages write atomic
files and retain completed work when restarted. The full command sequence is
given in `../../../docs/full_reproduction.md`.

Discarded candidate models and experiments not used in the article are not
part of this release.
