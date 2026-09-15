# Compact methods reference

## Threshold and forecasts

For each catchment, Q10 is the 10th percentile of normalized daily streamflow
in the pre-evaluation record. Forecasts are initialized weekly from observed
dry states. The selected hydrograph analog returns 101 complete future paths
conditioned on the hydrograph state and calendar information available at
forecast initialization.

## Same-marginal temporal-dependence experiment

Let \(x_{m,t}\) be flow for ensemble member \(m\) on future day \(t\). The
independent control applies a different random permutation to the member index
at every day. For any fixed day, the set \(\{x_{m,t}\}\) is unchanged, but a
member no longer traces a continuous hydrograph through time. Seasonal rank
reconstruction assigns the same sorted daily values to ranks taken from
complete historical trajectories selected by season.

## Brier score

For a binary event with observation \(y_i\in\{0,1\}\) and forecast probability
\(p_i\), the Brier score is

\[
\operatorname{BS}=\frac{1}{n}\sum_{i=1}^{n}(p_i-y_i)^2.
\]

Smaller values are better. Relative score reduction against a reference is

\[
100\left(1-\frac{S_{\mathrm{candidate}}}{S_{\mathrm{reference}}}\right).
\]

Positive values favor the candidate. Onset, duration, and cumulative deficit
use proper distributional scores in the full workflow; their implementations
are retained under `workflow/full_reproduction/lowflow_forecast_benchmark/`.

## Aggregation and uncertainty

Primary continental comparisons are paired within basin, event quantity, and
forecast-window length. The archived analysis uses basin-level replication and
retains all nine aggregated GAGES-II ecoregions in each stratified bootstrap
replicate. A wider two-stage sensitivity resamples both basins and regions.
Leave-one-water-year-out calculations test whether a single held-out year
controls the same-marginal result.

## Interpretation boundary

The analysis isolates temporal dependence under retrospective conditions. It
does not claim that the hydrograph analog is a complete operational forecast
system, and it does not use future weather observations to select eligible
forecast initializations in the final 1–120-day experiment.
