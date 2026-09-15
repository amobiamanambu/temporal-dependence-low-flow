# Literature foundation and claim boundaries

This benchmark translates the supplied suggestions into tests that the available data can
actually support.  It does not use method names as decoration: each label below states what
is implemented and what is not.

## Forecasting context

The closest current continental benchmark is the USGS River DroughtCast evaluation over
3,219 CONUS gauges and leads of 1--13 weeks.  LSTM and LightGBM forecasts improved continuous
low-percentile estimates, but discrete drought-class forecasts were generally worse than
persistence; severe-drought predictive power was concentrated at roughly 1--4 weeks.  That
makes persistence a mandatory comparator and makes calibration, event definition, and lead
dependence central rather than optional.

- Russell et al. (2026), *Machine learning-generated streamflow drought forecasts for the
  conterminous United States*, USGS publication page:
  https://www.usgs.gov/publications/machine-learning-generated-streamflow-drought-forecasts-conterminous-united-states-0

## Implemented methods

1. **Binary isotonic distributional regression.**  IDR provides an order-restricted,
   tuning-light probabilistic benchmark.  Here the order is current discharge and the
   response is the low-flow threshold event; this is the binary special case, not a claim
   that a multivariate full-distribution IDR was fitted.
   - Henzi, Ziegel, and Gneiting (2021), JRSS-B, doi:10.1111/rssb.12450.

2. **Generalized-Pareto lower-tail splice.**  Peaks-over-threshold theory motivates a GPD
   for excesses beyond a sufficiently extreme threshold.  The implementation splices a GPD
   onto the lower 10% of standardized innovations and retains the empirical body.  It is
   accepted only if held-out Brier score and lower-tail weighted CRPS improve.
   - Pickands (1975), *Statistical inference using extreme order statistics*, Annals of
     Statistics, doi:10.1214/aos/1176343003.

3. **Regional standardized-innovation borrowing.**  Ecoregional pools are made from
   pre-2010 standardized innovations in basins outside the candidate panel.  Local and
   regional distributions are partially pooled; raw discharge is never pooled across
   basins.

4. **Two-component asymmetric-Laplace mixture.**  A finite mixture is fitted to
   standardized innovations by constrained maximum likelihood.  This tests the distributional
   idea behind conditional mixture approaches without calling the result CMAL-LSTM, because
   no neural-network parameter model is fitted.
   - Papacharalampous et al. (2022), HESS 26, 1673--1705,
     doi:10.5194/hess-26-1673-2022.

5. **Quantile-regression-forest analogue.**  Random-forest terminal leaves retain empirical
   training outcomes and therefore define a predictive distribution.  The test uses only
   initialization-time hydrograph and antecedent GridMET predictors.
   - Meinshausen (2006), JMLR 7, 983--999:
     https://www.jmlr.org/papers/v7/meinshausen06a.html

6. **Coherent historical-block trajectories.**  Whole observed dry-spell trajectory blocks
   are resampled from hydroclimatically similar training initializations.  This preserves
   within-spell dependence.  The decisive control independently permutes ensemble-member
   identity at every lead, preserving every daily predictive marginal exactly while
   destroying cross-day dependence.  Any difference in first-passage time or deficit score
   can therefore be attributed to path coherence rather than one-day marginal skill.  The
   method is related to the dependence-restoration objective of the Schaake shuffle but is
   labelled analog block resampling because no post hoc ensemble-rank reordering is used.
   - Clark et al. (2004), *The Schaake shuffle*, Journal of Hydrometeorology,
     doi:10.1175/1525-7541(2004)005<0243:TSSATM>2.0.CO;2.

7. **Calibration-period stacking and recalibration.**  Component forecasts are fitted
   before 2010; only 2010--2015 is used to fit the regularized stack or isotonic probability
   recalibrator.  The 2016--2025 period is not used for weights.  Split conformal intervals
   are reported as a coverage correction, not as added discrimination.
   - Ziegel and Gneiting (2024), *Conformalized Isotonic Distributional Regression*,
     PMLR 230: http://proceedings.mlr.press/v230/ziegel24a.html

8. **Dry-spell survival mixture.**  Forecasts begin on an observed dry day.  A model using
   only initialization-time predictors estimates whether the dry spell will persist, then
   mixes separately estimated continued-dry and interrupted-spell low-flow branches.  The
   observed future branch is retained only as an explicitly ineligible oracle diagnostic.

9. **First-passage time and cumulative deficit.**  Coherent trajectory ensembles are scored
   for the first crossing of Q20/Q10/Q5 and accumulated deficit below each threshold.  These
   targets answer onset timing and severity questions that an endpoint probability cannot.

## Verification

Brier score and logarithmic score evaluate threshold probabilities.  CRPS evaluates full
predictive distributions.  A threshold-weighted CRPS with weight one below the low-flow
threshold tests the lower tail without the impropriety caused by simply scoring only cases
that later became extreme.

- Gneiting and Ranjan (2011), *Comparing Density Forecasts Using Threshold- and
  Quantile-Weighted Scoring Rules*, JBES 29, 411--422,
  doi:10.1198/jbes.2010.08110.

Uncertainty is assessed with paired resampling at the basin and GAGES-II ecoregion levels.
Promotion requires improvement over the archived adaptive heavy-tail/recession-logistic
model, not merely improvement over climatology.

## Tests that cannot be completed with the present archive

GEFS/ECMWF forcing ensembles, true groundwater states, irrigation-return observations,
reservoir release schedules, and operational hindcast issue times are absent.  Therefore
the benchmark cannot honestly claim a complete real-time forecasting system or test the
incremental value of numerical weather predictions.  These are data boundaries, not model
failures.

## Extended-range temporal-coherence experiment

The extended experiment uses the same conceptual distinction emphasized in multivariate
ensemble postprocessing: calibrated univariate forecast distributions do not determine a
valid multiday trajectory.  Ensemble copula coupling, the Schaake shuffle, and analog-based
dependence reconstruction all address this problem, so the paper must not claim that temporal
reordering itself is new.

- Schefzik, Thorarinsdottir, and Gneiting (2013), *Multivariate probabilistic forecasting
  methods based on ensemble copula coupling*, Statistical Science,
  doi:10.1214/13-STS443.
- Schefzik (2016), *A similarity-based implementation of the Schaake shuffle*, Monthly
  Weather Review, doi:10.1175/MWR-D-15-0227.1.
- Clark et al. (2004), *The Schaake shuffle*, Journal of Hydrometeorology,
  doi:10.1175/1525-7541(2004)005<0243:TSSATM>2.0.CO;2.

What the experiment can add is an exact controlled measurement of how much low-flow hazard
information is lost when temporal member identity is destroyed while every daily marginal
forecast is preserved.  The primary outcomes—first onset, onset time, duration, and cumulative
deficit—are path functionals, whereas endpoint state is an intentional negative control.

The closest continental operational comparison is River DroughtCast, which evaluates
streamflow drought forecasts at 3,219 CONUS gauges from 1 to 13 weeks.  It prevents any claim
that this is the first continental 90-day low-flow forecast, but it also establishes the
practical importance of persistence, drought-category prediction, and lead-dependent skill.
The present experiment differs by isolating temporal coherence and by scoring full-window
onset time, duration, and deficit.

- Russell et al. (2026), *Machine learning-generated streamflow drought forecasts for the
  conterminous United States*, USGS publication page:
  https://www.usgs.gov/publications/machine-learning-generated-streamflow-drought-forecasts-conterminous-united-states-0

Event onset, duration, and deficit are established drought characteristics rather than new
definitions.  The novelty boundary is therefore the controlled continental forecast result,
not the existence of these quantities.

- Tijdeman et al. (2021), *Reducing uncertainties in setting up a framework for drought
  hazard assessment across large regions*, Hydrology and Earth System Sciences 25,
  3991--4023: https://hess.copernicus.org/articles/25/3991/2021/
