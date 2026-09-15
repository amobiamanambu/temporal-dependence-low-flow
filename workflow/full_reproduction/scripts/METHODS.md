# Methods specification

## Analysis status

This revised workflow was designed after the initial continental outputs were inspected. Accordingly, the continental scaling law is treated as an exploratory discovery result rather than a data-blind confirmatory test. The revised decision rules are frozen in `config.json` before rerunning Stages 12–20, all exclusions and negative decisions are retained, and the 2016–2025 future-period test supplies temporal validation of conditional low-flow predictability horizons. Independent data would still be needed for fully external confirmation.

## Scope and estimands

The workflow studies two related but distinct empirical objects.

1. **Statistical time irreversibility of streamflow.** For a discretized, deseasonalized log-flow process with lag `τ`,

   `I(τ) = Σᵢⱼ pᵢⱼ(τ) log[pᵢⱼ(τ) / pⱼᵢ(τ)]`,

   where `pᵢⱼ(τ)` is the regularized joint probability of state `i` at `t` and state `j` at `t+τ`. The statistic is nonnegative and is zero for a pairwise time-reversible process. It is reported in nats and is not labeled physical thermodynamic entropy production.

2. **Conditional recession variability.** For normalized flow `q` and a forcing-screened increment `Δqτ = q(t+τ)-q(t)`,

   `D1(q,τ) = E[Δqτ | q(t)=q] / τ`

   `V(q,τ) = Var[Δqτ | q(t)=q] / (2τ)`.

   Power laws `-D1(q) = a qᵇ` and `V(q) = c qᵐ` are fit in log-log space. `m` is called the conditional recession-variance exponent. The raw finite-time Kramers–Moyal quantities `E[Δq²|q]/(2τ)` and `E[Δq⁴|q]/(24τ)` are stored separately; the centered variance is not mislabeled as a formal diffusion coefficient.

## Data preparation and eligibility

- Period: 1980-01-01 through 2025-12-31, controlled in `config.json`.
- Q: USGS daily mean discharge parameter 00060/statistic 00003. Source qualifiers are preserved.
- P and PET: basin GridMET inputs supplied in the project root.
- Temperature: GridMET daily minimum and maximum air temperature; Tmean is `(Tmin+Tmax)/2`.
- Discharge is converted from cubic feet per second to millimetres per day by `q = Q × 2.446575545 / area_km²`.
- Daily data are joined on exact gage ID and date. No general-purpose interpolation is performed.

The primary analysis requires an accepted discharge tier, at least 3,650 jointly complete days, at least 80% joint completeness within the observed Q record, nonnegative Q/P/PET under the configured tolerances, consistent Tmin ≤ Tmax, and at least two center-in-polygon GridMET cells. Tier 1, Tier 1+2, reference-only, diagnostic-complete, and finite-time-screen-pass populations are reported separately.

## Snow and no-input screening

Precipitation phase is linearly partitioned between configured cold and warm thresholds. The defaults, −0.4 °C and 2.4 °C, span the station-based range reported by Jennings et al. The storage screen uses a degree-day melt factor of 3 mm °C⁻¹ day⁻¹. This proxy omits important snow physics and is used only to exclude likely snowmelt-input days; it is not reconstructed SWE.

For an increment from `t` to `t+τ`, a dry state requires P below the threshold and no proxy snowmelt from `t-(a-1)` through `t+τ`. Primary values are P ≤ 0.1 mm/day and `a=3`. All configured P thresholds and antecedent windows are estimated across the continental population. Snow/rain temperature thresholds and degree-day factors are rerun on the prespecified stratified validation sample.

Three base transition populations are retained:

- Q-only monotone: valid `Δq ≤ 0` runs.
- P/snow-screened monotone: valid `Δq ≤ 0` dry-state runs.
- P/snow-screened dry state: all valid increments in dry-state runs, including positive increments.

The third is primary because preselecting only negative increments truncates conditional variability.

## Irreversibility spectrum

Q is transformed with `log1p`, and a calendar-day median climatology is removed. Only gaps of at most three days are interpolated for this isolated estimator. The longest qualifying complete segment must contain at least 500 days. Estimator-specific exclusions are recorded rather than treated as computational failures.

Quantile states are evaluated at 8, 12, and 16 bins; 12 is primary. A Jeffreys pseudocount of 0.5 prevents infinite finite-sample KL values. Primary continental lags are 1, 2, 3, 7, 14, 30, 60, and 90 days. A dominant lag of 90 days is marked right-censored.

IAAFT surrogates preserve the marginal distribution and approximately preserve the Fourier amplitude spectrum. They test whether path asymmetry exceeds the expectation from finite sample, autocorrelation, and a static nonlinear rescaling of a Gaussian linear process. Validation uses 199 surrogates and the continental run uses 19. An ordinal forward/reverse KL statistic provides a bin-free robustness measure.

The novelty claim is not “first streamflow irreversibility.” Stage 13 is an independent secondary analysis. Its 19-surrogate continental p-values have resolution 0.05 and are therefore reported as coarse exceedance ranks, not strong multiple-testing evidence. Unordered state-pair contributions use `(pij-pji) log(pij/pji)` and sum to the total path-asymmetry KL; tail labels refer to deseasonalized anomaly states.

## Recession drift, variance, and diagnostics

Flow is normalized by each basin’s positive median. Equal-frequency state bins reduce sample imbalance. Each bin requires at least 50 increments and each power fit requires at least six valid bins. Uncertainty uses a year-block bootstrap.

Finite-time stability is assessed at `τ=1, 2, 3` days. A full-series Chapman–Kolmogorov context screen compares the observed two-step transition matrix with the square of the one-step matrix; because it is not conditioned on the same dry-state path ensemble, it cannot establish Markov behavior for the forcing-screened process. Raw second and fourth Kramers–Moyal moments are retained. The centered fourth-moment ratio is compared with the locally Gaussian finite-time benchmark 0.5. This is a local-Gaussian adequacy check, not a Pawula-limit proof.

Diagnostic availability and passing are separate fields. Missing results never count as passing. A descriptive finite-time-screen subset requires both diagnostics to be finite, Chapman–Kolmogorov error at or below its tolerance, and local-Gaussian deviation at or below its tolerance. Even inside that subset, `V(q)` is described as conditional increment-variance scaling rather than an identified infinitesimal diffusion coefficient.

## Stage 16: scaling and model-class decisions

The central empirical model is `Var(Δqτ | q) = c q^m`. Stage 16 reports estimation coverage, bootstrap confidence-interval classifications relative to `m=2`, power-law fit, and stability across `τ=1,2,3`. A DerSimonian–Laird random-effects meta-analysis quantifies both a continental center and between-basin heterogeneity; its prediction interval is more relevant to universality than the pooled mean alone.

The near-multiplicative organizing tendency passes only if estimation coverage, positive confidence intervals, median fit, cross-lag rank stability, continental mean and median proximity to two, and replication in both Tier-1 and Tier-2 quality populations all meet prespecified thresholds. Exact universal `m=2` is a separate claim requiring at least 80% of individual intervals to contain two. The fraction passing both finite-time screens is reported descriptively; the workflow does not convert it into a formal Gaussian-diffusion claim.

Forcing thresholds, antecedent windows, and monotone-only selection are reported as sensitivities. Hydroclimatic driver correlations, HC3 regression, and leave-one-group-out transfer are secondary and cannot establish mechanism.

## Stage 17: distribution and observation robustness

For each basin, the binned conditional mean `E[Δq|q]` is interpolated and removed. Residuals are standardized using `sqrt(2 c q^m)` for a one-day increment. A valid location-scale collapse should yield standardized distributions with similar mean, variance, and shape across five flow-quantile groups; pairwise Kolmogorov–Smirnov distances quantify remaining dependence.

Non-Gaussianity is assessed using skewness, excess kurtosis, tail frequency, a normal KS distance, and normal-versus-moment-matched Student-t AIC. This identifies a useful predictive distribution family; it is not a proof of a unique physical jump process. Distributional collapse requires joint limits on pairwise KS distance, group mean, and group standard deviation. The exponent and tail diagnostics are independently re-estimated after removing transitions for which either endpoint's USGS qualifier contains `:e`. A second conservative tail analysis also removes both endpoints at or below the basin-specific first percentile of positive dry-transition endpoint flows and exact zero increments; availability, Student-t support, and tail frequency relative to the Gaussian reference must all pass prespecified population thresholds.

## Stage 18: conditional low-flow predictability horizons

Each basin is scaled by its positive pre-2016 median discharge. Q20, Q10, and Q5 thresholds are estimated using observations through 2015-12-31. Direct-lead model fitting uses only dry-state transitions whose destination also falls before that date; evaluation starts 2016-01-01. Leads are 1, 2, 3, 5, 7, 10, 14, 21, and 30 days. Evaluation is restricted to observed periods with precipitation below the primary threshold and no proxy snowmelt throughout the antecedent-plus-lead window.

The direct-lead conditional mean is estimated in equal-frequency flow bins. The conditional-variance exponent is regularized toward `m=2` with a prespecified normal prior scale of 0.75 and bounded to 0–4 for numerical stability; both raw and regularized estimates are retained. Prediction states are clipped to the training support before the mean and scale functions are evaluated, preventing uncontrolled power-law extrapolation. This regularization is a prediction safeguard, not evidence for universal `m=2`.

Stage 17 showed that one basin-wide standardized distribution did not collapse across flow groups. Therefore Stage 18 uses separate training-only empirical innovation distributions for five flow-state groups. Two Gaussian comparators are used: one basin-wide Gaussian and one state-specific Gaussian whose group means and standard deviations match the empirical model. State-Gaussian versus global-Gaussian skill tests state dependence; empirical versus state-Gaussian skill isolates non-Gaussian distribution shape. Predicted flows are constrained to be nonnegative.

Three prespecified probability benchmarks prevent weak claims:

1. the training dry-state event climatology;
2. persistence of the current low-flow status; and
3. a training-only, ten-bin empirical probability conditioned on current flow.

Q20/Q10/Q5 probabilities are verified with Brier and logarithmic scores. CRPS, 90% interval coverage/error/width, and median absolute error compare empirical and Gaussian continuous distributions. Basin-paired score improvements first receive a hierarchical bootstrap that resamples hydroclimatic spatial groups and then basins within groups. A separate block analysis resamples whole evaluation water years, spatial groups, and basins, preserving within-year serial and event clustering. A lead-level continental claim requires at least 500 paired basins, at least five spatial groups, and a 95% Brier-improvement interval wholly above zero under both analyses.

For each threshold and benchmark, the continental predictability horizon is the largest uninterrupted configured lead satisfying that rule from day 1 onward. The robust reported horizon is the smaller of the spatial-bootstrap and water-year-block horizons. The all-baseline horizon must beat all three benchmarks; its conservative improvement and confidence bound are the minima across the three comparisons. A later isolated positive lead cannot extend a horizon past an earlier failure. Basin horizons use the same uninterrupted rule with positive held-out Brier improvement but are descriptive because individual-basin confidence intervals are not used. Paired Q5-minus-Q10 and Q5-minus-Q20 water-year-block contrasts at the prespecified seven-day target compare Brier skill scores relative to the same flow-bin benchmark within each threshold, rather than comparing raw Brier improvements across unequal event rates.

The headline title requires the robust Q10 all-baseline horizon to reach at least seven days, measurement/rounding-robust heavy tails, and incremental CRPS value from both state dependence and empirical tail shape under both bootstrap designs. The underlying dynamics paper can remain scientifically supported if the scaling and measurement-robust distribution tests pass even when predictability is short. This is a conditional no-effective-input scenario test, not an unconditional forecast: future forcing is observed only to select evaluation transitions. A real-time implementation would require archived precipitation/snowmelt forecasts and latency testing.

## Failure criteria

The central claims must be narrowed, split, or rejected if:

- variance exponents are unstable across `τ`, have poor fit, or are artifacts of forcing-screen choices;
- the exponent disappears or rankings collapse after removing estimated discharge values;
- standardized residual distributions do not approximately collapse across flow groups and that dependence is ignored rather than modeled explicitly;
- Gaussian diffusion is inferred from finite-time daily diagnostics that do not identify an infinitesimal process;
- exact universal `m=2` is claimed when individual confidence intervals show heterogeneity;
- non-Gaussianity is given a unique physical cause without tail/jump/measurement discrimination;
- conditional models fail to improve proper scores over climatology, persistence, and the flow-bin benchmark, in which case the predictability horizon is short or zero;
- dry-spell conditional skill is described as unconditional operational forecasting; or
- irreversibility is used to rescue the primary paper despite lack of a reproducible mechanistic bridge.
