# Compact example data

These files are real, archived extracts from the study and are small enough to
track in Git.

| File | Contents |
|---|---|
| `forecast_predictions_sample.csv.gz` | 1,000 held-out Q10-onset forecasts from five USGS gages, four window lengths, and 2016–2025 initialization dates. |
| `figure_03_trajectory_anatomy_members.csv.gz` | One 120-day, 101-member forecast under intact, independently reordered, and seasonally reconstructed member continuity. |
| `figure_03_trajectory_anatomy_probabilities.csv` | Observed flow, Q10, and cumulative occurrence probabilities for the trajectory example. |
| `figure_03_trajectory_anatomy_metadata.json` | Gage, initialization date, and deterministic selection rule for the trajectory example. |
| `figure_03_same_marginal_scores.csv` | Archived continental score summary used to verify relative-score calculations. |
| `FORECAST_DATA_DICTIONARY.csv` | Field-level units, types, and definitions for the forecast sample. |

The forecast sample contains only Northeast gages and is not statistically
representative of the continental population. It exists to test input parsing,
scoring, and plotting. Use the archived full derived-data release to reproduce
paper estimates.

Station identifiers must remain strings. Missing onset times mean no observed
crossing occurred within that forecast window; the censored-onset column stores
`lead_days + 1` for those cases.
