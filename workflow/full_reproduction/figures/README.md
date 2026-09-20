# Figure workflow

These scripts generate the article’s final figure set from archived analysis
products. They do not contain or construct article text.

| Figure | Entry point | Content |
|---|---|---|
| 1 | `01_study_area.py --design hydrologic` | Study area and basin accounting |
| 2 | `02_representative_forecast_trajectories.py` | Representative forecasts and observations |
| 3, 5 | `03_04_05_08_core_figures.py` | Trajectory anatomy and score summaries |
| 4 | `04_same_marginal_basin_scores.py` | Basin scores with fixed daily marginals |
| 6 | `06_basin_skill_trajectories.py` | Basin-level skill across window lengths |
| 7 | `07_selected_basin_validation.py` | Selected-basin temporal validation |
| 8 | `08_ecoregional_calibration.py` | Ecoregional occurrence calibration |
| 9 | `09_continental_temporal_validation.py` | Continental temporal validation |
| 10 | `10_conditional_skill_span.py` | Conditional skill span |
| 11 | `11_spatial_calibration_bias.py --design 5` | Spatial calibration bias |

`03_04_05_08_core_figures.py` also supplies common source tables used by the
final wrappers. Run the complete suite with:

```bash
python temporal_coherence_paper/scripts/43_build_figure_suite.py
```

Generated files are written to `figures/output/`; tabular plotting data are
written to `figures/source_tables/`. Both directories are ignored by Git.
