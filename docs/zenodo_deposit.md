# Zenodo derived-data release

The supporting-data deposit is built from the final 3,402-basin primary Q10
forecast cohort, not from the earlier 30--90-day preliminary release. All six
forecast files use the same initialization records with complete 120-day
observed paths. Window-specific paper scores use only records marked
`lead_is_scorable`, which require at least 10 events and 10 non-events.

From the original project root, run:

```bash
python temporal_coherence_paper/scripts/41_export_zenodo_forecasts.py --workers 4
python temporal_coherence_paper/scripts/42_build_zenodo_deposit.py
```

Stage 41 writes restartable basin files for the 30-, 45-, 60-, 90-, 105-, and
120-day windows. Stage 42 assembles the lead-specific Parquet files, adds the
final basin and water-year scores, creates metadata and code archives, and
checks the released occurrence Brier scores against the frozen analysis.

The deposit intentionally excludes raw USGS, GAGES-II, and GridMET files.
Their authoritative access points are documented in the release provenance.
The output directory contains manual upload instructions, copy-ready Zenodo
metadata, a data dictionary, a file catalog, SHA-256 checksums, and a machine-
readable validation report.

Zenodo assigned DOI **10.5281/zenodo.22770792** to this derived-data
release. The identifier is recorded in this repository, `CITATION.cff`, and
the article Data Availability statement.

The frozen source-code snapshot contains only public analysis, validation,
data-release, and figure-generation code. Article text, authoring code,
submission utilities, and editorial correspondence are excluded.
