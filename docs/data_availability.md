# Data-availability plan

## What does not need to be republished

The repository should not duplicate the complete USGS daily-value archive,
GAGES-II shapefiles, or GridMET NetCDF files. These inputs are large and remain
available from their original custodians. The code records the variables,
dates, station identifiers, download services, screening rules, and local file
layout needed to reconstruct them.

## What is new research data

The study created outputs that do not exist at USGS or GridMET:

- accepted/excluded forecast-initialization records;
- 101-member path-derived event forecasts;
- basin-level proper scores and benchmark comparisons;
- same-marginal, seasonal-reconstruction, and sensitivity results;
- figure-source tables and data dictionaries.

These derived outputs support the paper's numerical claims and should be
archived with a permanent identifier. A separate data article is unnecessary,
but a versioned repository deposit is scientifically preferable to relying on
mutable GitHub files alone.

## Recommended release split

**GitHub**

- source code, configuration, tests, documentation;
- the small five-gage example;
- example figures and checksums.

**Zenodo or another DOI-issuing repository**

- a frozen release of this GitHub repository;
- basin-level score tables and exclusion records;
- complete figure-source data;
- six primary-Q10-cohort forecast tables for 30, 45, 60, 90, 105, and 120 days;
- final basin- and water-year-level score tables for all trajectory-ordering
  configurations;
- machine-readable metadata, licenses, checksums, and a data dictionary.

This arrangement avoids republishing raw provider data while making the
study-specific results independently citable and reproducible.

The final release is built by Stages 41 and 42 described in
[`zenodo_deposit.md`](zenodo_deposit.md). All six forecast files use the same
initialization records with complete 120-day observed paths for the 3,402-basin
primary Q10 forecast cohort. The `lead_is_scorable` field identifies the
smaller, window-specific subsets that meet the frozen minimum of 10 events and
10 non-events.

## Manuscript statement

> USGS daily discharge, GAGES-II attributes, and GridMET meteorological data
> are available from the original providers cited in the manuscript. The code
> and compact example are available at
> https://github.com/amobiamanambu/temporal-dependence-low-flow, and the
> basin-level verification outputs, figure-source data, and derived
> probabilistic forecasts supporting this study are archived in Zenodo at
> https://doi.org/10.5281/zenodo.22770792.
