# External data sources and expected inputs

Raw provider data are intentionally excluded from Git because they are large,
versioned by their custodians, and independently accessible.

## U.S. Geological Survey daily discharge

- Provider: USGS Water Data for the Nation.
- Parameter: 00060, discharge.
- Statistic: 00003, daily mean.
- Study interval: 1980-01-01 through 2025-12-31.
- Service: <https://waterservices.usgs.gov/>
- Download script:
  `workflow/full_reproduction/upstream/01_download_usgs_daily_discharge.py`

The downloader writes compressed RDB batches, SHA-256 checksums, and a
restartable manifest. Preserve station identifiers as eight-character strings.

## GAGES-II

- Provider: U.S. Geological Survey.
- Role: basin boundaries, drainage area, reference status, and catchment
  attributes.
- Reference: Falcone, J.A. (2011), *GAGES-II: Geospatial Attributes of Gages for
  Evaluating Streamflow*, USGS data release/report resources.

Required local files include the point shapefile, basin polygon shapefile, and
the GAGES-II attribute archive named in
`workflow/full_reproduction/scripts/config.json`.

## GridMET

- Provider: Climatology Lab / Northwest Knowledge Network.
- Variables: precipitation (`pr`), potential evapotranspiration (`pet`),
  minimum temperature (`tmmn`), and maximum temperature (`tmmx`).
- Study interval: 1980–2025.
- Archive: <https://www.northwestknowledge.net/metdata/>
- Reference: Abatzoglou, J.T. (2013), *Development of gridded surface
  meteorological data for ecological applications and modelling*,
  International Journal of Climatology, <https://doi.org/10.1002/joc.3413>.

The upstream scripts download annual `pr` and `pet` files. Production Stages
01–03 download, validate, and extract `tmmn` and `tmmx`. Basin averages use the
native grid, cosine-latitude area weighting, and a documented small-basin
fallback. The primary analysis requires at least two basin-interior grid-cell
centers.

## Expected full-reproduction layout

Place provider data under `workflow/full_reproduction/` so the configuration
resolves paths without editing code:

```text
workflow/full_reproduction/
├── basin_inventory.csv
├── basin_precipitation.csv
├── basin_pet.csv
├── gagesII_9322_sept30_2011.{shp,shx,dbf,prj}
├── basinchar_and_report_sept_2011/
├── gridmet_data/
└── usgs_dv_00060_1980_01_01_to_2025_12_31/
```

If filenames differ, edit only the `inputs` block of `scripts/config.json`.
