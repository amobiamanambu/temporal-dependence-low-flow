"""
Extract Basin-Averaged Precipitation and PET from GridMET
================================================================================
The basin shapefile is reprojected to the GridMET coordinate system (WGS84).

INPUT:
- gridmet_data/pr_1980_2025_combined.nc
- gridmet_data/pet_1980_2025_combined.nc
- gridmet_data/GAGE_II_Basins.shp (NAD83 Albers - will be reprojected)

OUTPUT:
- basin_inventory.csv (Basin metadata with basin_id)
- basin_precipitation.csv (Daily precipitation by basin)
- basin_pet.csv (Daily PET by basin)

USAGE:
    python3 extract_basin_climate_FIXED.py

Estimated time: 30-60 minutes for all 9,067 basins
================================================================================
"""

import os
import numpy as np
import pandas as pd
from netCDF4 import Dataset
import geopandas as gpd
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("BASIN-AVERAGED GRIDMET CLIMATE EXTRACTION")
print("=" * 80)

# Check for required libraries
try:
    import rasterio
    from rasterio.features import geometry_mask
    from affine import Affine
except ImportError:
    print("\n✗ ERROR: rasterio not installed")
    print("Install with: pip install rasterio")
    exit(1)

# Configuration
BASIN_SHP = "gridmet_data/GAGE_II_Basins.shp"
PR_NC = "gridmet_data/pr_1980_2025_combined.nc"
PET_NC = "gridmet_data/pet_1980_2025_combined.nc"

print("\n1. Loading and reprojecting basin shapefile...")
print("-" * 80)

# Read basins with geopandas
basins_gdf = gpd.read_file(BASIN_SHP)
print(f"   Loaded {len(basins_gdf)} basins")
print(f"   Original CRS: {basins_gdf.crs}")

# Reproject to WGS84 (EPSG:4326) to match GridMET
print("   Reprojecting to WGS84 (EPSG:4326)...")
basins_wgs84 = basins_gdf.to_crs('EPSG:4326')
print(f"   ✓ Reprojected to: {basins_wgs84.crs}")

# Add basin_id column
basins_wgs84['basin_id'] = [f"BASIN_{i+1:05d}" for i in range(len(basins_wgs84))]

# Convert area to km² (original is in square meters)
basins_wgs84['area_km2'] = basins_wgs84['AREA'] / 1e6

# Create basin inventory
basin_inventory = basins_wgs84[['basin_id', 'GAGE_ID', 'area_km2', 'ecoregion']].copy()
basin_inventory.to_csv('basin_inventory.csv', index=False)
print(f"   ✓ Created basin_inventory.csv")

print("\n2. Loading GridMET metadata...")
print("-" * 80)

# Open NetCDF to get grid info
with Dataset(PR_NC, 'r') as ds:
    lats = ds.variables['lat'][:]
    lons = ds.variables['lon'][:]
    times = ds.variables['day'][:]
    time_units = ds.variables['day'].units

print(f"   Grid: {len(lats)} lat x {len(lons)} lon")
print(f"   Longitude range: {lons.min():.2f} to {lons.max():.2f}")
print(f"   Latitude range: {lats.min():.2f} to {lats.max():.2f}")
print(f"   Time steps: {len(times)}")

# Convert time to dates
from netCDF4 import num2date
dates = num2date(times, units=time_units)
date_strings = [d.strftime('%Y-%m-%d') for d in dates]
print(f"   Date range: {date_strings[0]} to {date_strings[-1]}")

# Create affine transform for the grid
# GridMET is a regular lat-lon grid
lon_min, lon_max = lons.min(), lons.max()
lat_min, lat_max = lats.min(), lats.max()
lon_res = (lon_max - lon_min) / (len(lons) - 1)
lat_res = (lat_max - lat_min) / (len(lats) - 1)

# Affine transform (note: y resolution is negative for top-down)
transform = Affine.translation(lon_min - lon_res/2, lat_max + lat_res/2) * Affine.scale(lon_res, -lat_res)

print(f"   Grid resolution: {lon_res:.4f}° lon x {lat_res:.4f}° lat")

# Calculate grid cell areas in km²
lat_rad = np.deg2rad(lats)
earth_radius = 6371.0  # km
lat_spacing_km = earth_radius * np.deg2rad(abs(lat_res))
cell_areas = np.zeros((len(lats), len(lons)))
for i, lat in enumerate(lats):
    lon_spacing_km = earth_radius * np.cos(np.deg2rad(lat)) * np.deg2rad(abs(lon_res))
    cell_areas[i, :] = lat_spacing_km * lon_spacing_km

print(f"   Grid cell area range: {cell_areas.min():.1f} to {cell_areas.max():.1f} km²")

print("\n3. Pre-computing basin masks...")
print("-" * 80)

# Pre-compute mask for each basin
basin_masks = []
basin_weights = []
basins_with_no_cells = []

try:
    for idx, basin in basins_wgs84.iterrows():
        if (idx + 1) % 500 == 0:
            print(f"   Computing mask {idx+1}/{len(basins_wgs84)}...")

        try:
            # Create mask for this basin
            geom = [basin.geometry]
            mask = geometry_mask(
                geom,
                out_shape=(len(lats), len(lons)),
                transform=transform,
                invert=True  # True where geometry is
            )

            if mask.sum() == 0:
                # No cells in basin - basin might be too small or outside grid
                basins_with_no_cells.append(basin['GAGE_ID'])
                basin_masks.append(None)
                basin_weights.append(None)
                continue

            # Calculate weights based on cell areas in km²
            basin_cell_areas = cell_areas[mask]
            weights = basin_cell_areas / basin_cell_areas.sum()

            basin_masks.append(mask)
            basin_weights.append(weights)

        except Exception as e:
            print(f"   Warning: Error processing basin {idx+1} ({basin.get('GAGE_ID', 'unknown')}): {e}")
            basins_with_no_cells.append(basin.get('GAGE_ID', 'unknown'))
            basin_masks.append(None)
            basin_weights.append(None)
            continue

    n_valid = len([m for m in basin_masks if m is not None])
    print(f"   ✓ Computed {n_valid} valid basin masks out of {len(basins_wgs84)} basins")

    if len(basins_with_no_cells) > 0:
        print(f"   ⚠ Warning: {len(basins_with_no_cells)} basins have no grid cells")
        if len(basins_with_no_cells) <= 20:
            print(f"      Basins without cells: {', '.join(basins_with_no_cells[:20])}")
        else:
            print(f"      First 20 basins without cells: {', '.join(basins_with_no_cells[:20])}")

    if n_valid == 0:
        print("   ✗ ERROR: No valid basin masks were created!")
        print("   This means basins still don't overlap with the GridMET grid.")
        exit(1)

except Exception as e:
    print(f"   ✗ ERROR during mask computation: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print("\n4. Extracting precipitation data...")
print("-" * 80)

pr_results = []

try:
    with Dataset(PR_NC, 'r') as ds:
        pr_var = ds.variables['precipitation_amount']

        # Process in time chunks
        chunk_size = 365  # One year at a time
        n_chunks = (len(times) + chunk_size - 1) // chunk_size

        for chunk_idx in range(n_chunks):
            start_t = chunk_idx * chunk_size
            end_t = min(start_t + chunk_size, len(times))

            print(f"   Processing time {start_t+1}-{end_t} of {len(times)}...")

            # Read chunk
            pr_chunk = pr_var[start_t:end_t, :, :]

            # Process each basin
            for basin_idx, (mask, weights) in enumerate(zip(basin_masks, basin_weights)):
                if mask is None:
                    continue

                basin_id = basins_wgs84.iloc[basin_idx]['basin_id']
                gage_id = basins_wgs84.iloc[basin_idx]['GAGE_ID']

                # Extract values for each time step in chunk
                for t in range(end_t - start_t):
                    values = pr_chunk[t, :, :][mask]
                    weighted_mean = np.average(values, weights=weights)

                    pr_results.append({
                        'basin_id': basin_id,
                        'GAGE_ID': gage_id,
                        'date': date_strings[start_t + t],
                        'precipitation_mm': weighted_mean
                    })

    print(f"   ✓ Extracted {len(pr_results):,} precipitation records")

except Exception as e:
    print(f"   ✗ ERROR extracting precipitation: {e}")
    import traceback
    traceback.print_exc()

print("\n5. Extracting PET data...")
print("-" * 80)

pet_results = []

try:
    with Dataset(PET_NC, 'r') as ds:
        pet_var = ds.variables['potential_evapotranspiration']

        for chunk_idx in range(n_chunks):
            start_t = chunk_idx * chunk_size
            end_t = min(start_t + chunk_size, len(times))

            print(f"   Processing time {start_t+1}-{end_t} of {len(times)}...")

            # Read chunk
            pet_chunk = pet_var[start_t:end_t, :, :]

            # Process each basin
            for basin_idx, (mask, weights) in enumerate(zip(basin_masks, basin_weights)):
                if mask is None:
                    continue

                basin_id = basins_wgs84.iloc[basin_idx]['basin_id']
                gage_id = basins_wgs84.iloc[basin_idx]['GAGE_ID']

                # Extract values
                for t in range(end_t - start_t):
                    values = pet_chunk[t, :, :][mask]
                    weighted_mean = np.average(values, weights=weights)

                    pet_results.append({
                        'basin_id': basin_id,
                        'GAGE_ID': gage_id,
                        'date': date_strings[start_t + t],
                        'pet_mm': weighted_mean
                    })

    print(f"   ✓ Extracted {len(pet_results):,} PET records")

except Exception as e:
    print(f"   ✗ ERROR extracting PET: {e}")
    import traceback
    traceback.print_exc()

print("\n6. Saving results...")
print("-" * 80)

if len(pr_results) == 0:
    print("   ✗ ERROR: No precipitation data was extracted!")
    print("   Check the errors above. The script may have failed during extraction.")
    exit(1)

if len(pet_results) == 0:
    print("   ✗ ERROR: No PET data was extracted!")
    print("   Check the errors above. The script may have failed during extraction.")
    exit(1)

# Create DataFrames
pr_df = pd.DataFrame(pr_results)
pet_df = pd.DataFrame(pet_results)

print(f"   Created precipitation dataframe: {len(pr_df):,} records")
print(f"   Created PET dataframe: {len(pet_df):,} records")

# Save
print("   Saving basin_precipitation.csv...")
pr_df.to_csv('basin_precipitation.csv', index=False)
print(f"   ✓ Saved basin_precipitation.csv: {len(pr_df):,} records")

print("   Saving basin_pet.csv...")
pet_df.to_csv('basin_pet.csv', index=False)
print(f"   ✓ Saved basin_pet.csv: {len(pet_df):,} records")

# Summary
print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)

n_basins_pr = pr_df['basin_id'].nunique()
n_basins_pet = pet_df['basin_id'].nunique()

print(f"\nBasins with data:")
print(f"  Precipitation: {n_basins_pr}")
print(f"  PET: {n_basins_pet}")
print(f"  No data: {len(basins_with_no_cells)}")

print(f"\nDate range: {pr_df['date'].min()} to {pr_df['date'].max()}")
print(f"Time steps: {pr_df['date'].nunique()}")

print(f"\nPrecipitation: {pr_df['precipitation_mm'].mean():.2f} mm/day (mean)")
print(f"PET: {pet_df['pet_mm'].mean():.2f} mm/day (mean)")

print("\nOUTPUT FILES:")
print("  1. basin_inventory.csv     - Basin metadata (join key)")
print("  2. basin_precipitation.csv - Daily precip by basin")
print("  3. basin_pet.csv           - Daily PET by basin")

print("\nDATA FORMAT:")
print("  - Each row = one basin-date combination")
print("  - Join field: 'basin_id' or 'GAGE_ID'")
print("  - Ready for SPI/SPEI calculation")
print("  - Easy to merge with shapefile")

print("\n" + "=" * 80)
print("COMPLETE!")
print("=" * 80)
