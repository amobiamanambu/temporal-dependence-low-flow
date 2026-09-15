"""
GridMET Data Concatenation - FINAL VERSION
================================================================================
Concatenates annual GridMET NetCDF files into single combined files.

This script processes:
- Precipitation (pr): 46 files from 1980-2025  → pr_1980_2025_combined.nc
- PET: 46 files from 1980-2025 → pet_1980_2025_combined.nc

Output location: gridmet_data/

USAGE:
    python3 concatenate_gridmet_FINAL.py

NOTE: This requires substantial memory (~4-8 GB) due to large grid size.
If memory errors occur, consider using NCO tools (ncrcat) instead.
================================================================================
"""

import os
import glob
from netCDF4 import Dataset
from datetime import datetime
import gc

print("=" * 80)
print("GRIDMET DATA CONCATENATION")
print("=" * 80)

BASE_DIR = "gridmet_data"
PR_DIR = os.path.join(BASE_DIR, "pr")
PET_DIR = os.path.join(BASE_DIR, "pet")
PR_OUTPUT = os.path.join(BASE_DIR, "pr_1980_2025_combined.nc")
PET_OUTPUT = os.path.join(BASE_DIR, "pet_1980_2025_combined.nc")


def concatenate_files(input_dir, output_file, variable_name, file_prefix):
    """Concatenate NetCDF files along time dimension"""

    print(f"\nProcessing {file_prefix.upper()} files...")
    print("-" * 80)

    # Get sorted file list
    pattern = os.path.join(input_dir, f"{file_prefix}_*.nc")
    files = sorted(glob.glob(pattern))

    if not files:
        print(f"  ✗ No files found: {pattern}")
        return False

    print(f"  Found {len(files)} files")
    print(f"  Range: {os.path.basename(files[0])} to {os.path.basename(files[-1])}")

    # Get metadata from first file
    print(f"  Reading metadata...")
    with Dataset(files[0], 'r') as ds:
        lat = ds.variables['lat'][:]
        lon = ds.variables['lon'][:]
        nlat, nlon = len(lat), len(lon)

        # Get all attributes
        var_obj = ds.variables[variable_name]
        var_attrs = {a: var_obj.getncattr(a) for a in var_obj.ncattrs() if a != '_FillValue'}

        time_obj = ds.variables['day']
        time_units = time_obj.units
        time_calendar = getattr(time_obj, 'calendar', 'standard')

        lat_attrs = {a: ds.variables['lat'].getncattr(a) for a in ds.variables['lat'].ncattrs()}
        lon_attrs = {a: ds.variables['lon'].getncattr(a) for a in ds.variables['lon'].ncattrs()}

        # CRS if exists
        has_crs = 'crs' in ds.variables
        if has_crs:
            crs_data = ds.variables['crs'][:]
            crs_attrs = {a: ds.variables['crs'].getncattr(a) for a in ds.variables['crs'].ncattrs()}

        global_attrs = {a: ds.getncattr(a) for a in ds.ncattrs()}

    print(f"  Grid: {nlat} lat x {nlon} lon")

    # Count total time steps
    print(f"  Counting time steps...")
    total_time = 0
    for f in files:
        with Dataset(f, 'r') as ds:
            total_time += ds.dimensions['day'].size

    print(f"  Total time steps: {total_time:,}")
    print(f"  Estimated output size: ~{(total_time * nlat * nlon * 4) / (1024**3):.1f} GB")

    # Create output file
    print(f"\n  Creating output: {output_file}")

    with Dataset(output_file, 'w', format='NETCDF4') as out_ds:
        # Create dimensions
        out_ds.createDimension('day', total_time)
        out_ds.createDimension('lat', nlat)
        out_ds.createDimension('lon', nlon)
        if has_crs:
            out_ds.createDimension('crs', 1)

        # Create variables
        time_var = out_ds.createVariable('day', 'f8', ('day',), zlib=True)
        lat_var = out_ds.createVariable('lat', 'f4', ('lat',))
        lon_var = out_ds.createVariable('lon', 'f4', ('lon',))

        if has_crs:
            crs_var = out_ds.createVariable('crs', 'i4', ('crs',))
            crs_var[:] = crs_data
            for attr, val in crs_attrs.items():
                crs_var.setncattr(attr, val)

        # Create data variable with chunking
        data_var = out_ds.createVariable(
            variable_name, 'f4', ('day', 'lat', 'lon'),
            zlib=True, complevel=4,
            chunksizes=(min(30, total_time), nlat, nlon)
        )

        # Write coordinates
        lat_var[:] = lat
        lon_var[:] = lon

        # Set attributes
        time_var.units = time_units
        time_var.calendar = time_calendar

        for attr, val in lat_attrs.items():
            lat_var.setncattr(attr, val)
        for attr, val in lon_attrs.items():
            lon_var.setncattr(attr, val)
        for attr, val in var_attrs.items():
            data_var.setncattr(attr, val)
        for attr, val in global_attrs.items():
            out_ds.setncattr(attr, val)

        out_ds.history = f"Concatenated {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"

        # Write data file by file
        print(f"\n  Writing data:")
        time_idx = 0

        for i, file in enumerate(files, 1):
            year = os.path.basename(file).split('_')[1].split('.')[0]
            print(f"    [{i:2d}/{len(files)}] {year}...", end='', flush=True)

            with Dataset(file, 'r') as in_ds:
                ntime = in_ds.dimensions['day'].size

                # Write time and data
                time_var[time_idx:time_idx+ntime] = in_ds.variables['day'][:]
                data_var[time_idx:time_idx+ntime, :, :] = in_ds.variables[variable_name][:]

                time_idx += ntime
                print(f" {ntime:3d} days ✓")

            gc.collect()  # Force garbage collection

    # Report result
    size_gb = os.path.getsize(output_file) / (1024**3)
    print(f"\n  ✓ Created: {output_file}")
    print(f"  Size: {size_gb:.2f} GB")

    return True


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    print("\n1. Precipitation (pr)")
    print("=" * 80)
    success_pr = concatenate_files(PR_DIR, PR_OUTPUT, 'precipitation_amount', 'pr')

    print("\n\n2. Potential Evapotranspiration (pet)")
    print("=" * 80)
    success_pet = concatenate_files(PET_DIR, PET_OUTPUT, 'potential_evapotranspiration', 'pet')

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    if success_pr:
        print(f"✓ Precipitation: {PR_OUTPUT}")
    else:
        print(f"✗ Precipitation: FAILED")

    if success_pet:
        print(f"✓ PET: {PET_OUTPUT}")
    else:
        print(f"✗ PET: FAILED")

    print("\nOutput directory: gridmet_data/")
    print("=" * 80)


if __name__ == "__main__":
    main()
