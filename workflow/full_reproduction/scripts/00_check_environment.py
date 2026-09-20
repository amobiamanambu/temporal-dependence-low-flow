#!/usr/bin/env python3
"""Preflight: check Python packages, required local inputs, and available disk space."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from lib.common import (
    DEFAULT_CONFIG, dependency_versions, load_config, output_root, resolve_path, utc_now,
)


PACKAGES = [
    "numpy", "pandas", "scipy", "netCDF4", "geopandas", "rasterio", "shapely",
    "pyproj", "matplotlib", "scikit-learn", "pyshp",
]


def main() -> None:
    cfg = load_config(DEFAULT_CONFIG)
    paths = {name: resolve_path(cfg, value) for name, value in cfg["inputs"].items()}
    versions = dependency_versions(PACKAGES)
    missing_packages = [name for name, version in versions.items() if version == "NOT INSTALLED"]
    missing_inputs = [name for name, path in paths.items() if not path.exists()]
    usage = shutil.disk_usage(Path(cfg["_project_root"]))
    report = {
        "checked_utc": utc_now(),
        "config": str(DEFAULT_CONFIG),
        "packages": versions,
        "input_paths": {name: {"path": str(path), "exists": path.exists()} for name, path in paths.items()},
        "missing_packages": missing_packages,
        "missing_inputs": missing_inputs,
        "free_disk_gb": round(usage.free / 1e9, 2),
        "recommended_free_disk_gb": 150,
        "ready": not missing_packages and not missing_inputs and usage.free >= 150e9,
    }
    destination = output_root(cfg) / "00_environment"
    destination.mkdir(parents=True, exist_ok=True)
    output = destination / "environment_report.json"
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"\nSaved: {output}")
    if missing_packages:
        raise SystemExit("Missing packages. Install scripts/requirements.txt, then rerun preflight.")
    if missing_inputs:
        raise SystemExit(f"Missing required inputs: {', '.join(missing_inputs)}")
    if usage.free < 150e9:
        raise SystemExit("Less than the recommended 150 GB free disk space is available.")


if __name__ == "__main__":
    main()
