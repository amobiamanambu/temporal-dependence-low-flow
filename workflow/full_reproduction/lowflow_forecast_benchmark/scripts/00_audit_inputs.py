#!/usr/bin/env python3
"""Audit inputs without modifying the established continental workflow."""

from __future__ import annotations

import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from lib.common import (  # noqa: E402
    ATTRIBUTES_FILE, BENCHMARK_ROOT, DAILY_ROOT, INVENTORY_FILE, PROJECT_ROOT,
    RESULTS_ROOT, STAGE18_METRICS, atomic_json, eligible_inventory, ensure_dirs,
    load_benchmark_config,
)


def main() -> None:
    ensure_dirs()
    config = load_benchmark_config()
    required = [
        PROJECT_ROOT / "scripts" / "18_test_operational_value.py",
        INVENTORY_FILE, ATTRIBUTES_FILE, STAGE18_METRICS, DAILY_ROOT,
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required inputs: " + ", ".join(missing))
    inventory = eligible_inventory()
    files_present = inventory["file"].map(lambda value: Path(value).exists())
    report = {
        "status": "pass" if files_present.all() else "fail",
        "project_root": str(PROJECT_ROOT),
        "benchmark_root": str(BENCHMARK_ROOT),
        "eligible_basins": int(len(inventory)),
        "daily_files_present": int(files_present.sum()),
        "daily_files_missing": int((~files_present).sum()),
        "ecoregions": int(inventory["spatial_group"].nunique()),
        "reference_basins": int(inventory["is_reference"].fillna(False).sum()),
        "tier_counts": inventory["quality_tier"].value_counts(dropna=False).to_dict(),
        "date_splits": {key: config[key] for key in [
            "fit_end", "calibration_start", "calibration_end",
            "evaluation_start", "evaluation_end",
        ]},
        "manuscript_files_required": False,
    }
    atomic_json(report, RESULTS_ROOT / "00_audit" / "input_audit.json")
    print(json.dumps(report, indent=2))
    if report["status"] != "pass":
        raise RuntimeError("Input audit failed")


if __name__ == "__main__":
    main()
