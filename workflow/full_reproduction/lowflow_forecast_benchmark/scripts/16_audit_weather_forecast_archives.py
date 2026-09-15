#!/usr/bin/env python3
"""Record local and public weather-forecast archive availability and limits."""

from __future__ import annotations

import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from lib.common import BENCHMARK_ROOT, PROJECT_ROOT, RESULTS_ROOT, atomic_json  # noqa: E402


def main() -> None:
    patterns = ("gefs", "ecmwf", "forecast", "hindcast", "reforecast", "cfsv2", "s2s")
    local = []
    for path in PROJECT_ROOT.iterdir():
        if any(token in path.name.lower() for token in patterns) and path != BENCHMARK_ROOT:
            local.append(str(path))
    report = {
        "status": "complete",
        "local_archived_forecast_inputs": sorted(local),
        "local_ensemble_ready": bool(local),
        "public_archives": {
            "NOAA_GEFSv12_reforecast": {
                "period": "2000-2019 public AWS phase used for practical extraction",
                "ensemble": "5 daily members to 16 d; 11 weekly members to 35 d",
                "covers_90_days": False,
                "url": "https://registry.opendata.aws/noaa-gefs-reforecast/",
            },
            "NOAA_CFSv2": {
                "period": "reforecasts 1982-2011; operational archive from 2011",
                "ensemble": "45-day and seasonal/9-month integrations",
                "covers_90_days": True,
                "url": "https://www.ncei.noaa.gov/products/weather-climate-models/climate-forecast-system",
            },
            "ECMWF_S2S": {
                "period": "multi-center forecasts and on-the-fly reforecasts",
                "ensemble": "center-dependent; registration/API required",
                "covers_90_days": False,
                "url": "https://ecds.ecmwf.int/datasets/s2s-reforecasts",
            },
        },
        "decision": (
            "No genuine forecast forcing is currently local. Test future-unrestricted "
            "streamflow trajectories first; use the perfect-forcing oracle only to decide "
            "whether the large GEFS/CFS acquisition is scientifically justified."
        ),
        "manuscript_modified": False,
    }
    atomic_json(report, RESULTS_ROOT / "09_extended_forecast" / "weather_archive_audit.json")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
