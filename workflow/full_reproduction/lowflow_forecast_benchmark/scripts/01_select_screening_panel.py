#!/usr/bin/env python3
"""Materialize the frozen 192-basin development panel."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from lib.common import (  # noqa: E402
    RESULTS_ROOT, atomic_csv, atomic_json, eligible_inventory, ensure_dirs,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--basins", type=int, default=192)
    args = parser.parse_args()
    if args.basins != 192:
        raise ValueError("The archived development panel is fixed at 192 basins")
    ensure_dirs()
    panel = eligible_inventory()
    output = RESULTS_ROOT / "01_panel" / "candidate_screening_panel.csv"
    atomic_csv(panel, output)
    receipt = {
        "status": "complete", "basins": int(len(panel)),
        "ecoregions": int(panel["spatial_group"].nunique()),
        "reference_basins": int(panel["is_reference"].fillna(False).sum()),
        "panel_is_frozen": True,
        "selection_uses_forecast_scores_or_predictions": False,
        "output": str(output),
    }
    atomic_json(receipt, RESULTS_ROOT / "01_panel" / "_SUCCESS.json")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
