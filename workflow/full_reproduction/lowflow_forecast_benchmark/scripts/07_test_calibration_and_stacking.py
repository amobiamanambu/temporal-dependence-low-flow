#!/usr/bin/env python3
"""Test held-out isotonic recalibration and regularized probability stacking."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib.entry import endpoint_main  # noqa: E402

if __name__ == "__main__":
    endpoint_main(
        __doc__, ["binary_idr", "gpd_tail_splice", "gpd_isotonic_calibrated",
                  "calibration_stack", "state_dependent_stack"],
        "07_calibration_stacking",
    )
