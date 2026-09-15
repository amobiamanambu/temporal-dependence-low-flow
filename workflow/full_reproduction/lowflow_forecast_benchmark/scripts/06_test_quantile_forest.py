#!/usr/bin/env python3
"""Test a terminal-leaf empirical quantile-regression forest analogue."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib.entry import endpoint_main  # noqa: E402

if __name__ == "__main__":
    endpoint_main(__doc__, ["quantile_forest"], "06_quantile_forest")

