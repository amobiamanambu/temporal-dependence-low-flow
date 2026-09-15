#!/usr/bin/env python3
"""Test monotone isotonic distributional regression for threshold-event probabilities."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib.entry import endpoint_main  # noqa: E402

if __name__ == "__main__":
    endpoint_main(__doc__, ["binary_idr"], "03_binary_idr")

