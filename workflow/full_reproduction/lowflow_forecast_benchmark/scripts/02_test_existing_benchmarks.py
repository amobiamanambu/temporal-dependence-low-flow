#!/usr/bin/env python3
"""Reproduce the archived heavy-tail model, persistence, q-bin, and adaptive baseline."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib.entry import endpoint_main  # noqa: E402

if __name__ == "__main__":
    endpoint_main(__doc__, [], "02_existing_benchmarks")

