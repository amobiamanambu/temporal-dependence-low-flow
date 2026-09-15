#!/usr/bin/env python3
"""Test Student-t innovations and an extreme-value lower-tail GPD splice."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib.entry import endpoint_main  # noqa: E402

if __name__ == "__main__":
    endpoint_main(__doc__, ["student_t_innovations", "gpd_tail_splice"], "04_gpd_tail")

