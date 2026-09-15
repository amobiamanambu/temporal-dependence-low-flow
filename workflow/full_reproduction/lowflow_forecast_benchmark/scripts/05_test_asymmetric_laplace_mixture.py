#!/usr/bin/env python3
"""Test a two-component asymmetric-Laplace standardized-innovation mixture."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib.entry import endpoint_main  # noqa: E402

if __name__ == "__main__":
    endpoint_main(__doc__, ["asymmetric_laplace_mixture"], "05_asymmetric_laplace")

