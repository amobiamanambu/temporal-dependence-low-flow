#!/usr/bin/env python3
"""Check Python dependencies and compact-example input files."""

from __future__ import annotations

import importlib
import os
import platform
from pathlib import Path
import sys


REPOSITORY = Path(__file__).resolve().parents[1]
MPL_CACHE = REPOSITORY / ".cache" / "matplotlib"
if "MPLCONFIGDIR" not in os.environ:
    MPL_CACHE.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(MPL_CACHE)
os.environ.setdefault("XDG_CACHE_HOME", str(REPOSITORY / ".cache"))
SAMPLE = REPOSITORY / "data" / "sample"
REQUIRED_MODULES = ("numpy", "pandas", "matplotlib")
REQUIRED_FILES = (
    "forecast_predictions_sample.csv.gz",
    "figure_03_same_marginal_scores.csv",
    "figure_03_trajectory_anatomy_members.csv.gz",
    "figure_03_trajectory_anatomy_probabilities.csv",
    "figure_03_trajectory_anatomy_metadata.json",
)


def main() -> None:
    print(f"Python: {platform.python_version()}")
    if sys.version_info < (3, 10):
        raise RuntimeError("Python 3.10 or newer is required")
    missing_modules = []
    for name in REQUIRED_MODULES:
        try:
            module = importlib.import_module(name)
            print(f"{name}: {getattr(module, '__version__', 'installed')}")
        except ImportError:
            missing_modules.append(name)
    if missing_modules:
        raise RuntimeError(
            "Missing Python packages: " + ", ".join(missing_modules)
            + ". Run: python -m pip install -r requirements.txt"
        )
    missing_files = [name for name in REQUIRED_FILES if not (SAMPLE / name).is_file()]
    if missing_files:
        raise FileNotFoundError("Missing sample files: " + ", ".join(missing_files))
    print("Environment and sample inputs are ready.")


if __name__ == "__main__":
    main()
