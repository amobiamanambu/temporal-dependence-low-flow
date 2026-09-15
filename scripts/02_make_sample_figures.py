#!/usr/bin/env python3
"""Create compact verification and trajectory-anatomy figures."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

REPOSITORY = Path(__file__).resolve().parents[1]
MPL_CACHE = REPOSITORY / ".cache" / "matplotlib"
if "MPLCONFIGDIR" not in os.environ:
    MPL_CACHE.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(MPL_CACHE)
os.environ.setdefault("XDG_CACHE_HOME", str(REPOSITORY / ".cache"))
sys.path.insert(0, str(REPOSITORY / "src"))

import pandas as pd  # noqa: E402

from lowflow_coherence.plotting import (  # noqa: E402
    plot_sample_verification,
    plot_trajectory_anatomy,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sample-directory", type=Path, default=REPOSITORY / "data" / "sample"
    )
    parser.add_argument(
        "--table-directory", type=Path, default=REPOSITORY / "outputs" / "tables"
    )
    parser.add_argument(
        "--output-directory", type=Path, default=REPOSITORY / "outputs" / "figures"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    horizon = pd.read_csv(args.table_directory / "sample_horizon_verification.csv")
    basin = pd.read_csv(args.table_directory / "sample_basin_verification.csv", dtype={"GAGE_ID": str})
    reliability = pd.read_csv(args.table_directory / "sample_reliability.csv")
    plot_sample_verification(horizon, basin, reliability, args.output_directory)
    plot_trajectory_anatomy(args.sample_directory, args.output_directory)
    for name in ("sample_forecast_verification", "sample_trajectory_anatomy"):
        print(f"Wrote {args.output_directory / (name + '.png')}")
        print(f"Wrote {args.output_directory / (name + '.svg')}")


if __name__ == "__main__":
    main()
