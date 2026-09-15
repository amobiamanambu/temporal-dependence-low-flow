#!/usr/bin/env python3
"""Validate compact forecasts and write reproducible verification tables."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY / "src"))

from lowflow_coherence.analysis import (  # noqa: E402
    basin_verification,
    horizon_verification,
    load_sample_forecasts,
    reliability_summary,
    verify_same_marginal_scores,
)


def _write(frame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sample-directory",
        type=Path,
        default=REPOSITORY / "data" / "sample",
        help="Directory containing the compact example inputs.",
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=REPOSITORY / "outputs" / "tables",
        help="Directory for generated CSV tables.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    forecasts = load_sample_forecasts(args.sample_directory / "forecast_predictions_sample.csv.gz")
    outputs = {
        "sample_horizon_verification.csv": horizon_verification(forecasts),
        "sample_basin_verification.csv": basin_verification(forecasts),
        "sample_reliability.csv": reliability_summary(forecasts),
        "same_marginal_score_check.csv": verify_same_marginal_scores(
            args.sample_directory / "figure_03_same_marginal_scores.csv"
        ),
    }
    for name, frame in outputs.items():
        destination = args.output_directory / name
        _write(frame, destination)
        print(f"Wrote {destination} ({len(frame):,} rows)")

    score_check = outputs["same_marginal_score_check.csv"]
    maximum_difference = score_check["absolute_recalculation_difference"].max()
    if maximum_difference > 1e-9:
        raise RuntimeError(
            f"Same-marginal skill values did not reproduce (maximum difference {maximum_difference:g})"
        )
    print(
        f"Validated {len(forecasts):,} forecasts from "
        f"{forecasts['GAGE_ID'].nunique():,} gages; same-marginal score check passed."
    )


if __name__ == "__main__":
    main()
