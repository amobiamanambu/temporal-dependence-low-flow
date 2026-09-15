"""Command-line helper shared by the method-specific endpoint scripts."""

from __future__ import annotations

import argparse
import json

import pandas as pd

from .common import RESULTS_ROOT, load_benchmark_config
from .endpoint import run_endpoint


def endpoint_main(description: str, methods: list[str], default_subdir: str) -> None:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--panel", default=str(
        RESULTS_ROOT / "01_panel" / "candidate_screening_panel.csv"
    ))
    parser.add_argument("--output-subdir", default=default_subdir)
    parser.add_argument("--all-leads", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--max-basins", type=int)
    args = parser.parse_args()
    panel = pd.read_csv(args.panel, dtype={"GAGE_ID": str})
    if args.max_basins:
        panel = panel.head(int(args.max_basins)).copy()
    config = load_benchmark_config()
    leads = config["lead_days"] if args.all_leads else config["sentinel_leads"]
    receipt = run_endpoint(
        panel, methods, [int(value) for value in leads], int(args.workers),
        str(args.output_subdir), bool(args.force),
    )
    print(json.dumps(receipt, indent=2))
