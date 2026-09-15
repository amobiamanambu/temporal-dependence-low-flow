#!/usr/bin/env python3
"""Assign discharge record-quality tiers before discharge–climate matching.

Tier 1 requires at least 90% completeness and 30 years of record. Tier 2
requires at least 80% completeness and 20 years but does not meet Tier 1.
Other records are retained in the audit table as Tier 3.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from lib.common import normalize_gage_id, utc_now  # noqa: E402
from lib.discharge import parse_rdb_gzip  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--basin-inventory", type=Path, default=Path("basin_inventory.csv"))
    parser.add_argument(
        "--discharge-directory",
        type=Path,
        default=Path("usgs_dv_00060_1980_01_01_to_2025_12_31/raw_rdb_gz"),
    )
    parser.add_argument("--output-directory", type=Path, default=Path("quality_control"))
    parser.add_argument("--tier1-completeness", type=float, default=90.0)
    parser.add_argument("--tier2-completeness", type=float, default=80.0)
    parser.add_argument("--tier1-years", type=float, default=30.0)
    parser.add_argument("--tier2-years", type=float, default=20.0)
    return parser.parse_args()


def summarize_files(files: list[Path], wanted: set[str]) -> dict[str, dict[str, object]]:
    summaries: dict[str, dict[str, object]] = defaultdict(
        lambda: {"first_date": None, "last_date": None, "total_days": 0, "non_null_days": 0}
    )
    for index, path in enumerate(files, start=1):
        for gage, date, q_cfs, _qualifier, _source in parse_rdb_gzip(path):
            if gage not in wanted:
                continue
            item = summaries[gage]
            item["first_date"] = date if item["first_date"] is None else min(str(item["first_date"]), date)
            item["last_date"] = date if item["last_date"] is None else max(str(item["last_date"]), date)
            item["total_days"] = int(item["total_days"]) + 1
            if q_cfs is not None:
                item["non_null_days"] = int(item["non_null_days"]) + 1
        if index % 10 == 0 or index == len(files):
            print(f"Parsed {index}/{len(files)} compressed RDB batches", flush=True)
    return dict(summaries)


def assign_tier(row: pd.Series, args: argparse.Namespace) -> str:
    if not row["has_discharge_data"]:
        return "TIER3_NO_DATA"
    if (
        row["discharge_completeness"] >= args.tier1_completeness
        and row["record_years"] >= args.tier1_years
    ):
        return "TIER1_HIGH_QUALITY"
    if (
        row["discharge_completeness"] >= args.tier2_completeness
        and row["record_years"] >= args.tier2_years
    ):
        return "TIER2_ACCEPTABLE_QUALITY"
    return "TIER3_INSUFFICIENT_QUALITY"


def main() -> None:
    args = parse_args()
    if not args.basin_inventory.is_file():
        raise FileNotFoundError(args.basin_inventory)
    files = sorted(args.discharge_directory.glob("*.rdb.gz"))
    if not files:
        raise FileNotFoundError(f"No .rdb.gz files found in {args.discharge_directory}")
    inventory = pd.read_csv(args.basin_inventory, dtype={"GAGE_ID": "string"})
    required = {"basin_id", "GAGE_ID"}
    missing = sorted(required.difference(inventory.columns))
    if missing:
        raise ValueError(f"Basin inventory is missing columns: {missing}")
    inventory["GAGE_ID"] = inventory["GAGE_ID"].map(normalize_gage_id)
    summaries = summarize_files(files, set(inventory["GAGE_ID"]))

    records: list[dict[str, object]] = []
    for basin_id, gage in inventory[["basin_id", "GAGE_ID"]].itertuples(index=False, name=None):
        item = summaries.get(gage)
        if item is None or item["first_date"] is None or item["last_date"] is None:
            records.append(
                {
                    "basin_id": basin_id,
                    "GAGE_ID": gage,
                    "has_discharge_data": False,
                    "first_date": None,
                    "last_date": None,
                    "total_days": 0,
                    "non_null_days": 0,
                    "discharge_completeness": 0.0,
                    "record_years": 0.0,
                }
            )
            continue
        first = pd.Timestamp(item["first_date"])
        last = pd.Timestamp(item["last_date"])
        span_days = int((last - first).days + 1)
        records.append(
            {
                "basin_id": basin_id,
                "GAGE_ID": gage,
                "has_discharge_data": True,
                "first_date": first.date().isoformat(),
                "last_date": last.date().isoformat(),
                "total_days": int(item["total_days"]),
                "non_null_days": int(item["non_null_days"]),
                "discharge_completeness": 100.0 * int(item["non_null_days"]) / span_days,
                "record_years": (last - first).days / 365.25,
            }
        )
    quality = pd.DataFrame.from_records(records)
    quality["quality_tier"] = quality.apply(assign_tier, axis=1, args=(args,))
    quality["meets_tier1_discharge"] = quality["discharge_completeness"].ge(args.tier1_completeness)
    quality["meets_tier2_discharge"] = quality["discharge_completeness"].ge(args.tier2_completeness)
    quality["meets_tier1_length"] = quality["record_years"].ge(args.tier1_years)
    quality["meets_tier2_length"] = quality["record_years"].ge(args.tier2_years)

    args.output_directory.mkdir(parents=True, exist_ok=True)
    main_output = args.output_directory / "basin_quality_control_discharge_v2.csv"
    quality.to_csv(main_output, index=False)
    columns = ["basin_id", "GAGE_ID", "record_years", "discharge_completeness"]
    quality.loc[quality["quality_tier"].eq("TIER1_HIGH_QUALITY"), columns].to_csv(
        args.output_directory / "tier1_basin_list_v2.csv", index=False
    )
    quality.loc[
        quality["quality_tier"].isin(("TIER1_HIGH_QUALITY", "TIER2_ACCEPTABLE_QUALITY")), columns
    ].to_csv(args.output_directory / "tier2_basin_list_v2.csv", index=False)
    receipt = {
        "completed_utc": utc_now(),
        "source_files": len(files),
        "basins": len(quality),
        "tier_counts": quality["quality_tier"].value_counts().to_dict(),
        "criteria": {
            "tier1_minimum_completeness_percent": args.tier1_completeness,
            "tier1_minimum_record_years": args.tier1_years,
            "tier2_minimum_completeness_percent": args.tier2_completeness,
            "tier2_minimum_record_years": args.tier2_years,
        },
    }
    (args.output_directory / "discharge_qc_receipt.json").write_text(
        json.dumps(receipt, indent=2), encoding="utf-8"
    )
    print(f"Wrote {main_output}")
    print(json.dumps(receipt["tier_counts"], indent=2))


if __name__ == "__main__":
    main()
