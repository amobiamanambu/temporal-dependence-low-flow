#!/usr/bin/env python3
"""Download USGS daily mean discharge in restartable, checksummed batches.

The output layout and manifest are compatible with the continental preparation
workflow. Run this script from ``workflow/full_reproduction``.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import gzip
import hashlib
import math
from pathlib import Path
import time

import requests
import shapefile


SERVICE_URL = "https://waterservices.usgs.gov/nwis/dv/"


def normalize_gage(value: object) -> str:
    text = str(value).strip()
    if text.endswith(".0") and text[:-2].isdigit():
        text = text[:-2]
    digits = "".join(character for character in text if character.isdigit())
    return digits.zfill(8) if digits else ""


def sites_from_shapefile(path: Path) -> list[str]:
    reader = shapefile.Reader(str(path))
    fields = [field[0] for field in reader.fields[1:]]
    candidates = {name.upper(): index for index, name in enumerate(fields)}
    index = next(
        (candidates[name] for name in ("STAID", "GAGE_ID", "SITE_NO", "SITENO") if name in candidates),
        None,
    )
    if index is None:
        raise ValueError(f"No gage identifier field found in {path}; fields are {fields}")
    sites = {normalize_gage(record[index]) for record in reader.records()}
    return sorted(site for site in sites if site)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def inspect_rdb(path: Path) -> tuple[int, str | None, str | None, int]:
    rows = 0
    first_date: str | None = None
    last_date: str | None = None
    sites: set[str] = set()
    columns: list[str] | None = None
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as handle:
        for raw_line in handle:
            line = raw_line.rstrip("\n")
            if not line or line.startswith("#"):
                continue
            if line.startswith("agency_cd"):
                columns = line.split("\t")
                continue
            if columns is None or line.startswith("5s\t"):
                continue
            values = line.split("\t")
            if len(values) != len(columns):
                continue
            record = dict(zip(columns, values))
            date = record.get("datetime")
            site = normalize_gage(record.get("site_no", ""))
            if date:
                first_date = date if first_date is None else min(first_date, date)
                last_date = date if last_date is None else max(last_date, date)
            if site:
                sites.add(site)
            rows += 1
    return rows, first_date, last_date, len(sites)


def download_batch(
    session: requests.Session,
    sites: list[str],
    destination: Path,
    start_date: str,
    end_date: str,
    retries: int,
    timeout: int,
) -> None:
    parameters = {
        "format": "rdb",
        "sites": ",".join(sites),
        "startDT": start_date,
        "endDT": end_date,
        "parameterCd": "00060",
        "statCd": "00003",
        "siteStatus": "all",
    }
    temporary = destination.with_suffix(destination.suffix + ".part")
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            response = session.get(SERVICE_URL, params=parameters, timeout=timeout)
            response.raise_for_status()
            if "agency_cd" not in response.text:
                raise RuntimeError("USGS response did not contain an RDB table")
            with gzip.open(temporary, "wt", encoding="utf-8", newline="") as handle:
                handle.write(response.text)
            temporary.replace(destination)
            return
        except (OSError, requests.RequestException, RuntimeError) as error:
            last_error = error
            if temporary.exists():
                temporary.unlink()
            if attempt < retries:
                time.sleep(2.0 * attempt)
    raise RuntimeError(f"Download failed after {retries} attempts: {last_error}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shapefile", type=Path, default=Path("gagesII_9322_sept30_2011.shp"))
    parser.add_argument("--start-date", default="1980-01-01")
    parser.add_argument("--end-date", default="2025-12-31")
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--retries", type=int, default=6)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--pause-seconds", type=float, default=0.5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.shapefile.is_file():
        raise FileNotFoundError(args.shapefile)
    sites = sites_from_shapefile(args.shapefile)
    if not sites:
        raise RuntimeError("No station identifiers were found")
    date_tag = f"{args.start_date.replace('-', '_')}_to_{args.end_date.replace('-', '_')}"
    output = Path(f"usgs_dv_00060_{date_tag}")
    raw = output / "raw_rdb_gz"
    raw.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "manifest.csv"
    failures_path = output / "errors.csv"
    manifest_fields = [
        "batch_index", "raw_file", "site_count", "sites", "start_dt", "end_dt",
        "parameter_cd", "stat_cd", "download_utc", "http_status", "bytes_gz",
        "sha256_gz", "rows", "min_datetime", "max_datetime", "sites_with_data",
    ]
    completed: set[int] = set()
    if manifest_path.exists():
        with manifest_path.open(newline="", encoding="utf-8") as handle:
            completed = {int(row["batch_index"]) for row in csv.DictReader(handle)}

    session = requests.Session()
    session.headers.update({"User-Agent": "lowflow-temporal-dependence/0.1 (acamanambu@ua.edu)"})
    total = math.ceil(len(sites) / args.batch_size)
    for offset in range(0, len(sites), args.batch_size):
        batch = offset // args.batch_size + 1
        selection = sites[offset : offset + args.batch_size]
        filename = f"dv_00060_{date_tag}_batch_{batch:05d}.rdb.gz"
        destination = raw / filename
        if batch in completed and destination.is_file():
            print(f"[{batch}/{total}] retained {filename}")
            continue
        print(f"[{batch}/{total}] downloading {len(selection)} gages", flush=True)
        try:
            download_batch(
                session, selection, destination, args.start_date, args.end_date,
                args.retries, args.timeout,
            )
            rows, first_date, last_date, found = inspect_rdb(destination)
            record = {
                "batch_index": batch,
                "raw_file": filename,
                "site_count": len(selection),
                "sites": ",".join(selection),
                "start_dt": args.start_date,
                "end_dt": args.end_date,
                "parameter_cd": "00060",
                "stat_cd": "00003",
                "download_utc": datetime.now(timezone.utc).isoformat(),
                "http_status": 200,
                "bytes_gz": destination.stat().st_size,
                "sha256_gz": sha256(destination),
                "rows": rows,
                "min_datetime": first_date,
                "max_datetime": last_date,
                "sites_with_data": found,
            }
            write_header = not manifest_path.exists()
            with manifest_path.open("a", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=manifest_fields)
                if write_header:
                    writer.writeheader()
                writer.writerow(record)
        except Exception as error:
            write_header = not failures_path.exists()
            with failures_path.open("a", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                if write_header:
                    writer.writerow(["batch_index", "sites", "error"])
                writer.writerow([batch, ",".join(selection), repr(error)])
            print(f"[{batch}/{total}] failed: {error}")
        time.sleep(args.pause_seconds)


if __name__ == "__main__":
    main()
