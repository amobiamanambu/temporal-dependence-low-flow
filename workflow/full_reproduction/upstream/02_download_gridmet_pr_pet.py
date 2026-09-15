#!/usr/bin/env python3
"""Download annual GridMET precipitation and potential-evapotranspiration files."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
from pathlib import Path
import time
import urllib.error
import urllib.request


BASE_URL = "https://www.northwestknowledge.net/metdata/data"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download(url: str, destination: Path, retries: int, timeout: int) -> None:
    temporary = destination.with_suffix(destination.suffix + ".part")
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "lowflow-temporal-dependence/0.1"})
            with urllib.request.urlopen(request, timeout=timeout) as response, temporary.open("wb") as handle:
                while block := response.read(1024 * 1024):
                    handle.write(block)
            if temporary.stat().st_size < 1_000_000:
                raise RuntimeError("downloaded file is unexpectedly small")
            temporary.replace(destination)
            return
        except (OSError, urllib.error.URLError, RuntimeError) as error:
            last_error = error
            if temporary.exists():
                temporary.unlink()
            if attempt < retries:
                time.sleep(2.0 * attempt)
    raise RuntimeError(f"Download failed after {retries} attempts: {last_error}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-year", type=int, default=1980)
    parser.add_argument("--end-year", type=int, default=2025)
    parser.add_argument("--output-directory", type=Path, default=Path("gridmet_data"))
    parser.add_argument("--variables", nargs="+", choices=("pr", "pet"), default=("pr", "pet"))
    parser.add_argument("--retries", type=int, default=4)
    parser.add_argument("--timeout", type=int, default=300)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.start_year > args.end_year:
        raise ValueError("start year cannot exceed end year")
    manifest_path = args.output_directory / "gridmet_pr_pet_manifest.csv"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    completed: set[tuple[str, int]] = set()
    if manifest_path.exists():
        with manifest_path.open(newline="", encoding="utf-8") as handle:
            completed = {(row["variable"], int(row["year"])) for row in csv.DictReader(handle)}
    fields = ("variable", "year", "file", "bytes", "sha256", "download_utc", "source_url")
    for variable in args.variables:
        directory = args.output_directory / variable
        directory.mkdir(parents=True, exist_ok=True)
        for year in range(args.start_year, args.end_year + 1):
            destination = directory / f"{variable}_{year}.nc"
            if (variable, year) in completed and destination.stat().st_size >= 1_000_000:
                print(f"retained {destination}")
                continue
            url = f"{BASE_URL}/{variable}_{year}.nc"
            print(f"downloading {url}", flush=True)
            download(url, destination, args.retries, args.timeout)
            record = {
                "variable": variable,
                "year": year,
                "file": destination.as_posix(),
                "bytes": destination.stat().st_size,
                "sha256": sha256(destination),
                "download_utc": datetime.now(timezone.utc).isoformat(),
                "source_url": url,
            }
            write_header = not manifest_path.exists()
            with manifest_path.open("a", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                if write_header:
                    writer.writeheader()
                writer.writerow(record)


if __name__ == "__main__":
    main()
