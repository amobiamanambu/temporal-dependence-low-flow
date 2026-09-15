#!/usr/bin/env python3
"""Audit repository hygiene and optionally write a SHA-256 file manifest."""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "REPOSITORY_MANIFEST.csv"
AUDIT = ROOT / "REPOSITORY_AUDIT.json"
MAX_TRACKED_BYTES = 25 * 1024 * 1024
SKIP_PARTS = {".git", ".cache", ".venv", "__pycache__"}
FORBIDDEN_SUFFIXES = {
    ".nc", ".nc4", ".grib", ".grb", ".sqlite", ".db", ".shp", ".shx",
    ".dbf", ".doc", ".docx", ".odt",
}
PRIVATE_AUTHORING_NAMES = {
    "29_build_wrr_manuscript.py",
    "33_build_wrr_manuscript_v2.py",
    "37_prepare_peer_review_package.py",
    "38_audit_submission.py",
    "40_format_joh_tables.py",
    "integrate_figure_06_selected_basins.py",
}
PRIVATE_AUTHORING_PATTERNS = (
    re.compile(r"wrr\.add_body\("),
    re.compile(r"from\s+docx\s+import\s+Document"),
    re.compile(r"import\s+docx"),
)
MANUSCRIPT_SCAN_EXCLUSIONS = {
    # This release builder contains the forbidden strings only as guardrails
    # used to reject private authoring material from the Zenodo code snapshot.
    "42_build_zenodo_deposit.py",
}
TEXT_SUFFIXES = {".py", ".md", ".json", ".toml", ".yml", ".yaml", ".cff", ".txt", ".csv"}
SECRET_PATTERNS = (
    re.compile(r"sk-[A-Za-z0-9]{20,}"),
    re.compile(r"(?i)(api[_-]?key|secret|token)\s*=\s*['\"][^'\"]+['\"]"),
)
PERSONAL_PATH = re.compile(r"/Users/[A-Za-z0-9._-]+/|[A-Za-z]:\\Users\\")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tracked_files() -> list[Path]:
    files = []
    for path in ROOT.rglob("*"):
        if not path.is_file() or any(part in SKIP_PARTS for part in path.parts):
            continue
        if path in {MANIFEST, AUDIT}:
            continue
        files.append(path)
    return sorted(files)


def audit(files: list[Path]) -> dict[str, object]:
    problems: list[str] = []
    python_files = 0
    for path in files:
        relative = path.relative_to(ROOT).as_posix()
        if path.name in PRIVATE_AUTHORING_NAMES:
            problems.append(f"private manuscript-authoring file found: {relative}")
        if path.stat().st_size > MAX_TRACKED_BYTES:
            problems.append(f"file exceeds 25 MiB: {relative}")
        if path.suffix.lower() in FORBIDDEN_SUFFIXES:
            problems.append(f"raw/large data type should not be tracked: {relative}")
        if path.suffix.lower() in TEXT_SUFFIXES:
            try:
                text = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                problems.append(f"text file is not UTF-8: {relative}")
                continue
            if PERSONAL_PATH.search(text):
                problems.append(f"personal absolute path found: {relative}")
            for pattern in SECRET_PATTERNS:
                if pattern.search(text):
                    problems.append(f"possible credential found: {relative}")
            if (
                path.suffix.lower() == ".py"
                and path.name not in MANUSCRIPT_SCAN_EXCLUSIONS
            ):
                for pattern in PRIVATE_AUTHORING_PATTERNS:
                    if pattern.search(text):
                        problems.append(
                            f"possible manuscript-authoring code found: {relative}"
                        )
                        break
            if path.suffix.lower() == ".py":
                python_files += 1
                try:
                    ast.parse(text, filename=relative)
                except SyntaxError as error:
                    problems.append(f"Python syntax error in {relative}: {error}")
    return {
        "status": "pass" if not problems else "fail",
        "files_audited": len(files),
        "python_files_parsed": python_files,
        "largest_file_bytes": max((path.stat().st_size for path in files), default=0),
        "problems": problems,
    }


def write_manifest(files: list[Path]) -> None:
    temporary = MANIFEST.with_suffix(".csv.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("path", "bytes", "sha256"))
        writer.writeheader()
        for path in files:
            writer.writerow(
                {
                    "path": path.relative_to(ROOT).as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": sha256(path),
                }
            )
    temporary.replace(MANIFEST)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-manifest", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    files = tracked_files()
    report = audit(files)
    if args.write_manifest and report["status"] == "pass":
        write_manifest(files)
        report["manifest"] = MANIFEST.name
    AUDIT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if report["status"] != "pass":
        raise RuntimeError("Repository audit failed")


if __name__ == "__main__":
    main()
