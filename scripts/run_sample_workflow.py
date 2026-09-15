#!/usr/bin/env python3
"""Run the complete compact example in numbered order."""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys


SCRIPTS = Path(__file__).resolve().parent


def main() -> None:
    for name in (
        "00_check_environment.py",
        "01_run_sample_analysis.py",
        "02_make_sample_figures.py",
    ):
        command = [sys.executable, str(SCRIPTS / name)]
        print(f"\nRunning {' '.join(command)}", flush=True)
        subprocess.run(command, check=True)
    print("\nSample workflow completed successfully.")


if __name__ == "__main__":
    main()
