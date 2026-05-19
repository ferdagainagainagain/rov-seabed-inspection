#!/usr/bin/env python3
"""Run the full ROV Seabed Inspection pipeline (Stage 1 -> 2 -> 3) on one video."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"

STAGES = [
    ("Stage 1 - Keyframe selection", SCRIPTS_DIR / "select_keyframes.py"),
    ("Stage 2 - VLM annotation", SCRIPTS_DIR / "analyze_keyframes_vlm.py"),
    ("Stage 3 - Final report", SCRIPTS_DIR / "synthesize_report.py"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the full ROV inspection pipeline on one video using a YAML config.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        required=True,
        type=Path,
        help="YAML config file (one per video). See configs/video1.yaml for an example.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config_path = args.config.expanduser()
    if not config_path.is_file():
        print(f"Error: config file does not exist: {config_path}", file=sys.stderr)
        return 1

    for label, script_path in STAGES:
        print(f"\n=== {label} ===", flush=True)
        result = subprocess.run(
            [sys.executable, str(script_path), "--config", str(config_path)],
            check=False,
        )
        if result.returncode != 0:
            print(f"\nError: {label} failed (exit code {result.returncode}).", file=sys.stderr)
            return result.returncode

    print("\n=== Pipeline complete ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
