#!/usr/bin/env python3
"""Compare small summary stats for keyframe output folders."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare keyframe run output folders.")
    parser.add_argument("run_dirs", nargs="+", type=Path, help="Output folders containing keyframes.csv.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    for run_dir in args.run_dirs:
        csv_path = run_dir / "keyframes.csv"
        if not csv_path.exists():
            print(f"{run_dir}: missing keyframes.csv")
            continue

        df = pd.read_csv(csv_path)
        selected_count = len(df)
        mean_gap = df["timestamp_sec"].diff().dropna().mean() if selected_count > 1 else None
        median_novelty = df["novelty_distance"].dropna().median() if "novelty_distance" in df.columns else None

        print(f"Run: {run_dir}")
        print(f"  selected frames: {selected_count}")
        print(f"  mean time gap: {_format_optional(mean_gap)} sec")
        print(f"  median novelty distance: {_format_optional(median_novelty)}")
        print(f"  contact sheet: {run_dir / 'contact_sheet.jpg'}")
    return 0


def _format_optional(value: float | None) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"{value:.3f}"


if __name__ == "__main__":
    raise SystemExit(main())
