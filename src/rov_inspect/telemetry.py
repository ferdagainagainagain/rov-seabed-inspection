"""Depth telemetry helpers for optional underwater frame filtering."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

TIME_SEC_COLUMN = "_time_sec"
DEPTH_M_COLUMN = "_depth_m"


def load_depth_log(path: Path) -> pd.DataFrame:
    """Load a depth CSV and add normalized time/depth columns."""

    if not path.exists():
        raise FileNotFoundError(f"Depth CSV does not exist: {path}")
    if not path.is_file():
        raise FileNotFoundError(f"Depth CSV path is not a file: {path}")

    df = pd.read_csv(path)
    time_column, depth_column = infer_time_and_depth_columns(df)
    normalized = df.copy()
    normalized[TIME_SEC_COLUMN] = _to_seconds(normalized[time_column])
    normalized[DEPTH_M_COLUMN] = _to_depth_meters(normalized[depth_column])
    normalized = normalized.dropna(subset=[TIME_SEC_COLUMN, DEPTH_M_COLUMN])
    normalized = normalized.sort_values(TIME_SEC_COLUMN).reset_index(drop=True)
    normalized.attrs["time_column"] = time_column
    normalized.attrs["depth_column"] = depth_column
    return normalized


def infer_time_and_depth_columns(df: pd.DataFrame) -> tuple[str, str]:
    """Infer timestamp and depth columns from a telemetry table."""

    columns = list(df.columns)
    lower_to_original = {column.lower().strip(): column for column in columns}

    time_candidates = ["timestamp_sec", "time_sec", "elapsed_sec", "seconds", "timestamp", "time"]
    time_column = next(
        (lower_to_original[name] for name in time_candidates if name in lower_to_original),
        None,
    )
    if time_column is None:
        time_column = next(
            (column for column in columns if "time" in column.lower()),
            None,
        )

    depth_column = next(
        (column for column in columns if "depth" in column.lower()),
        None,
    )

    if time_column is None:
        raise ValueError(f"Could not infer a time column from: {columns}")
    if depth_column is None:
        raise ValueError(f"Could not infer a depth column from: {columns}")
    return time_column, depth_column


def depth_at_timestamp(df: pd.DataFrame, timestamp_sec: float) -> float | None:
    """Interpolate depth in meters at a video timestamp."""

    if df.empty:
        return None

    times = df[TIME_SEC_COLUMN].to_numpy(dtype=float)
    depths = df[DEPTH_M_COLUMN].to_numpy(dtype=float)
    if timestamp_sec < times[0] or timestamp_sec > times[-1]:
        return None

    return float(np.interp(timestamp_sec, times, depths))


def depth_eligibility(
    timestamp_sec: float,
    df: pd.DataFrame,
    min_depth_m: float,
    stable_window_sec: float,
    allow_missing_depth: bool = False,
) -> tuple[bool, float | None, str]:
    """Return depth eligibility, interpolated depth, and a short reason."""

    depth_m = depth_at_timestamp(df, timestamp_sec)
    if depth_m is None:
        if allow_missing_depth:
            return True, None, "missing_depth_allowed"
        return False, None, "missing_depth"

    if depth_m < min_depth_m:
        return False, depth_m, "below_min_depth"

    if stable_window_sec <= 0:
        return True, depth_m, "depth_ok"

    start_time = timestamp_sec - stable_window_sec
    start_depth = depth_at_timestamp(df, start_time)
    if start_depth is None:
        if allow_missing_depth:
            return True, depth_m, "stable_window_missing_depth_allowed"
        return False, depth_m, "stable_window_missing_depth"

    times = df[TIME_SEC_COLUMN].to_numpy(dtype=float)
    depths = df[DEPTH_M_COLUMN].to_numpy(dtype=float)
    in_window = (times >= start_time) & (times <= timestamp_sec)
    window_depths = np.concatenate(([start_depth], depths[in_window], [depth_m]))
    if np.any(window_depths < min_depth_m):
        return False, depth_m, "not_stably_underwater"

    return True, depth_m, "depth_ok"


def _to_seconds(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().sum() == len(series):
        return numeric.astype(float)

    parsed = pd.to_datetime(series, format="%Y.%m.%d %H:%M:%S:%f", errors="coerce")
    if parsed.isna().all():
        parsed = pd.to_datetime(series, errors="coerce")
    if parsed.isna().all():
        raise ValueError("Could not parse telemetry timestamps")

    first_timestamp = parsed.dropna().iloc[0]
    return (parsed - first_timestamp).dt.total_seconds()


def _to_depth_meters(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return series.astype(float)

    cleaned = (
        series.astype(str)
        .str.strip()
        .str.replace(",", ".", regex=False)
        .str.extract(r"([-+]?\d*\.?\d+)", expand=False)
    )
    return pd.to_numeric(cleaned, errors="coerce")
