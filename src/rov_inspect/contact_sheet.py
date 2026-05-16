"""Create a simple labeled contact sheet for selected keyframes."""

from __future__ import annotations

from math import ceil
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np


def make_contact_sheet(
    frames: Sequence[np.ndarray],
    labels: Sequence[str],
    output_path: Path,
    columns: int = 4,
    thumb_width: int = 260,
    label_height: int = 28,
) -> Path:
    """Save a JPEG contact sheet with timestamp labels."""

    if len(frames) != len(labels):
        raise ValueError("frames and labels must have the same length")
    if not frames:
        raise ValueError("Cannot create a contact sheet with no frames")
    if columns <= 0:
        raise ValueError("columns must be greater than 0")

    thumbnails = [_make_tile(frame, label, thumb_width, label_height) for frame, label in zip(frames, labels)]
    columns = min(columns, len(thumbnails))
    rows = int(ceil(len(thumbnails) / columns))
    tile_height, tile_width = thumbnails[0].shape[:2]

    sheet = np.full(
        (rows * tile_height, columns * tile_width, 3),
        fill_value=245,
        dtype=np.uint8,
    )

    for index, tile in enumerate(thumbnails):
        row = index // columns
        column = index % columns
        y0 = row * tile_height
        x0 = column * tile_width
        sheet[y0 : y0 + tile_height, x0 : x0 + tile_width] = tile

    output_path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(output_path), sheet, [cv2.IMWRITE_JPEG_QUALITY, 92])
    if not ok:
        raise IOError(f"Could not write contact sheet: {output_path}")
    return output_path


def _make_tile(frame: np.ndarray, label: str, thumb_width: int, label_height: int) -> np.ndarray:
    height, width = frame.shape[:2]
    scale = thumb_width / width
    thumb_height = max(1, int(round(height * scale)))
    resized = cv2.resize(frame, (thumb_width, thumb_height), interpolation=cv2.INTER_AREA)

    tile = np.full((thumb_height + label_height, thumb_width, 3), 255, dtype=np.uint8)
    tile[:thumb_height, :thumb_width] = resized
    cv2.putText(
        tile,
        label,
        org=(8, thumb_height + 19),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=0.48,
        color=(30, 30, 30),
        thickness=1,
        lineType=cv2.LINE_AA,
    )
    return tile
