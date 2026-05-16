"""Compact classical visual descriptors for seabed frame novelty."""

from __future__ import annotations

import cv2
import numpy as np


def l2_normalize(vector: np.ndarray) -> np.ndarray:
    """Return a float32 vector with unit L2 norm when possible."""

    vector = vector.astype(np.float32, copy=False)
    norm = float(np.linalg.norm(vector))
    if norm == 0.0:
        return vector
    return vector / norm


def compute_visual_descriptor(frame: np.ndarray, size: tuple[int, int] = (160, 90)) -> np.ndarray:
    """Build a normalized color and texture descriptor for one frame."""

    small = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

    color_hist = cv2.calcHist(
        [hsv],
        channels=[0, 1],
        mask=None,
        histSize=[16, 8],
        ranges=[0, 180, 0, 256],
    ).flatten()
    color_hist = color_hist / max(float(color_hist.sum()), 1.0)

    gray_hist = cv2.calcHist(
        [gray],
        channels=[0],
        mask=None,
        histSize=[16],
        ranges=[0, 256],
    ).flatten()
    gray_hist = gray_hist / max(float(gray_hist.sum()), 1.0)

    edges = cv2.Canny(gray, threshold1=50, threshold2=150)
    edge_density = np.count_nonzero(edges) / edges.size
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    texture_stats = np.array(
        [
            edge_density,
            gray.mean() / 255.0,
            gray.std() / 255.0,
            min(laplacian_var / 1000.0, 1.0),
        ],
        dtype=np.float32,
    )

    descriptor = np.concatenate(
        [
            color_hist.astype(np.float32),
            gray_hist.astype(np.float32),
            texture_stats,
        ]
    )
    return l2_normalize(descriptor)


def cosine_distance(left: np.ndarray, right: np.ndarray) -> float:
    """Compute cosine distance between two descriptor vectors."""

    left_norm = float(np.linalg.norm(left))
    right_norm = float(np.linalg.norm(right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 1.0
    similarity = float(np.dot(left, right) / (left_norm * right_norm))
    similarity = max(-1.0, min(1.0, similarity))
    return 1.0 - similarity
