"""Simple frame quality metrics."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class FrameQuality:
    """Quality metrics used to avoid unusable frames."""

    sharpness: float
    brightness_mean: float
    brightness_std: float
    edge_density: float


def compute_quality(frame: np.ndarray) -> FrameQuality:
    """Compute sharpness, brightness, contrast, and edge density."""

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    brightness_mean = float(gray.mean())
    brightness_std = float(gray.std())

    edges = cv2.Canny(gray, threshold1=50, threshold2=150)
    edge_density = float(np.count_nonzero(edges) / edges.size)

    return FrameQuality(
        sharpness=sharpness,
        brightness_mean=brightness_mean,
        brightness_std=brightness_std,
        edge_density=edge_density,
    )


def is_unusable_quality(
    quality: FrameQuality,
    min_sharpness: float = 12.0,
    min_brightness: float = 12.0,
    max_brightness: float = 245.0,
    min_brightness_std: float = 3.0,
) -> bool:
    """Return True for frames that are very dark, blown out, or blurry."""

    return (
        quality.sharpness < min_sharpness
        or quality.brightness_mean < min_brightness
        or quality.brightness_mean > max_brightness
        or quality.brightness_std < min_brightness_std
    )


def quality_score(quality: FrameQuality) -> float:
    """Combine simple metrics into one score for fallback selection."""

    sharpness_score = min(quality.sharpness / 250.0, 1.0)
    contrast_score = min(quality.brightness_std / 64.0, 1.0)
    exposure_score = 1.0 - min(abs(quality.brightness_mean - 127.5) / 127.5, 1.0)
    edge_score = min(quality.edge_density / 0.08, 1.0)
    return (
        0.45 * sharpness_score
        + 0.25 * contrast_score
        + 0.20 * exposure_score
        + 0.10 * edge_score
    )
