"""Keyframe selection based on visual novelty against the last selected frame."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import pandas as pd

from rov_inspect.descriptors import DistanceBreakdown, descriptor_distance
from rov_inspect.features import compute_visual_descriptor
from rov_inspect.quality import FrameQuality, compute_quality, is_unusable_quality, quality_score
from rov_inspect.video_io import SampledFrame


@dataclass(frozen=True)
class FrameCandidate:
    """A sampled frame with quality metrics and visual descriptors."""

    frame_index: int
    timestamp_sec: float
    image: np.ndarray
    quality: FrameQuality
    classical_descriptor: np.ndarray
    dino_descriptor: np.ndarray | None = None
    depth_m: float | None = None
    depth_eligible: bool | None = None
    telemetry_reason: str = "not_used"


@dataclass(frozen=True)
class SelectedFrame:
    """A keyframe kept by the selector plus the reason and metrics."""

    candidate: FrameCandidate
    reason: str
    distances: DistanceBreakdown
    descriptor_backend: str
    novelty_threshold_used: float
    adaptive_threshold_enabled: bool


@dataclass(frozen=True)
class SelectionResult:
    """Selected keyframes and counters describing the input stream."""

    selected: list[SelectedFrame]
    sampled_count: int
    skipped_by_quality: int


def make_candidate(
    sampled_frame: SampledFrame,
    depth_m: float | None = None,
    depth_eligible: bool | None = None,
    telemetry_reason: str = "not_used",
    dino_descriptor: np.ndarray | None = None,
) -> FrameCandidate:
    """Compute quality metrics and the classical descriptor for one frame."""

    return FrameCandidate(
        frame_index=sampled_frame.frame_index,
        timestamp_sec=sampled_frame.timestamp_sec,
        image=sampled_frame.image,
        quality=compute_quality(sampled_frame.image),
        classical_descriptor=compute_visual_descriptor(sampled_frame.image),
        dino_descriptor=dino_descriptor,
        depth_m=depth_m,
        depth_eligible=depth_eligible,
        telemetry_reason=telemetry_reason,
    )


def select_keyframes(
    candidates: Iterable[FrameCandidate],
    novelty_threshold: float = 0.30,
    min_gap_sec: float = 5.0,
    max_gap_sec: float = 45.0,
    descriptor_backend: str = "classical",
    hybrid_dino_weight: float = 0.7,
    adaptive_threshold_enabled: bool = False,
) -> SelectionResult:
    """Select frames whose descriptor differs enough from the last kept frame."""

    if novelty_threshold < 0:
        raise ValueError("novelty_threshold must be non-negative")
    if min_gap_sec < 0:
        raise ValueError("min_gap_sec must be non-negative")
    if max_gap_sec <= 0:
        raise ValueError("max_gap_sec must be greater than 0")

    def _make(candidate: FrameCandidate, reason: str, distances: DistanceBreakdown) -> SelectedFrame:
        return SelectedFrame(
            candidate=candidate,
            reason=reason,
            distances=distances,
            descriptor_backend=descriptor_backend,
            novelty_threshold_used=novelty_threshold,
            adaptive_threshold_enabled=adaptive_threshold_enabled,
        )

    selected: list[SelectedFrame] = []
    sampled_count = 0
    skipped_by_quality = 0

    for candidate in candidates:
        sampled_count += 1
        if is_unusable_quality(candidate.quality):
            skipped_by_quality += 1
            continue

        if not selected:
            selected.append(_make(candidate, "first_frame", DistanceBreakdown(novelty_distance=0.0)))
            continue

        previous = selected[-1].candidate
        time_since_last = candidate.timestamp_sec - previous.timestamp_sec
        distances = frame_distance(
            candidate, previous, descriptor_backend=descriptor_backend, hybrid_dino_weight=hybrid_dino_weight
        )

        if time_since_last >= min_gap_sec and distances.novelty_distance >= novelty_threshold:
            selected.append(_make(candidate, "visual_change", distances))
        elif time_since_last >= max_gap_sec:
            selected.append(_make(candidate, "max_gap_fallback", distances))

    return SelectionResult(
        selected=selected, sampled_count=sampled_count, skipped_by_quality=skipped_by_quality
    )


def frame_distance(
    current: FrameCandidate,
    previous: FrameCandidate,
    descriptor_backend: str = "classical",
    hybrid_dino_weight: float = 0.7,
) -> DistanceBreakdown:
    """Compute the novelty distance between two candidates for the given backend."""

    return descriptor_distance(
        classical_left=current.classical_descriptor,
        classical_right=previous.classical_descriptor,
        dino_left=current.dino_descriptor,
        dino_right=previous.dino_descriptor,
        backend=descriptor_backend,
        hybrid_dino_weight=hybrid_dino_weight,
    )


def consecutive_distances(
    candidates: list[FrameCandidate],
    descriptor_backend: str = "classical",
    hybrid_dino_weight: float = 0.7,
) -> list[float]:
    """Compute novelty distances between consecutive usable candidates."""

    usable = [candidate for candidate in candidates if not is_unusable_quality(candidate.quality)]
    return [
        frame_distance(
            current, previous, descriptor_backend=descriptor_backend, hybrid_dino_weight=hybrid_dino_weight
        ).novelty_distance
        for previous, current in zip(usable, usable[1:])
    ]


def save_keyframes(selected: list[SelectedFrame], output_dir: Path) -> tuple[pd.DataFrame, list[Path]]:
    """Save selected frames as JPEGs and return a metadata table."""

    output_dir.mkdir(parents=True, exist_ok=True)
    for old_image_path in output_dir.glob("frame_*.jpg"):
        old_image_path.unlink()

    rows: list[dict[str, object]] = []
    image_paths: list[Path] = []

    for selected_id, frame in enumerate(selected, start=1):
        candidate = frame.candidate
        image_path = output_dir / f"frame_{selected_id:04d}_t{candidate.timestamp_sec:07.1f}.jpg"
        if not cv2.imwrite(str(image_path), candidate.image, [cv2.IMWRITE_JPEG_QUALITY, 92]):
            raise IOError(f"Could not write keyframe image: {image_path}")

        image_paths.append(image_path)
        rows.append(_metadata_row(selected_id, image_path, frame))

    return pd.DataFrame(rows), image_paths


def _metadata_row(selected_id: int, image_path: Path, frame: SelectedFrame) -> dict[str, object]:
    candidate = frame.candidate
    distances = frame.distances
    return {
        "selected_id": selected_id,
        "frame_index": candidate.frame_index,
        "timestamp_sec": round(candidate.timestamp_sec, 3),
        "image_path": str(image_path),
        "reason": frame.reason,
        "novelty_distance": round(distances.novelty_distance, 6),
        "depth_m": _round(candidate.depth_m),
        "depth_eligible": candidate.depth_eligible,
        "telemetry_reason": candidate.telemetry_reason,
        "descriptor_backend": frame.descriptor_backend,
        "novelty_threshold_used": round(frame.novelty_threshold_used, 6),
        "adaptive_threshold_enabled": frame.adaptive_threshold_enabled,
        "dino_distance": _round(distances.dino_distance),
        "classical_distance": _round(distances.classical_distance),
        "hybrid_distance": _round(distances.hybrid_distance),
        "sharpness": round(candidate.quality.sharpness, 6),
        "brightness_mean": round(candidate.quality.brightness_mean, 6),
        "brightness_std": round(candidate.quality.brightness_std, 6),
        "edge_density": round(candidate.quality.edge_density, 6),
    }


def _round(value: float | None) -> float | None:
    return None if value is None else round(value, 6)
