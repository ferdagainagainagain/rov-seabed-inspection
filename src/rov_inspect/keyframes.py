"""Keyframe selection based on simple visual novelty."""

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
    """A sampled frame with quality metrics and a visual descriptor."""

    frame_index: int
    timestamp_sec: float
    image: np.ndarray
    quality: FrameQuality
    descriptor: np.ndarray
    classical_descriptor: np.ndarray | None = None
    dino_descriptor: np.ndarray | None = None
    depth_m: float | None = None
    depth_eligible: bool | None = None
    telemetry_reason: str = "not_used"


@dataclass(frozen=True)
class SelectedFrame:
    """A selected keyframe and the reason it was kept."""

    frame_index: int
    timestamp_sec: float
    image: np.ndarray
    quality: FrameQuality
    descriptor: np.ndarray
    classical_descriptor: np.ndarray | None
    dino_descriptor: np.ndarray | None
    reason: str
    novelty_distance: float
    depth_m: float | None = None
    depth_eligible: bool | None = None
    telemetry_reason: str = "not_used"
    descriptor_backend: str = "classical"
    novelty_threshold_used: float | None = None
    adaptive_threshold_enabled: bool = False
    dino_distance: float | None = None
    classical_distance: float | None = None
    hybrid_distance: float | None = None


@dataclass(frozen=True)
class SelectionResult:
    """Selected keyframes plus the number of sampled frames inspected."""

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
    """Compute quality metrics and descriptor for one sampled frame."""

    classical_descriptor = compute_visual_descriptor(sampled_frame.image)
    return FrameCandidate(
        frame_index=sampled_frame.frame_index,
        timestamp_sec=sampled_frame.timestamp_sec,
        image=sampled_frame.image,
        quality=compute_quality(sampled_frame.image),
        descriptor=classical_descriptor,
        classical_descriptor=classical_descriptor,
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
    """Select frames using novelty against the last selected descriptor."""

    if novelty_threshold < 0:
        raise ValueError("novelty_threshold must be non-negative")
    if min_gap_sec < 0:
        raise ValueError("min_gap_sec must be non-negative")
    if max_gap_sec <= 0:
        raise ValueError("max_gap_sec must be greater than 0")

    selected: list[SelectedFrame] = []
    best_seen: FrameCandidate | None = None
    sampled_count = 0
    skipped_by_quality = 0

    for candidate in candidates:
        sampled_count += 1
        if best_seen is None or quality_score(candidate.quality) > quality_score(best_seen.quality):
            best_seen = _copy_candidate(candidate)

        if is_unusable_quality(candidate.quality):
            skipped_by_quality += 1
            continue

        if not selected:
            selected.append(
                _select(
                    candidate,
                    "first_frame",
                    DistanceBreakdown(novelty_distance=0.0),
                    descriptor_backend,
                    novelty_threshold,
                    adaptive_threshold_enabled,
                )
            )
            continue

        last_selected = selected[-1]
        time_since_last = candidate.timestamp_sec - last_selected.timestamp_sec
        distances = frame_distance(
            candidate,
            last_selected,
            descriptor_backend=descriptor_backend,
            hybrid_dino_weight=hybrid_dino_weight,
        )

        if time_since_last >= min_gap_sec and distances.novelty_distance >= novelty_threshold:
            selected.append(
                _select(
                    candidate,
                    "visual_change",
                    distances,
                    descriptor_backend,
                    novelty_threshold,
                    adaptive_threshold_enabled,
                )
            )
        elif time_since_last >= max_gap_sec:
            selected.append(
                _select(
                    candidate,
                    "max_gap_fallback",
                    distances,
                    descriptor_backend,
                    novelty_threshold,
                    adaptive_threshold_enabled,
                )
            )

    if not selected and best_seen is not None:
        selected.append(
            _select(
                best_seen,
                "best_available",
                DistanceBreakdown(novelty_distance=0.0),
                descriptor_backend,
                novelty_threshold,
                adaptive_threshold_enabled,
            )
        )

    return SelectionResult(
        selected=selected,
        sampled_count=sampled_count,
        skipped_by_quality=skipped_by_quality,
    )


def frame_distance(
    current: FrameCandidate | SelectedFrame,
    previous: FrameCandidate | SelectedFrame,
    descriptor_backend: str = "classical",
    hybrid_dino_weight: float = 0.7,
) -> DistanceBreakdown:
    """Compute distance between two frames for the selected backend."""

    current_classical = current.classical_descriptor if current.classical_descriptor is not None else current.descriptor
    previous_classical = previous.classical_descriptor if previous.classical_descriptor is not None else previous.descriptor
    return descriptor_distance(
        classical_left=current_classical,
        classical_right=previous_classical,
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
    """Compute distances between consecutive usable candidates."""

    usable = [candidate for candidate in candidates if not is_unusable_quality(candidate.quality)]
    distances: list[float] = []
    for previous, current in zip(usable, usable[1:]):
        distances.append(
            frame_distance(
                current,
                previous,
                descriptor_backend=descriptor_backend,
                hybrid_dino_weight=hybrid_dino_weight,
            ).novelty_distance
        )
    return distances


def save_keyframes(selected: list[SelectedFrame], output_dir: Path) -> tuple[pd.DataFrame, list[Path]]:
    """Save selected frames as JPEGs and return metadata rows."""

    output_dir.mkdir(parents=True, exist_ok=True)
    for old_image_path in output_dir.glob("frame_*.jpg"):
        old_image_path.unlink()

    rows: list[dict[str, object]] = []
    image_paths: list[Path] = []

    for selected_id, frame in enumerate(selected, start=1):
        image_path = output_dir / _image_name(selected_id, frame.timestamp_sec)
        ok = cv2.imwrite(str(image_path), frame.image, [cv2.IMWRITE_JPEG_QUALITY, 92])
        if not ok:
            raise IOError(f"Could not write keyframe image: {image_path}")

        image_paths.append(image_path)
        rows.append(
            {
                "selected_id": selected_id,
                "frame_index": frame.frame_index,
                "timestamp_sec": round(frame.timestamp_sec, 3),
                "image_path": str(image_path),
                "reason": frame.reason,
                "novelty_distance": round(frame.novelty_distance, 6),
                "depth_m": _round_optional(frame.depth_m),
                "depth_eligible": frame.depth_eligible,
                "telemetry_reason": frame.telemetry_reason,
                "descriptor_backend": frame.descriptor_backend,
                "novelty_threshold_used": _round_optional(frame.novelty_threshold_used),
                "adaptive_threshold_enabled": frame.adaptive_threshold_enabled,
                "dino_distance": _round_optional(frame.dino_distance),
                "classical_distance": _round_optional(frame.classical_distance),
                "hybrid_distance": _round_optional(frame.hybrid_distance),
                "sharpness": round(frame.quality.sharpness, 6),
                "brightness_mean": round(frame.quality.brightness_mean, 6),
                "brightness_std": round(frame.quality.brightness_std, 6),
                "edge_density": round(frame.quality.edge_density, 6),
            }
        )

    return pd.DataFrame(rows), image_paths


def _select(
    candidate: FrameCandidate,
    reason: str,
    distances: DistanceBreakdown,
    descriptor_backend: str,
    novelty_threshold: float,
    adaptive_threshold_enabled: bool,
) -> SelectedFrame:
    return SelectedFrame(
        frame_index=candidate.frame_index,
        timestamp_sec=candidate.timestamp_sec,
        image=candidate.image.copy(),
        quality=candidate.quality,
        descriptor=candidate.descriptor.copy(),
        classical_descriptor=_copy_optional_array(candidate.classical_descriptor),
        dino_descriptor=_copy_optional_array(candidate.dino_descriptor),
        reason=reason,
        novelty_distance=distances.novelty_distance,
        depth_m=candidate.depth_m,
        depth_eligible=candidate.depth_eligible,
        telemetry_reason=candidate.telemetry_reason,
        descriptor_backend=descriptor_backend,
        novelty_threshold_used=novelty_threshold,
        adaptive_threshold_enabled=adaptive_threshold_enabled,
        dino_distance=distances.dino_distance,
        classical_distance=distances.classical_distance,
        hybrid_distance=distances.hybrid_distance,
    )


def _copy_candidate(candidate: FrameCandidate) -> FrameCandidate:
    return FrameCandidate(
        frame_index=candidate.frame_index,
        timestamp_sec=candidate.timestamp_sec,
        image=candidate.image.copy(),
        quality=candidate.quality,
        descriptor=candidate.descriptor.copy(),
        classical_descriptor=_copy_optional_array(candidate.classical_descriptor),
        dino_descriptor=_copy_optional_array(candidate.dino_descriptor),
        depth_m=candidate.depth_m,
        depth_eligible=candidate.depth_eligible,
        telemetry_reason=candidate.telemetry_reason,
    )


def _image_name(selected_id: int, timestamp_sec: float) -> str:
    return f"frame_{selected_id:04d}_t{timestamp_sec:07.1f}.jpg"


def _round_optional(value: float | None) -> float | None:
    if value is None:
        return None
    return round(value, 6)


def _copy_optional_array(value: np.ndarray | None) -> np.ndarray | None:
    if value is None:
        return None
    return value.copy()
