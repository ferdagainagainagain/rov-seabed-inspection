"""Descriptor distance helpers for keyframe novelty backends."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rov_inspect.features import cosine_distance


@dataclass(frozen=True)
class DistanceBreakdown:
    """Novelty distance plus optional per-backend components."""

    novelty_distance: float
    classical_distance: float | None = None
    dino_distance: float | None = None
    hybrid_distance: float | None = None


def descriptor_distance(
    classical_left: np.ndarray | None,
    classical_right: np.ndarray | None,
    dino_left: np.ndarray | None,
    dino_right: np.ndarray | None,
    backend: str,
    hybrid_dino_weight: float = 0.7,
) -> DistanceBreakdown:
    """Compute novelty distance for the selected descriptor backend."""

    if backend == "classical":
        classical = _required_distance(classical_left, classical_right, "classical")
        return DistanceBreakdown(
            novelty_distance=classical,
            classical_distance=classical,
        )

    if backend == "dino":
        dino = _required_distance(dino_left, dino_right, "dino")
        return DistanceBreakdown(
            novelty_distance=dino,
            dino_distance=dino,
        )

    if backend == "hybrid":
        if not 0.0 <= hybrid_dino_weight <= 1.0:
            raise ValueError("hybrid_dino_weight must be between 0 and 1")
        classical = _required_distance(classical_left, classical_right, "classical")
        dino = _required_distance(dino_left, dino_right, "dino")
        hybrid = hybrid_distance(classical, dino, hybrid_dino_weight)
        return DistanceBreakdown(
            novelty_distance=hybrid,
            classical_distance=classical,
            dino_distance=dino,
            hybrid_distance=hybrid,
        )

    raise ValueError(f"Unknown descriptor backend: {backend}")


def hybrid_distance(classical_distance: float, dino_distance: float, dino_weight: float) -> float:
    """Combine classical and DINO distances with a fixed DINO weight."""

    if not 0.0 <= dino_weight <= 1.0:
        raise ValueError("dino_weight must be between 0 and 1")
    return dino_weight * dino_distance + (1.0 - dino_weight) * classical_distance


def adaptive_threshold(distances: list[float], k: float = 2.0, fallback: float = 0.30) -> float:
    """Compute median + k * MAD for robust novelty thresholding."""

    valid = np.array([value for value in distances if np.isfinite(value)], dtype=np.float32)
    if valid.size == 0:
        return fallback

    median = float(np.median(valid))
    mad = float(np.median(np.abs(valid - median)))
    return median + k * mad


def _required_distance(
    left: np.ndarray | None,
    right: np.ndarray | None,
    backend_name: str,
) -> float:
    if left is None or right is None:
        raise ValueError(f"Missing {backend_name} descriptor")
    return cosine_distance(left, right)
