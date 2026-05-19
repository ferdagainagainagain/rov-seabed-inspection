from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rov_inspect.keyframes import FrameCandidate, select_keyframes
from rov_inspect.quality import FrameQuality


class KeyframeSelectionTests(unittest.TestCase):
    def test_selects_visual_change_after_min_gap(self) -> None:
        candidates = [
            _candidate(timestamp=0.0, descriptor=np.array([1.0, 0.0], dtype=np.float32)),
            _candidate(timestamp=2.0, descriptor=np.array([0.0, 1.0], dtype=np.float32)),
            _candidate(timestamp=6.0, descriptor=np.array([0.0, 1.0], dtype=np.float32)),
        ]

        result = select_keyframes(
            candidates,
            novelty_threshold=0.3,
            min_gap_sec=5.0,
            max_gap_sec=45.0,
        )

        self.assertEqual(result.sampled_count, 3)
        self.assertEqual([frame.reason for frame in result.selected], ["first_frame", "visual_change"])
        self.assertEqual(
            [frame.candidate.timestamp_sec for frame in result.selected], [0.0, 6.0]
        )

    def test_selects_max_gap_fallback_for_long_boring_scene(self) -> None:
        candidates = [
            _candidate(timestamp=0.0, descriptor=np.array([1.0, 0.0], dtype=np.float32)),
            _candidate(timestamp=10.0, descriptor=np.array([1.0, 0.0], dtype=np.float32)),
            _candidate(timestamp=46.0, descriptor=np.array([1.0, 0.0], dtype=np.float32)),
        ]

        result = select_keyframes(
            candidates,
            novelty_threshold=0.3,
            min_gap_sec=5.0,
            max_gap_sec=45.0,
        )

        self.assertEqual([frame.reason for frame in result.selected], ["first_frame", "max_gap_fallback"])
        self.assertEqual(
            [frame.candidate.timestamp_sec for frame in result.selected], [0.0, 46.0]
        )

    def test_skips_bad_initial_frame_when_good_frame_exists(self) -> None:
        bad_quality = FrameQuality(
            sharpness=0.0,
            brightness_mean=1.0,
            brightness_std=0.0,
            edge_density=0.0,
        )
        candidates = [
            _candidate(
                timestamp=0.0,
                descriptor=np.array([1.0, 0.0], dtype=np.float32),
                quality=bad_quality,
            ),
            _candidate(timestamp=1.0, descriptor=np.array([0.0, 1.0], dtype=np.float32)),
        ]

        result = select_keyframes(candidates)

        self.assertEqual([frame.reason for frame in result.selected], ["first_frame"])
        self.assertEqual(result.selected[0].candidate.timestamp_sec, 1.0)


def _candidate(
    timestamp: float,
    descriptor: np.ndarray,
    quality: FrameQuality | None = None,
) -> FrameCandidate:
    if quality is None:
        quality = FrameQuality(
            sharpness=100.0,
            brightness_mean=120.0,
            brightness_std=40.0,
            edge_density=0.05,
        )
    return FrameCandidate(
        frame_index=int(timestamp * 10),
        timestamp_sec=timestamp,
        image=np.full((24, 32, 3), 120, dtype=np.uint8),
        quality=quality,
        classical_descriptor=descriptor,
    )


if __name__ == "__main__":
    unittest.main()
