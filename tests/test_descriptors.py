from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rov_inspect.descriptors import adaptive_threshold, descriptor_distance, hybrid_distance
from rov_inspect.features import cosine_distance


class DescriptorTests(unittest.TestCase):
    def test_cosine_distance_identical_vectors(self) -> None:
        vector = np.array([1.0, 0.0], dtype=np.float32)

        self.assertAlmostEqual(cosine_distance(vector, vector), 0.0)

    def test_hybrid_distance_uses_dino_weight(self) -> None:
        self.assertAlmostEqual(
            hybrid_distance(classical_distance=0.2, dino_distance=0.8, dino_weight=0.75),
            0.65,
        )

    def test_descriptor_distance_hybrid_reports_components(self) -> None:
        classical_a = np.array([1.0, 0.0], dtype=np.float32)
        classical_b = np.array([0.0, 1.0], dtype=np.float32)
        dino_a = np.array([1.0, 0.0], dtype=np.float32)
        dino_b = np.array([1.0, 0.0], dtype=np.float32)

        distances = descriptor_distance(
            classical_a,
            classical_b,
            dino_a,
            dino_b,
            backend="hybrid",
            hybrid_dino_weight=0.7,
        )

        self.assertAlmostEqual(distances.classical_distance, 1.0)
        self.assertAlmostEqual(distances.dino_distance, 0.0)
        self.assertAlmostEqual(distances.hybrid_distance, 0.3)
        self.assertAlmostEqual(distances.novelty_distance, 0.3)

    def test_adaptive_threshold_uses_median_and_mad(self) -> None:
        threshold = adaptive_threshold([0.1, 0.2, 0.3], k=2.0)

        self.assertAlmostEqual(threshold, 0.4, places=6)


if __name__ == "__main__":
    unittest.main()
