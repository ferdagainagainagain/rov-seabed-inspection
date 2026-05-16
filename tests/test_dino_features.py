from __future__ import annotations

from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


class DinoFeatureTests(unittest.TestCase):
    def test_choose_cpu_device_when_dependencies_exist(self) -> None:
        try:
            import torch  # noqa: F401
            import transformers  # noqa: F401
        except ImportError:
            self.skipTest("DINO dependencies are not installed")

        from rov_inspect.dino_features import choose_device

        self.assertEqual(choose_device("cpu"), "cpu")


if __name__ == "__main__":
    unittest.main()
