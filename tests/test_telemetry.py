from __future__ import annotations

from pathlib import Path
import sys
import unittest

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rov_inspect.telemetry import (
    depth_at_timestamp,
    depth_eligibility,
    infer_time_and_depth_columns,
    load_depth_log,
)


class TelemetryTests(unittest.TestCase):
    def test_infers_depth_log_columns(self) -> None:
        df = pd.DataFrame(
            {
                "timestamp": ["2025.05.13 09:49:43:745"],
                "depth (m)": [0.4],
            }
        )

        self.assertEqual(infer_time_and_depth_columns(df), ("timestamp", "depth (m)"))

    def test_interpolates_depth_at_timestamp(self) -> None:
        df = pd.DataFrame(
            {
                "timestamp_sec": [0.0, 10.0],
                "depth": [0.0, 2.0],
            }
        )
        path = Path("/private/tmp/rov_depth_test.csv")
        df.to_csv(path, index=False)
        loaded = load_depth_log(path)

        self.assertAlmostEqual(depth_at_timestamp(loaded, 5.0), 1.0)

    def test_requires_stable_depth_window(self) -> None:
        df = pd.DataFrame(
            {
                "timestamp_sec": [0.0, 1.0, 2.0, 3.0],
                "depth": [0.5, 1.2, 1.4, 1.5],
            }
        )
        path = Path("/private/tmp/rov_depth_stable_test.csv")
        df.to_csv(path, index=False)
        loaded = load_depth_log(path)

        eligible, depth_m, reason = depth_eligibility(
            timestamp_sec=2.0,
            df=loaded,
            min_depth_m=1.0,
            stable_window_sec=2.0,
        )

        self.assertFalse(eligible)
        self.assertAlmostEqual(depth_m, 1.4)
        self.assertEqual(reason, "not_stably_underwater")


if __name__ == "__main__":
    unittest.main()
