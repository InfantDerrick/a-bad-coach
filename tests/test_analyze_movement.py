from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "analyze_movement.py"
SPEC = importlib.util.spec_from_file_location("analyze_movement", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class AnalyzeMovementTests(unittest.TestCase):
    def test_transform_points_identity(self) -> None:
        points = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        transformed = MODULE.transform_points(points, np.eye(3, dtype=np.float32))
        np.testing.assert_allclose(transformed, points)

    def test_stationary_player_has_zero_distance(self) -> None:
        points = np.repeat(np.array([[3.05, 10.0]], dtype=np.float64), 60, axis=0)
        valid = np.ones(60, dtype=bool)
        metrics, _, speeds = MODULE.movement_metrics(points, 60.0, valid)
        self.assertEqual(metrics["distance_m"], 0.0)
        self.assertTrue(np.allclose(speeds, 0.0))

    def test_six_zone_occupancy(self) -> None:
        points = np.array([[1.0, 1.0], [5.0, 6.0], [1.0, 12.0]], dtype=np.float64)
        occupancy = MODULE.six_zone_occupancy(points, np.ones(3, dtype=bool))
        self.assertAlmostEqual(occupancy["left_front"], 1 / 3, places=4)
        self.assertAlmostEqual(occupancy["right_mid"], 1 / 3, places=4)
        self.assertAlmostEqual(occupancy["left_rear"], 1 / 3, places=4)


if __name__ == "__main__":
    unittest.main()
