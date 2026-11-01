from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pipeline.court_keypoints import LANDMARKS, fit_homography_from_landmarks, validate_annotation


SCRIPT = ROOT / "scripts" / "train_court_keypoints.py"
SPEC = importlib.util.spec_from_file_location("train_court_keypoints", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class CourtTrainingTests(unittest.TestCase):
    def test_annotation_validation(self) -> None:
        payload = {
            "image": "frame.jpg",
            "image_width": 100,
            "image_height": 200,
            "complete": True,
            "points": {landmark.name: None for landmark in LANDMARKS},
        }
        payload["points"][LANDMARKS[0].name] = [25.0, 50.0]
        self.assertEqual(validate_annotation(payload), [])

    def test_model_output_shape(self) -> None:
        model = MODULE.CourtKeypointNet()
        output = model(torch.zeros(1, 3, 128, 72))
        self.assertEqual(tuple(output.shape), (1, len(LANDMARKS), 32, 18))

    def test_letterbox_preserves_aspect(self) -> None:
        image = np.zeros((200, 100, 3), dtype=np.uint8)
        output, scale, left, top = MODULE.letterbox(image, 200, 200)
        self.assertEqual(output.shape, (200, 200, 3))
        self.assertEqual(scale, 1.0)
        self.assertEqual(left, 50)
        self.assertEqual(top, 0)

    def test_partial_landmarks_recover_homography(self) -> None:
        selected = [LANDMARKS[0], LANDMARKS[1], LANDMARKS[-2], LANDMARKS[-1], LANDMARKS[5]]
        points = {landmark.name: None for landmark in LANDMARKS}
        expected = np.array([[80.0, 100.0], [520.0, 130.0], [610.0, 850.0], [20.0, 900.0]], dtype=np.float32)
        court = np.array([[0.0, 0.0], [6.1, 0.0], [6.1, 13.4], [0.0, 13.4]], dtype=np.float32)
        court_to_image = __import__("cv2").getPerspectiveTransform(court, expected)
        for landmark in selected:
            source = np.array([[[landmark.court_x_m, landmark.court_y_m]]], dtype=np.float32)
            point = __import__("cv2").perspectiveTransform(source, court_to_image).reshape(2)
            points[landmark.name] = point.tolist()
        image_to_court, diagnostics = fit_homography_from_landmarks(points)
        np.testing.assert_allclose(image_to_court / image_to_court[2, 2], np.linalg.inv(court_to_image) / np.linalg.inv(court_to_image)[2, 2], atol=1e-3)
        self.assertEqual(diagnostics["inlier_landmarks"], len(selected))


if __name__ == "__main__":
    unittest.main()
