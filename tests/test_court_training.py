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

    def test_collapsed_5080_predictions_are_rejected(self) -> None:
        points = dict(zip(
            ["far_baseline_right", "far_short_service_center", "far_short_service_right",
             "net_right", "near_short_service_center", "near_short_service_right"],
            [[1485, 1965], [1395, 2265], [1485, 1935], [1485, 1935], [1185, 1785], [1485, 1935]],
        ))
        with self.assertRaisesRegex(ValueError, "duplicate|collapsed"):
            fit_homography_from_landmarks(points)

    def test_collinear_and_nonfinite_image_points_are_rejected(self) -> None:
        for coords in ([[0, 0], [100, 0], [200, 0], [300, 0]],
                       [[0, 0], [100, 0], [100, 100], [float('nan'), 100]]):
            with self.subTest(coords=coords), self.assertRaises(ValueError):
                fit_homography_from_landmarks(dict(zip(
                    [LANDMARKS[i].name for i in (0, 1, 14, 15)], coords)))

    def test_real_annotations_still_fit(self) -> None:
        import json
        paths = sorted((ROOT / 'data/court_dataset/annotations').glob('bad_1*.json'))
        self.assertEqual(len(paths), 12)
        for path in paths:
            with self.subTest(frame=path.name):
                if path.stem.endswith('f000144'):
                    # Small visible-point residuals do not guarantee a valid
                    # full-court extrapolation: this label fit crosses infinity.
                    with self.assertRaisesRegex(ValueError, 'horizon'):
                        fit_homography_from_landmarks(json.loads(path.read_text())['points'])
                    continue
                _, diagnostics = fit_homography_from_landmarks(json.loads(path.read_text())['points'])
                self.assertLessEqual(diagnostics['max_reprojection_error_px'], 8.0)
                self.assertGreaterEqual(diagnostics['inlier_landmarks'], 4)
                self.assertIn('per_landmark_residual_px', diagnostics)

    def test_consensus_refit_retains_residual_for_outlier(self) -> None:
        points = {x.name: [100 + x.court_x_m * 50, 80 + x.court_y_m * 50] for x in LANDMARKS}
        points['net_right'] = [1800, 1700]
        matrix, diagnostics = fit_homography_from_landmarks(points)
        self.assertEqual(diagnostics['inlier_landmarks'], 15)
        self.assertNotIn('net_right', diagnostics['inlier_names'])
        self.assertGreater(diagnostics['per_landmark_residual_px']['net_right'], 100)
        np.testing.assert_allclose(matrix, [[.02, 0, -2], [0, .02, -1.6], [0, 0, 1]], atol=1e-6)

    def test_abstention_removes_stale_calibration(self) -> None:
        import json
        import tempfile
        from unittest.mock import patch
        sys.path.insert(0, str(ROOT / 'scripts'))
        import infer_court_keypoints as inference
        with tempfile.TemporaryDirectory(dir=ROOT) as temp:
            directory = Path(temp)
            source = directory / 'frame.png'
            checkpoint = directory / 'model.pt'
            source.touch()
            checkpoint.touch()
            (directory / 'calibration.json').write_text('{}')
            (directory / 'court_overlay.png').touch()
            with patch.object(inference, 'read_input', return_value=np.zeros((8, 8, 3))), \
                 patch.object(inference, 'predict', return_value=({}, {}, {})):
                code = inference.main([str(source), '--checkpoint', str(checkpoint),
                                       '--output-dir', str(directory), '--device', 'cpu'])
            self.assertEqual(code, 1)
            self.assertEqual(json.loads((directory / 'prediction.json').read_text())['status'], 'abstained')
            self.assertFalse((directory / 'calibration.json').exists())
            self.assertFalse((directory / 'court_overlay.png').exists())

    def test_evaluation_distinguishes_missing_points_from_accuracy(self) -> None:
        sys.path.insert(0, str(ROOT / 'scripts'))
        from evaluate_court_keypoints import evaluate
        annotation = {'points': {'far_baseline_left': [10, 20], 'far_baseline_right': [40, 20]},
                      'image_width': 300, 'image_height': 400}
        result = evaluate(annotation, {'far_baseline_left': [13, 24]})
        self.assertEqual(result['mae_source_px'], 5)
        self.assertEqual(result['normalized_mae_diagonal'], .01)
        self.assertEqual(result['visible_coverage'], .5)
        self.assertEqual(result['status'], 'abstained')
        empty = evaluate(annotation, {})
        self.assertIsNone(empty['mae_source_px'])
        self.assertEqual(empty['visible_coverage'], 0)

    def test_inference_round_trip_matches_training_grid_with_padding(self) -> None:
        from unittest.mock import patch, MagicMock
        sys.path.insert(0, str(ROOT / 'scripts'))
        import infer_court_keypoints as inference
        # Portrait frame letterboxes to 256x512 with 16 px horizontal padding.
        # Source (35,50) -> input (128,160) -> heatmap (32,40).
        logits = torch.full((1, len(LANDMARKS), 128, 72), -20.)
        logits[0, 0, 40, 32] = 20
        network = MagicMock()
        network.to.return_value = network
        network.return_value = logits
        checkpoint = {'landmarks': [x.name for x in LANDMARKS],
                      'input_width': 288, 'input_height': 512, 'output_stride': 4, 'model_state': {}}
        with patch.object(inference.torch, 'load', return_value=checkpoint), \
             patch.object(inference, 'CourtKeypointNet', return_value=network):
            points, _, _ = inference.predict(np.zeros((160, 80, 3), dtype=np.uint8), Path('unused.pt'), .55, torch.device('cpu'))
        self.assertEqual(points[LANDMARKS[0].name], [35., 50.])
        self.assertIsNone(points[LANDMARKS[1].name])


if __name__ == "__main__":
    unittest.main()
