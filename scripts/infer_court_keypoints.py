#!/usr/bin/env python3
"""Infer visible named court landmarks and extrapolate the full court template."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from pipeline.court_keypoints import (  # noqa: E402
    COURT_LENGTH_M,
    COURT_WIDTH_M,
    LANDMARKS,
    fit_homography_from_landmarks,
)
from train_court_keypoints import CourtKeypointNet, choose_device, letterbox  # noqa: E402


def read_input(path: Path, frame_index: int) -> np.ndarray:
    image = cv2.imread(str(path))
    if image is not None:
        return image
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not read image or video: {path}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Could not read frame {frame_index} from {path}")
    return frame


@torch.no_grad()
def predict(
    frame: np.ndarray,
    checkpoint_path: Path,
    confidence_threshold: float,
    device: torch.device,
) -> tuple[dict[str, list[float] | None], dict[str, float], dict]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    expected_names = [landmark.name for landmark in LANDMARKS]
    if checkpoint.get("landmarks") != expected_names:
        raise RuntimeError("Checkpoint landmark schema does not match the current court schema")
    input_width = int(checkpoint["input_width"])
    input_height = int(checkpoint["input_height"])
    stride = int(checkpoint["output_stride"])
    prepared, scale, pad_x, pad_y = letterbox(frame, input_width, input_height)
    rgb = cv2.cvtColor(prepared, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    tensor = torch.from_numpy(rgb.transpose(2, 0, 1)).unsqueeze(0).to(device)
    model = CourtKeypointNet().to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    probabilities = torch.sigmoid(model(tensor))[0].cpu().numpy()

    points: dict[str, list[float] | None] = {}
    confidences: dict[str, float] = {}
    frame_h, frame_w = frame.shape[:2]
    for channel, landmark in enumerate(LANDMARKS):
        heatmap = probabilities[channel]
        flat_index = int(np.argmax(heatmap))
        y_heat, x_heat = np.unravel_index(flat_index, heatmap.shape)
        confidence = float(heatmap[y_heat, x_heat])
        x_input = (x_heat + 0.5) * stride
        y_input = (y_heat + 0.5) * stride
        x_source = (x_input - pad_x) / scale
        y_source = (y_input - pad_y) / scale
        confidences[landmark.name] = round(confidence, 6)
        if confidence >= confidence_threshold and 0 <= x_source < frame_w and 0 <= y_source < frame_h:
            points[landmark.name] = [round(float(x_source), 3), round(float(y_source), 3)]
        else:
            points[landmark.name] = None
    metadata = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_validation_mae_input_px": checkpoint.get("validation_mae_input_px"),
        "input_width": input_width,
        "input_height": input_height,
        "confidence_threshold": confidence_threshold,
        "device": str(device),
    }
    return points, confidences, metadata


def draw_overlay(frame: np.ndarray, points: dict[str, list[float] | None], image_to_court: np.ndarray, diagnostics: dict) -> np.ndarray:
    output = frame.copy()
    court = np.asarray([[0, 0], [COURT_WIDTH_M, 0], [COURT_WIDTH_M, COURT_LENGTH_M], [0, COURT_LENGTH_M]], dtype=np.float32)
    court_to_image = np.linalg.inv(image_to_court)
    outline = cv2.perspectiveTransform(court.reshape(-1, 1, 2), court_to_image).reshape(-1, 2)
    cv2.polylines(output, [np.round(outline).astype(np.int32)], True, (0, 220, 255), 8, cv2.LINE_AA)
    for index, landmark in enumerate(LANDMARKS):
        point = points[landmark.name]
        if point is None:
            continue
        location = tuple(np.round(point).astype(int))
        cv2.circle(output, location, 13, (60, 255, 90), -1)
        cv2.putText(output, str(index + 1), (location[0] + 15, location[1] - 12), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3, cv2.LINE_AA)
    cv2.rectangle(output, (20, 20), (760, 120), (12, 18, 24), -1)
    message = f"{diagnostics['inlier_landmarks']}/{diagnostics['visible_landmarks']} inliers | reprojection {diagnostics['reprojection_rmse_px']:.1f}px"
    cv2.putText(output, message, (38, 82), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 220, 255), 3, cv2.LINE_AA)
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Detect a partial badminton court and fit the canonical court template")
    parser.add_argument("input", type=Path)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--frame-index", type=int, default=0)
    parser.add_argument("--confidence", type=float, default=0.55)
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/court_prediction"))
    parser.add_argument("--device", default="auto")
    args = parser.parse_args(argv)
    if not args.input.is_file() or not args.checkpoint.is_file():
        parser.error("input and checkpoint must exist")
    try:
        frame = read_input(args.input, args.frame_index)
        device = choose_device(args.device)
        points, confidences, metadata = predict(frame, args.checkpoint, args.confidence, device)
        image_to_court, diagnostics = fit_homography_from_landmarks(points)
    except (RuntimeError, ValueError, KeyError, OSError) as exc:
        print(f"Court inference failed: {exc}", file=sys.stderr)
        return 1
    args.output_dir.mkdir(parents=True, exist_ok=True)
    overlay = draw_overlay(frame, points, image_to_court, diagnostics)
    cv2.imwrite(str(args.output_dir / "court_overlay.png"), overlay)
    visible_names = diagnostics["inlier_names"]
    calibration = {
        "status": "model_prediction_unverified",
        "image_points": [points[name] for name in visible_names],
        "court_points_m": [
            [next(item for item in LANDMARKS if item.name == name).court_x_m, next(item for item in LANDMARKS if item.name == name).court_y_m]
            for name in visible_names
        ],
        "notes": [
            "Predicted from visible landmarks using the canonical full-court template.",
            "Off-screen court boundaries are geometric extrapolations and require confidence/reprojection quality gates.",
        ],
    }
    (args.output_dir / "calibration.json").write_text(json.dumps(calibration, indent=2))
    (args.output_dir / "prediction.json").write_text(
        json.dumps({"points": points, "confidence": confidences, "fit": diagnostics, "model": metadata}, indent=2)
    )
    print(json.dumps({"output": str(args.output_dir), "fit": diagnostics}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
