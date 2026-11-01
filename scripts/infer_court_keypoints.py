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
)
from pipeline.court_tracking import (  # noqa: E402
    CourtTemporalLock,
    decode_heatmap_peak_subpixel,
    movement_metrics_allowed,
    select_primary_court,
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
        x_heat, y_heat, confidence = decode_heatmap_peak_subpixel(heatmap)
        # Training centers targets at input_coordinate / stride, without a
        # half-cell shift. Decode on that same grid, with subpixel refinement.
        x_input = x_heat * stride
        y_input = y_heat * stride
        x_source = (x_input - pad_x) / scale
        y_source = (y_input - pad_y) / scale
        confidences[landmark.name] = round(float(confidence), 6)
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
        "subpixel_decode": True,
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
    cv2.rectangle(output, (20, 20), (900, 120), (12, 18, 24), -1)
    message = (
        f"{diagnostics.get('inlier_landmarks', 0)}/{diagnostics.get('visible_landmarks', 0)} inliers | "
        f"reprojection {diagnostics.get('reprojection_rmse_px', float('nan')):.1f}px | "
        f"{diagnostics.get('status', 'fit')}"
    )
    cv2.putText(output, message, (38, 82), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 220, 255), 3, cv2.LINE_AA)
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Detect a partial badminton court and fit the canonical court template")
    parser.add_argument("input", type=Path)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--frame-index", type=int, default=0)
    parser.add_argument("--confidence", type=float, default=0.55)
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/court_prediction"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--temporal-lock", action="store_true", help="Stabilize across sequential --frame-index calls via lock file")
    args = parser.parse_args(argv)
    if not args.input.is_file() or not args.checkpoint.is_file():
        parser.error("input and checkpoint must exist")
    try:
        frame = read_input(args.input, args.frame_index)
        device = choose_device(args.device)
        points, confidences, metadata = predict(frame, args.checkpoint, args.confidence, device)
    except (RuntimeError, ValueError, KeyError, OSError) as exc:
        print(f"Court inference failed: {exc}", file=sys.stderr)
        return 1
    args.output_dir.mkdir(parents=True, exist_ok=True)
    lock_path = args.output_dir / "temporal_lock.json"
    try:
        if args.temporal_lock:
            lock = CourtTemporalLock()
            if lock_path.exists():
                payload = json.loads(lock_path.read_text())
                if payload.get("matrix") is not None:
                    lock.locked_matrix = np.asarray(payload["matrix"], dtype=np.float64)
                    lock.lock_count = int(payload.get("lock_count", 0))
            image_to_court, diagnostics = lock.update(points, frame.shape)
            if image_to_court is None:
                raise ValueError(diagnostics.get("reason", "temporal lock abstained"))
            lock_path.write_text(
                json.dumps({"matrix": image_to_court.tolist(), "lock_count": lock.lock_count, "status": diagnostics.get("status")}, indent=2)
            )
        else:
            image_to_court, diagnostics = select_primary_court(points, frame.shape)
            diagnostics.setdefault("status", "primary_selected")
    except ValueError as exc:
        for filename in ("calibration.json", "court_overlay.png"):
            (args.output_dir / filename).unlink(missing_ok=True)
        result = {"status": "abstained", "reason": str(exc), "points": points,
                  "confidence": confidences, "model": metadata,
                  "metrics_trustworthy": False}
        (args.output_dir / "prediction.json").write_text(json.dumps(result, indent=2))
        print(json.dumps(result, indent=2))
        return 1
    overlay = draw_overlay(frame, points, image_to_court, diagnostics)
    cv2.imwrite(str(args.output_dir / "court_overlay.png"), overlay)
    visible_names = diagnostics["inlier_names"]
    calibration_status = "model_prediction_unverified"
    calibration = {
        "status": calibration_status,
        "metrics_trustworthy": movement_metrics_allowed(calibration_status),
        "image_points": [points[name] for name in visible_names],
        "court_points_m": [
            [next(item for item in LANDMARKS if item.name == name).court_x_m, next(item for item in LANDMARKS if item.name == name).court_y_m]
            for name in visible_names
        ],
        "fit": diagnostics,
        "notes": [
            "Predicted from visible landmarks using the canonical full-court template.",
            "Off-screen court boundaries are geometric extrapolations and require confidence/reprojection quality gates.",
            "Movement metrics remain gated until calibration status is verified.",
        ],
    }
    (args.output_dir / "calibration.json").write_text(json.dumps(calibration, indent=2))
    (args.output_dir / "prediction.json").write_text(
        json.dumps(
            {
                "status": calibration_status,
                "metrics_trustworthy": False,
                "points": points,
                "confidence": confidences,
                "fit": diagnostics,
                "model": metadata,
            },
            indent=2,
        )
    )
    print(json.dumps({"output": str(args.output_dir), "fit": diagnostics, "metrics_trustworthy": False}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
