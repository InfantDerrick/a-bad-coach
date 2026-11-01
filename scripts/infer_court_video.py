#!/usr/bin/env python3
"""Run court detection across a video with temporal lock and primary-court selection."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from infer_court_keypoints import draw_overlay, predict  # noqa: E402
from pipeline.court_tracking import CourtTemporalLock, movement_metrics_allowed  # noqa: E402
from train_court_keypoints import choose_device  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/court_video"))
    parser.add_argument("--confidence", type=float, default=0.55)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--stride", type=int, default=30, help="Process every Nth frame")
    parser.add_argument("--max-frames", type=int, default=0, help="Optional cap on processed frames (0 = all)")
    parser.add_argument("--write-overlay-video", action="store_true")
    args = parser.parse_args(argv)
    if not args.video.is_file() or not args.checkpoint.is_file():
        parser.error("video and checkpoint must exist")
    if args.stride < 1:
        parser.error("--stride must be positive")

    device = choose_device(args.device)
    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        print(f"Could not open video: {args.video}", file=sys.stderr)
        return 1
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    writer = None
    if args.write_overlay_video:
        writer = cv2.VideoWriter(
            str(args.output_dir / "court_overlay.mp4"),
            cv2.VideoWriter_fourcc(*"mp4v"),
            max(1.0, fps / args.stride),
            (width, height),
        )

    lock = CourtTemporalLock()
    rows: list[dict] = []
    index = 0
    processed = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if index % args.stride != 0:
            index += 1
            continue
        points, confidences, metadata = predict(frame, args.checkpoint, args.confidence, device)
        matrix, diagnostics = lock.update(points, frame.shape)
        row = {
            "frame_index": index,
            "status": diagnostics.get("status"),
            "reason": diagnostics.get("reason"),
            "fit": {k: v for k, v in diagnostics.items() if k not in {"reason"}} if matrix is not None else None,
            "confidence": confidences,
            "points": points,
            "metrics_trustworthy": False,
        }
        if matrix is not None:
            overlay = draw_overlay(frame, points, matrix, diagnostics)
            if writer is not None:
                writer.write(overlay)
            if processed == 0:
                cv2.imwrite(str(args.output_dir / "first_lock_overlay.png"), overlay)
                calibration_status = "model_prediction_unverified"
                (args.output_dir / "calibration.json").write_text(
                    json.dumps(
                        {
                            "status": calibration_status,
                            "metrics_trustworthy": movement_metrics_allowed(calibration_status),
                            "frame_index": index,
                            "matrix_image_to_court": matrix.tolist(),
                            "fit": diagnostics,
                            "model": metadata,
                            "notes": [
                                "Temporally locked court from model predictions.",
                                "Not verified for trustworthy movement metrics.",
                            ],
                        },
                        indent=2,
                    )
                )
        rows.append(row)
        processed += 1
        index += 1
        if args.max_frames and processed >= args.max_frames:
            break

    cap.release()
    if writer is not None:
        writer.release()
    locked = sum(1 for row in rows if row["status"] in {"locked", "locked_new", "held_previous", "locking", "locked_hold_pending_switch", "switched"})
    abstained = sum(1 for row in rows if row["status"] == "abstained")
    summary = {
        "video": str(args.video),
        "checkpoint": str(args.checkpoint),
        "frames_processed": len(rows),
        "locked_or_held": locked,
        "abstained": abstained,
        "metrics_trustworthy": False,
        "results": rows,
    }
    (args.output_dir / "video_prediction.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "results"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
