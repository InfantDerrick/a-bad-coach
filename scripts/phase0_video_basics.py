#!/usr/bin/env python3
"""
Phase 0.5–0.6: OpenCV video I/O + NumPy image as array (resize, normalize, BGR↔RGB).

Usage:
  python scripts/phase0_video_basics.py path/to/video.mp4
  python scripts/phase0_video_basics.py   # uses data/raw/sample.mp4 if present

Requires: opencv-python, numpy, matplotlib (see requirements.txt).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 0: video frames + array basics")
    parser.add_argument(
        "video",
        nargs="?",
        default=None,
        help="Path to a video file (default: data/raw/sample.mp4)",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=5,
        help="How many frames to save under data/processed/phase0/ (default: 5)",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Skip cv2.imshow (use on headless servers)",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    default_video = root / "data" / "raw" / "sample.mp4"
    path = Path(args.video) if args.video else default_video

    if not path.is_file():
        print(f"No video at {path}. Place a clip in data/raw/ or pass a path.", file=sys.stderr)
        return 1

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        print(f"Could not open: {path}", file=sys.stderr)
        return 1

    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    print(f"Opened: {path}")
    print(f"  resolution: {w}x{h}, fps: {fps:.3f}, frame_count: {nframes}")

    out_dir = root / "data" / "processed" / "phase0"
    out_dir.mkdir(parents=True, exist_ok=True)

    idx = 0
    saved = 0
    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break

        # --- 0.6: treat frame as NumPy array (H, W, 3), uint8 BGR ---
        # Resize (e.g. for a fixed model input later)
        small = cv2.resize(frame_bgr, (320, 180), interpolation=cv2.INTER_AREA)
        # Normalize to [0, 1] float32 — common before neural nets
        normalized = small.astype(np.float32) / 255.0
        # BGR (OpenCV default) → RGB (matplotlib / many models)
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

        if saved < args.max_frames:
            # Save original BGR slice so you can open files in an image viewer
            cv2.imwrite(str(out_dir / f"frame_{idx:05d}_bgr.png"), small)
            # Save RGB for sanity check (should look like a normal photo)
            cv2.imwrite(
                str(out_dir / f"frame_{idx:05d}_rgb_via_cv2.png"),
                cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
            )
            saved += 1

        if not args.no_show:
            cv2.imshow("phase0 — BGR (OpenCV)", small)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        idx += 1

    cap.release()
    if not args.no_show:
        cv2.destroyAllWindows()

    print(f"Read {idx} frames; wrote first {saved} under {out_dir}")
    if idx and saved:
        print(f"Example array stats (first saved frame): shape {small.shape}, dtype {normalized.dtype}, min/max {normalized.min():.3f}/{normalized.max():.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
