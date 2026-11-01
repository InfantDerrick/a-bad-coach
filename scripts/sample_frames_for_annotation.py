#!/usr/bin/env python3
"""
Sample frames from video for manual annotation.

Usage:
  python3 scripts/sample_frames_for_annotation.py data/raw/bad_1.MOV --out data/processed/phase1/annotation --count 200

Writes images to OUT/images and a CSV with frame indices.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import csv
import cv2


def sample_frames(video: Path, out: Path, count: int = 200):
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video {video}")
    nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    out_images = out / 'images'
    out_images.mkdir(parents=True, exist_ok=True)
    indices = []
    if nframes <= 0:
        return
    # choose frames uniformly across video
    step = max(1, nframes // count)
    idx = 0
    saved = 0
    rows = []
    while idx < nframes and saved < count:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, f = cap.read()
        if not ok:
            break
        fname = out_images / f"frame_{saved:05d}.jpg"
        cv2.imwrite(str(fname), f)
        rows.append((str(fname.name), int(idx)))
        saved += 1
        idx += step
    cap.release()
    # write csv
    with open(out / 'frames.csv', 'w', newline='') as fh:
        writer = csv.writer(fh)
        writer.writerow(['filename', 'frame_index'])
        writer.writerows(rows)
    print(f"Wrote {saved} frames to {out_images}")


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('video')
    parser.add_argument('--out', type=Path, default=Path('data/processed/phase1/annotation'))
    parser.add_argument('--count', type=int, default=200)
    args = parser.parse_args(argv)
    sample_frames(Path(args.video), args.out, args.count)


if __name__ == '__main__':
    raise SystemExit(main())
