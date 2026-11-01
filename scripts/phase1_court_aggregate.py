#!/usr/bin/env python3
"""
Aggregate per-frame line detections to estimate stable court corners and draw court lines.

Usage:
  python3 scripts/phase1_court_aggregate.py data/raw/bad_1.MOV --samples 40

Outputs written to `data/processed/phase1/`:
  - court_overlay.png  (frame with estimated court lines drawn)
  - court_corners.json (4 image pixel corner points: tl,tr,br,bl)
  - court_homography.npy (3x3 homography mapping image->court template)

This script depends on `scripts/phase1_court_player.py` functions for line detection and player anchoring.
"""
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np


def kmeans_2d(points: np.ndarray, k: int = 4, iters: int = 50, restarts: int = 5) -> np.ndarray:
    best_inertia = float('inf')
    best_centers = None
    n = points.shape[0]
    if n == 0:
        return np.zeros((k, 2), dtype=float)
    for _ in range(restarts):
        # initialize centers by sampling
        idx = np.random.choice(n, k, replace=False) if n >= k else np.random.choice(n, k, replace=True)
        centers = points[idx].astype(float)
        for _ in range(iters):
            # assign
            dists = np.linalg.norm(points[:, None, :] - centers[None, :, :], axis=2)
            labels = np.argmin(dists, axis=1)
            new_centers = np.array([points[labels == j].mean(axis=0) if np.any(labels == j) else centers[j] for j in range(k)])
            if np.allclose(new_centers, centers):
                break
            centers = new_centers
        inertia = np.sum((points - centers[labels]) ** 2)
        if inertia < best_inertia:
            best_inertia = inertia
            best_centers = centers.copy()
    return best_centers


def order_corners(pts: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    # Order: top-left, top-right, bottom-right, bottom-left
    arr = np.array(pts)
    s = arr.sum(axis=1)
    diff = np.diff(arr, axis=1).reshape(-1)
    tl = arr[np.argmin(s)]
    br = arr[np.argmax(s)]
    tr = arr[np.argmin(diff)]
    bl = arr[np.argmax(diff)]
    return [(float(tl[0]), float(tl[1])), (float(tr[0]), float(tr[1])), (float(br[0]), float(br[1])), (float(bl[0]), float(bl[1]))]


def sample_frames(video_path: Path, nsamples: int) -> List[np.ndarray]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video {video_path}")
    nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    idxs = np.linspace(0, max(0, nframes - 1), num=min(nsamples, max(1, nframes)), dtype=int)
    frames = []
    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, f = cap.read()
        if not ok:
            continue
        frames.append(f)
    cap.release()
    return frames


def collect_intersections(frames: List[np.ndarray]) -> np.ndarray:
    # import helpers from phase1_court_player via importlib (scripts may not be a package)
    import importlib.util
    import sys

    repo_root = Path(__file__).resolve().parents[1]
    mod_path = repo_root / 'scripts' / 'phase1_court_player.py'
    spec = importlib.util.spec_from_file_location('phase1_court_player', str(mod_path))
    p1 = importlib.util.module_from_spec(spec)
    sys.modules['phase1_court_player'] = p1
    spec.loader.exec_module(p1)  # type: ignore

    intersections = []
    for frame in frames:
        try:
            player = None
            try:
                player = p1.detect_player_simple(frame)
            except Exception:
                player = None
            vis, lines = p1.detect_lines(frame, player_bbox=player)
            filt = p1.filter_lines(lines)
            # split into two angle groups if possible
            if len(filt) < 2:
                continue
            # compute intersections between pairs from different angle clusters
            for i in range(len(filt)):
                for j in range(i + 1, len(filt)):
                    pt = p1.line_intersection(filt[i], filt[j])
                    if pt is not None:
                        x, y = pt
                        if math.isfinite(x) and math.isfinite(y) and 0 <= x <= frame.shape[1] and 0 <= y <= frame.shape[0]:
                            intersections.append([x, y])
        except Exception:
            continue
    if not intersections:
        return np.zeros((0, 2), dtype=float)
    return np.array(intersections, dtype=float)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('video')
    parser.add_argument('--samples', type=int, default=40)
    parser.add_argument('--out-dir', type=Path, default=Path('data/processed/phase1'))
    args = parser.parse_args(argv)

    video = Path(args.video)
    frames = sample_frames(video, args.samples)
    if not frames:
        print('No frames sampled', flush=True)
        return 2

    intersections = collect_intersections(frames)
    if intersections.shape[0] < 8:
        print('Not enough intersections found to cluster corners', flush=True)
        return 3

    centers = kmeans_2d(intersections, k=4, iters=100, restarts=8)
    # ensure centers are finite
    centers = centers[np.isfinite(centers).all(axis=1)]
    if centers.shape[0] != 4:
        print('Could not find 4 corner clusters', flush=True)
        return 4

    ordered = order_corners([tuple(c) for c in centers])

    # target court size (pixels) for homography mapping — use typical badminton court aspect ratio
    dst_size = (1000, 455)
    dst = np.array([[0, 0], [dst_size[0] - 1, 0], [dst_size[0] - 1, dst_size[1] - 1], [0, dst_size[1] - 1]], dtype=np.float32)
    src = np.array(ordered, dtype=np.float32)
    H, mask = cv2.findHomography(src, dst, method=cv2.RANSAC)
    if H is None:
        print('Homography estimation failed', flush=True)
        return 5

    args.out_dir.mkdir(parents=True, exist_ok=True)
    # draw overlay on middle frame
    mid = frames[len(frames) // 2].copy()
    # project dst court rectangle back to image
    rect = np.array([[0, 0], [dst_size[0] - 1, 0], [dst_size[0] - 1, dst_size[1] - 1], [0, dst_size[1] - 1]], dtype=np.float32).reshape(-1, 1, 2)
    H_inv = np.linalg.inv(H)
    # map dst rect to image using inverse homography
    pts_img = cv2.perspectiveTransform(rect, H_inv).reshape(-1, 2)
    pts = pts_img.astype(int).tolist()
    # draw lines
    for i in range(4):
        p0 = tuple(map(int, pts[i]))
        p1i = tuple(map(int, pts[(i + 1) % 4]))
        cv2.line(mid, p0, p1i, (0, 255, 255), 4)
    # save overlay and data
    out_img = args.out_dir / 'court_overlay.png'
    cv2.imwrite(str(out_img), mid)
    (args.out_dir / 'court_corners.json').write_text(json.dumps(ordered, indent=2))
    np.save(str(args.out_dir / 'court_homography.npy'), H)
    print(f'Wrote overlay to {out_img}', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
