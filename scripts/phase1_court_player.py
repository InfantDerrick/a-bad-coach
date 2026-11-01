#!/usr/bin/env python3
"""
Phase 1: court line detection, homography helpers, and simple player-detection wrapper.

Usage examples:
  # show CLI help
  python scripts/phase1_court_player.py --help

  # extract a representative frame, run Canny+Hough and save visualizations
  python scripts/phase1_court_player.py data/raw/bad_1.MOV --frame-time 2.0

  # compute a homography from 4 src points (JSON) and warp the frame to bird's-eye
  python scripts/phase1_court_player.py data/raw/bad_1.MOV --frame-time 2.0 --src-pts points.json --warp-out data/processed/phase1/warped.png

Notes:
  - For full automation of court corner selection we'll add RANSAC-based corner heuristics later.
  - This script avoids importing heavyweight detector libraries unless used interactively.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np


def extract_frame(path: Path, time_s: float | None = None) -> np.ndarray:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if time_s is None:
        frame_idx = max(0, nframes // 2)
    else:
        frame_idx = int(round(time_s * fps))
    frame_idx = min(max(0, frame_idx), max(0, nframes - 1))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Failed to read frame {frame_idx} from {path}")
    return frame


def detect_lines(frame: np.ndarray, player_bbox: Tuple[int, int, int, int] | None = None) -> Tuple[np.ndarray, List[Tuple[int, int, int, int]]]:
    # Convert to HSV and build a white-line color mask to reduce ceiling and dark-object edges
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # white-ish range in HSV (low saturation, high value)
    lower_white = np.array([0, 0, 180], dtype=np.uint8)
    upper_white = np.array([179, 60, 255], dtype=np.uint8)
    color_mask = cv2.inRange(hsv, lower_white, upper_white)

    # Build region mask. If player bbox is provided, expand it and focus around player side.
    h, w = frame.shape[:2]
    region_mask = np.zeros_like(color_mask)
    if player_bbox is not None:
        x, y, bw, bh = player_bbox
        # expand box (padding)
        pad_x = int(bw * 1.2)
        pad_y = int(bh * 2.0)
        x0 = max(0, x - pad_x)
        x1 = min(w, x + bw + pad_x)
        y0 = max(0, y - pad_y)
        y1 = min(h, y + bh + pad_y)
        region_mask[y0:y1, x0:x1] = 255
    else:
        # keep bottom 65% by default
        y0 = int(h * 0.35)
        region_mask[y0:, :] = 255

    # Combine color and region masks
    combined_mask = cv2.bitwise_and(color_mask, region_mask)

    # Edge detection on grayscale, but mask to likely court-lines
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150, apertureSize=3)
    # apply combined mask to edges
    edges_masked = cv2.bitwise_and(edges, combined_mask)

    lines_p = cv2.HoughLinesP(edges_masked, 1, np.pi / 180, threshold=60, minLineLength=80, maxLineGap=12)
    lines = []
    vis = cv2.cvtColor(edges_masked, cv2.COLOR_GRAY2BGR)
    if lines_p is not None:
        for x1, y1, x2, y2 in lines_p.reshape(-1, 4):
            lines.append((int(x1), int(y1), int(x2), int(y2)))
            cv2.line(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
    return vis, lines


def length_of_line(line: Tuple[int, int, int, int]) -> float:
    x1, y1, x2, y2 = line
    return float(((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5)


def angle_of_line_deg(line: Tuple[int, int, int, int]) -> float:
    x1, y1, x2, y2 = line
    ang = np.degrees(np.arctan2((y2 - y1), (x2 - x1)))
    # normalize to [0, 180)
    ang = ang % 180.0
    return float(ang)


def filter_lines(lines: List[Tuple[int, int, int, int]], n_keep_per_group: int = 4, angle_thresh: float = 12.0) -> List[Tuple[int, int, int, int]]:
    if not lines:
        return []
    angles = np.array([angle_of_line_deg(l) for l in lines])
    lengths = np.array([length_of_line(l) for l in lines])

    # Build angle histogram and pick two dominant peaks
    hist, bin_edges = np.histogram(angles, bins=36, range=(0.0, 180.0))
    peaks = np.argsort(hist)[-2:][::-1]
    peak_angles = [(bin_edges[p] + bin_edges[p + 1]) / 2.0 for p in peaks]

    groups: List[List[Tuple[int, int, int, int]]] = [[], []]
    for i, l in enumerate(lines):
        ang = angles[i]
        # find nearest peak
        dists = [abs((ang - pa + 90) % 180 - 90) for pa in peak_angles]
        grp = int(np.argmin(dists))
        if dists[grp] <= angle_thresh:
            groups[grp].append((l, lengths[i]))

    kept: List[Tuple[int, int, int, int]] = []
    for grp in groups:
        if not grp:
            continue
        # sort by length desc and keep top N
        grp_sorted = sorted(grp, key=lambda x: x[1], reverse=True)
        for l, _ in grp_sorted[:n_keep_per_group]:
            kept.append(l)

    return kept


def line_intersection(l1: Tuple[int, int, int, int], l2: Tuple[int, int, int, int]) -> Tuple[float, float] | None:
    x1, y1, x2, y2 = l1
    x3, y3, x4, y4 = l2
    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denom) < 1e-6:
        return None
    px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / denom
    py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / denom
    return float(px), float(py)


def detect_player_simple(frame: np.ndarray) -> Tuple[int, int, int, int] | None:
    """Detect a person using OpenCV HOGDescriptor (fast, classical). Returns bbox (x,y,w,h) or None."""
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    # resize down for speed if very large
    h, w = frame.shape[:2]
    scale = 1.0
    if max(h, w) > 1200:
        scale = 1200.0 / max(h, w)
        small = cv2.resize(frame, (int(w * scale), int(h * scale)))
    else:
        small = frame

    rects, weights = hog.detectMultiScale(small, winStride=(8, 8), padding=(8, 8), scale=1.05)
    if len(rects) == 0:
        return None
    # pick the largest detection
    areas = [rw * rh for (rx, ry, rw, rh) in rects]
    idx = int(np.argmax(areas))
    rx, ry, rw, rh = rects[idx]
    # scale bbox back to original frame size
    if scale != 1.0:
        rx = int(rx / scale)
        ry = int(ry / scale)
        rw = int(rw / scale)
        rh = int(rh / scale)
    return (int(rx), int(ry), int(rw), int(rh))


def compute_homography(src_pts: List[Tuple[float, float]], dst_size: Tuple[int, int] = (1000, 455)) -> Tuple[np.ndarray, np.ndarray]:
    # src_pts: list of four (x, y) in image pixels in order: top-left, top-right, bottom-right, bottom-left
    if len(src_pts) != 4:
        raise ValueError("src_pts must contain 4 points")
    dst_w, dst_h = dst_size
    # Target court rectangle in pixels (aspect ratio approximating 13.4m x 6.1m)
    dst = np.array([[0, 0], [dst_w - 1, 0], [dst_w - 1, dst_h - 1], [0, dst_h - 1]], dtype=np.float32)
    src = np.array(src_pts, dtype=np.float32)
    H, mask = cv2.findHomography(src, dst, method=cv2.RANSAC)
    return H, mask


def warp_frame(frame: np.ndarray, H: np.ndarray, dst_size: Tuple[int, int] = (1000, 455)) -> np.ndarray:
    return cv2.warpPerspective(frame, H, dst_size)


def save_lines_and_report(out_dir: Path, frame: np.ndarray, vis: np.ndarray, lines: List[Tuple[int, int, int, int]]):
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "frame.png").write_bytes(cv2.imencode('.png', frame)[1].tobytes())
    (out_dir / "edges_lines.png").write_bytes(cv2.imencode('.png', vis)[1].tobytes())
    # write simple JSON lines
    json_lines = [list(map(int, l)) for l in lines]
    (out_dir / "lines.json").write_text(json.dumps(json_lines, indent=2))


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Phase 1: court detection, homography, and simple player helpers")
    parser.add_argument("video", help="Path to video file")
    parser.add_argument("--frame-time", type=float, default=None, help="Time (s) of frame to sample; default=middle frame")
    parser.add_argument("--out-dir", type=Path, default=Path("data/processed/phase1"), help="Output directory")
    parser.add_argument("--src-pts", type=Path, default=None, help="JSON file with 4 source points (x,y) in image pixels")
    parser.add_argument("--warp-out", type=Path, default=None, help="If provided, write warped bird's-eye frame here")
    args = parser.parse_args(argv)

    video = Path(args.video)
    if not video.is_file():
        print(f"Video not found: {video}", file=sys.stderr)
        return 2

    frame = extract_frame(video, time_s=args.frame_time)
    # Detect player first and then focus the line detector around them
    player_bbox = None
    try:
        player_bbox = detect_player_simple(frame)
    except Exception:
        player_bbox = None

    vis, lines = detect_lines(frame, player_bbox=player_bbox)
    save_lines_and_report(args.out_dir, frame, vis, lines)
    print(f"Wrote frame and line visualizations to {args.out_dir}")

    # save player bbox if present
    if player_bbox is not None:
        (args.out_dir / "player_bbox.json").write_text(json.dumps(list(map(int, player_bbox)), indent=2))
        # write overlay image
        ov = frame.copy()
        x, y, bw, bh = player_bbox
        cv2.rectangle(ov, (x, y), (x + bw, y + bh), (0, 0, 255), 3)
        (args.out_dir / "player_bbox.png").write_bytes(cv2.imencode('.png', ov)[1].tobytes())

    # Filter detected Hough lines to the most likely court lines (by angle + length)
    filtered = filter_lines(lines)
    if filtered:
        vis_f = frame.copy()
        for x1, y1, x2, y2 in filtered:
            cv2.line(vis_f, (x1, y1), (x2, y2), (0, 255, 0), 4)
        (args.out_dir / "edges_lines_filtered.png").write_bytes(cv2.imencode('.png', vis_f)[1].tobytes())
        (args.out_dir / "lines_filtered.json").write_text(json.dumps([list(map(int, l)) for l in filtered], indent=2))

        # Attempt to compute four corner guesses by intersecting two representative lines
        corners = []
        # take two from each group if available (we kept up to n_keep_per_group per group)
        if len(filtered) >= 4:
            # naive: pair first two with last two
            a1, a2, b1, b2 = filtered[0], filtered[1], filtered[2], filtered[3]
            for la in (a1, a2):
                for lb in (b1, b2):
                    pt = line_intersection(la, lb)
                    if pt:
                        corners.append([pt[0], pt[1]])
        if corners:
            (args.out_dir / "corners.json").write_text(json.dumps(corners, indent=2))
            print(f"Wrote filtered lines and corner guesses to {args.out_dir}")
        else:
            print("Filtered lines found but could not compute corner intersections.")

    if args.src_pts:
        pts = json.loads(args.src_pts.read_text())
        if not isinstance(pts, list) or len(pts) != 4:
            print("src_pts must be a JSON list of four [x,y] points", file=sys.stderr)
            return 3
        H, mask = compute_homography(pts)
        warped = warp_frame(frame, H)
        outp = args.warp_out or (args.out_dir / "warped.png")
        outp.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(outp), warped)
        print(f"Warped image written to {outp}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
