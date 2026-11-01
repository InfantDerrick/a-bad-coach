#!/usr/bin/env python3
"""Build an offline movement report from one fixed-camera badminton clip.

This is the first Phase A vertical slice. It intentionally accepts an explicit
court calibration and player seed instead of pretending the current automatic
detectors are reliable enough. The outputs are local JSON, HTML, PNG, and MP4
artifacts that can be inspected and corrected.

Example:
    python scripts/analyze_movement.py \
      data/raw/bad_1.MOV \
      --calibration data/config/sample_calibration.json \
      --player-seed data/config/sample_player_seed.json
"""
from __future__ import annotations

import argparse
import csv
import html
import json
import math
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np


COURT_WIDTH_M = 6.10
COURT_LENGTH_M = 13.40


@dataclass(frozen=True)
class VideoMetadata:
    path: str
    width: int
    height: int
    fps: float
    frame_count: int
    duration_s: float
    size_bytes: int
    codec: str | None
    average_fps: float | None


@dataclass(frozen=True)
class Calibration:
    image_points: np.ndarray
    court_points_m: np.ndarray
    homography: np.ndarray
    reprojection_rmse_m: float
    status: str
    notes: list[str]


def _rational_to_float(value: str | None) -> float | None:
    if not value or value == "0/0":
        return None
    if "/" not in value:
        return float(value)
    numerator, denominator = value.split("/", 1)
    denominator_f = float(denominator)
    return float(numerator) / denominator_f if denominator_f else None


def probe_video(path: Path) -> VideoMetadata:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {path}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()

    codec: str | None = None
    average_fps: float | None = None
    ffprobe = shutil.which("ffprobe")
    if ffprobe:
        command = [
            ffprobe,
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=codec_name,avg_frame_rate",
            "-of",
            "json",
            str(path),
        ]
        try:
            result = subprocess.run(command, check=True, capture_output=True, text=True)
            streams = json.loads(result.stdout).get("streams", [])
            if streams:
                codec = streams[0].get("codec_name")
                average_fps = _rational_to_float(streams[0].get("avg_frame_rate"))
        except (OSError, subprocess.CalledProcessError, json.JSONDecodeError):
            pass

    duration_s = frame_count / fps if fps > 0 else 0.0
    return VideoMetadata(
        path=str(path),
        width=width,
        height=height,
        fps=fps,
        frame_count=frame_count,
        duration_s=duration_s,
        size_bytes=path.stat().st_size,
        codec=codec,
        average_fps=average_fps,
    )


def capture_quality(metadata: VideoMetadata) -> dict[str, Any]:
    long_side = max(metadata.width, metadata.height)
    short_side = min(metadata.width, metadata.height)
    core_supported = short_side >= 1080 and long_side >= 1920 and metadata.fps >= 50
    shuttle_speed_supported = metadata.fps >= 100
    warnings: list[str] = []
    if metadata.fps < 50:
        warnings.append("Below the recommended 60 FPS for movement/contact timing.")
    if short_side < 1080:
        warnings.append("Below the recommended 1080p short edge.")
    if metadata.fps < 100:
        warnings.append("Shuttle trajectory/speed beta requires at least 100 FPS.")
    warnings.append("Motion blur and shutter speed are not yet measured automatically.")
    return {
        "core_movement_tier": "supported" if core_supported else "provisional",
        "shuttle_speed_tier": "supported" if shuttle_speed_supported else "unavailable",
        "warnings": warnings,
    }


def load_calibration(path: Path) -> Calibration:
    payload = json.loads(path.read_text())
    image_points = np.asarray(payload["image_points"], dtype=np.float32)
    court_points = np.asarray(payload["court_points_m"], dtype=np.float32)
    if image_points.ndim != 2 or image_points.shape[1] != 2:
        raise ValueError("calibration image_points must be an N x 2 array")
    if court_points.shape != image_points.shape or len(image_points) < 4:
        raise ValueError("calibration needs at least four matching image/court points")

    homography, _ = cv2.findHomography(image_points, court_points, method=0)
    if homography is None:
        raise ValueError("could not compute calibration homography")
    projected = cv2.perspectiveTransform(image_points.reshape(-1, 1, 2), homography).reshape(-1, 2)
    rmse = float(np.sqrt(np.mean(np.sum((projected - court_points) ** 2, axis=1))))

    notes = [str(note) for note in payload.get("notes", [])]
    status = str(payload.get("status", "unverified"))
    spans_full_width = float(np.ptp(court_points[:, 0])) >= COURT_WIDTH_M * 0.95
    spans_full_length = float(np.ptp(court_points[:, 1])) >= COURT_LENGTH_M * 0.95
    if not spans_full_width or not spans_full_length:
        notes.append("Calibration anchors do not span the full doubles court.")
        status = "provisional"
    if rmse > 0.10:
        notes.append(f"Anchor reprojection RMSE is high ({rmse:.3f} m).")
        status = "provisional"

    return Calibration(
        image_points=image_points,
        court_points_m=court_points,
        homography=homography,
        reprojection_rmse_m=rmse,
        status=status,
        notes=notes,
    )


def load_player_seed(path: Path) -> tuple[int, tuple[float, float, float, float], str, list[str]]:
    payload = json.loads(path.read_text())
    frame_index = int(payload.get("frame_index", 0))
    bbox = tuple(float(value) for value in payload["bbox_xywh"])
    if len(bbox) != 4 or bbox[2] <= 0 or bbox[3] <= 0:
        raise ValueError("player seed bbox_xywh must be [x, y, width, height]")
    return frame_index, bbox, str(payload.get("status", "unverified")), list(payload.get("notes", []))


def create_tracker() -> Any:
    if hasattr(cv2, "TrackerCSRT_create"):
        return cv2.TrackerCSRT_create()
    legacy = getattr(cv2, "legacy", None)
    if legacy is not None and hasattr(legacy, "TrackerCSRT_create"):
        return legacy.TrackerCSRT_create()
    raise RuntimeError("OpenCV CSRT tracker is unavailable; install opencv-contrib-python")


def resize_for_processing(frame: np.ndarray, width: int) -> tuple[np.ndarray, float]:
    scale = min(1.0, width / frame.shape[1])
    if scale == 1.0:
        return frame, scale
    resized = cv2.resize(
        frame,
        (int(round(frame.shape[1] * scale)), int(round(frame.shape[0] * scale))),
        interpolation=cv2.INTER_AREA,
    )
    return resized, scale


def transform_points(points: np.ndarray, homography: np.ndarray) -> np.ndarray:
    return cv2.perspectiveTransform(points.astype(np.float32).reshape(-1, 1, 2), homography).reshape(-1, 2)


def smooth_series(values: np.ndarray, window: int) -> np.ndarray:
    if len(values) < 3 or window <= 1:
        return values.copy()
    window = min(window, len(values))
    if window % 2 == 0:
        window -= 1
    pad = window // 2
    kernel = np.ones(window, dtype=np.float64) / window
    result = np.empty_like(values, dtype=np.float64)
    for axis in range(values.shape[1]):
        padded = np.pad(values[:, axis], (pad, pad), mode="edge")
        result[:, axis] = np.convolve(padded, kernel, mode="valid")
    return result


def movement_metrics(points_m: np.ndarray, fps: float, valid: np.ndarray) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    if len(points_m) == 0:
        raise ValueError("no tracked points")
    smoothed = smooth_series(points_m, max(3, int(round(fps * 0.20))))
    deltas = np.diff(smoothed, axis=0, prepend=smoothed[[0]])
    step_distance = np.linalg.norm(deltas, axis=1)
    pair_valid = valid & np.roll(valid, 1)
    pair_valid[0] = False
    # Reject implausible jumps from tracker drift. This threshold is deliberately
    # generous; rejected frames are exposed in report coverage.
    plausible = step_distance * fps <= 12.0
    accepted = pair_valid & plausible
    speeds = np.zeros(len(points_m), dtype=np.float64)
    speeds[accepted] = step_distance[accepted] * fps
    distance_m = float(step_distance[accepted].sum())

    in_bounds = (
        (smoothed[:, 0] >= -0.25)
        & (smoothed[:, 0] <= COURT_WIDTH_M + 0.25)
        & (smoothed[:, 1] >= -0.25)
        & (smoothed[:, 1] <= COURT_LENGTH_M + 0.25)
    )
    measured_speeds = speeds[accepted]
    metrics = {
        "distance_m": round(distance_m, 3),
        "average_moving_speed_mps": round(float(measured_speeds[measured_speeds > 0.15].mean()), 3)
        if np.any(measured_speeds > 0.15)
        else 0.0,
        "speed_p50_mps": round(float(np.percentile(measured_speeds, 50)), 3) if len(measured_speeds) else 0.0,
        "speed_p90_mps": round(float(np.percentile(measured_speeds, 90)), 3) if len(measured_speeds) else 0.0,
        "peak_speed_mps": round(float(measured_speeds.max()), 3) if len(measured_speeds) else 0.0,
        "tracked_frames": int(valid.sum()),
        "tracking_coverage": round(float(valid.mean()), 4),
        "in_court_coverage": round(float((valid & in_bounds).sum() / max(1, valid.sum())), 4),
        "rejected_jump_frames": int((pair_valid & ~plausible).sum()),
    }
    return metrics, smoothed, speeds


def six_zone_occupancy(points_m: np.ndarray, valid: np.ndarray) -> dict[str, float]:
    labels = [
        "left_front",
        "right_front",
        "left_mid",
        "right_mid",
        "left_rear",
        "right_rear",
    ]
    counts = dict.fromkeys(labels, 0)
    usable = 0
    for (x_m, y_m), is_valid in zip(points_m, valid):
        if not is_valid or not (0 <= x_m <= COURT_WIDTH_M and 0 <= y_m <= COURT_LENGTH_M):
            continue
        side = "left" if x_m < COURT_WIDTH_M / 2 else "right"
        third = "front" if y_m < COURT_LENGTH_M / 3 else "mid" if y_m < 2 * COURT_LENGTH_M / 3 else "rear"
        counts[f"{side}_{third}"] += 1
        usable += 1
    return {label: round(counts[label] / usable, 4) if usable else 0.0 for label in labels}


def court_outline_image_points(calibration: Calibration) -> np.ndarray:
    court = np.array(
        [[0, 0], [COURT_WIDTH_M, 0], [COURT_WIDTH_M, COURT_LENGTH_M], [0, COURT_LENGTH_M]],
        dtype=np.float32,
    )
    inverse = np.linalg.inv(calibration.homography)
    return transform_points(court, inverse)


def draw_court_top_down(
    path: Path,
    points_m: np.ndarray,
    valid: np.ndarray,
    occupancy: dict[str, float],
) -> None:
    scale = 80
    margin = 55
    width = int(COURT_WIDTH_M * scale + margin * 2)
    height = int(COURT_LENGTH_M * scale + margin * 2)
    canvas = np.full((height, width, 3), (28, 82, 58), dtype=np.uint8)

    def pt(x_m: float, y_m: float) -> tuple[int, int]:
        return int(margin + x_m * scale), int(margin + y_m * scale)

    zone_names = [
        ("left_front", 0, 0), ("right_front", 1, 0),
        ("left_mid", 0, 1), ("right_mid", 1, 1),
        ("left_rear", 0, 2), ("right_rear", 1, 2),
    ]
    for name, col, row in zone_names:
        x0 = margin + int(col * COURT_WIDTH_M / 2 * scale)
        x1 = margin + int((col + 1) * COURT_WIDTH_M / 2 * scale)
        y0 = margin + int(row * COURT_LENGTH_M / 3 * scale)
        y1 = margin + int((row + 1) * COURT_LENGTH_M / 3 * scale)
        intensity = int(35 + min(1.0, occupancy[name] * 4) * 150)
        overlay = canvas.copy()
        cv2.rectangle(overlay, (x0, y0), (x1, y1), (30, 70 + intensity, 170), -1)
        cv2.addWeighted(overlay, 0.35, canvas, 0.65, 0, canvas)

    white = (245, 245, 245)
    cv2.rectangle(canvas, pt(0, 0), pt(COURT_WIDTH_M, COURT_LENGTH_M), white, 3)
    cv2.line(canvas, pt(0, COURT_LENGTH_M / 2), pt(COURT_WIDTH_M, COURT_LENGTH_M / 2), white, 3)
    cv2.line(canvas, pt(COURT_WIDTH_M / 2, 0), pt(COURT_WIDTH_M / 2, COURT_LENGTH_M), white, 2)
    for y_m in (1.98, COURT_LENGTH_M - 1.98):
        cv2.line(canvas, pt(0, y_m), pt(COURT_WIDTH_M, y_m), white, 2)

    usable = points_m[valid]
    if len(usable) > 1:
        pixels = np.array([pt(float(x), float(y)) for x, y in usable], dtype=np.int32)
        cv2.polylines(canvas, [pixels], False, (36, 220, 255), 3, cv2.LINE_AA)
        cv2.circle(canvas, tuple(pixels[0]), 7, (80, 255, 80), -1)
        cv2.circle(canvas, tuple(pixels[-1]), 7, (80, 80, 255), -1)

    cv2.putText(canvas, "FAR", (margin, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, white, 2, cv2.LINE_AA)
    cv2.putText(canvas, "NEAR", (margin, height - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.8, white, 2, cv2.LINE_AA)
    cv2.imwrite(str(path), canvas)


def draw_overlay_frame(
    frame: np.ndarray,
    bbox: tuple[float, float, float, float] | None,
    court_outline: np.ndarray,
    trail: Iterable[tuple[float, float]],
    speed_mps: float,
    scale: float,
    calibration_status: str,
) -> np.ndarray:
    output = frame.copy()
    outline = np.round(court_outline * scale).astype(np.int32)
    cv2.polylines(output, [outline], True, (0, 210, 255), 3, cv2.LINE_AA)
    trail_points = np.array([(int(x * scale), int(y * scale)) for x, y in trail], dtype=np.int32)
    if len(trail_points) > 1:
        cv2.polylines(output, [trail_points], False, (255, 180, 30), 3, cv2.LINE_AA)
    if bbox is not None:
        x, y, width, height = bbox
        x0, y0 = int(x * scale), int(y * scale)
        x1, y1 = int((x + width) * scale), int((y + height) * scale)
        cv2.rectangle(output, (x0, y0), (x1, y1), (80, 255, 80), 3)
        cv2.circle(output, (int((x + width / 2) * scale), y1), 7, (40, 60, 255), -1)
    cv2.rectangle(output, (15, 15), (510, 104), (12, 18, 24), -1)
    cv2.putText(output, f"Player speed: {speed_mps:.2f} m/s", (30, 51), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (245, 245, 245), 2, cv2.LINE_AA)
    cv2.putText(output, f"Calibration: {calibration_status}", (30, 86), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (0, 210, 255), 2, cv2.LINE_AA)
    return output


def render_html(path: Path, report: dict[str, Any]) -> None:
    video = report["video"]
    metrics = report["movement"]
    quality = report["capture_quality"]
    calibration = report["calibration"]
    warnings = quality["warnings"] + calibration["notes"] + report["player_seed"]["notes"]
    warning_items = "".join(f"<li>{html.escape(str(item))}</li>" for item in warnings)
    zone_rows = "".join(
        f"<tr><td>{html.escape(name.replace('_', ' ').title())}</td><td>{value * 100:.1f}%</td></tr>"
        for name, value in report["zone_occupancy"].items()
    )
    document = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Badminton movement report</title>
  <style>
    :root {{ color-scheme: dark; --bg:#0c1118; --card:#151d28; --muted:#9cafc3; --accent:#46d6a0; --warn:#ffd166; }}
    * {{ box-sizing:border-box; }} body {{ margin:0; font-family:Inter,system-ui,sans-serif; background:var(--bg); color:#f5f7fa; }}
    main {{ width:min(1180px,94vw); margin:36px auto 80px; }} h1 {{ margin-bottom:6px; }} .sub {{ color:var(--muted); margin-top:0; }}
    .notice {{ padding:16px 18px; border:1px solid #765f23; background:#2a2414; border-radius:12px; }}
    .grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(180px,1fr)); gap:14px; margin:22px 0; }}
    .card,.panel {{ background:var(--card); border:1px solid #253244; border-radius:14px; padding:18px; }}
    .metric {{ font-size:1.85rem; font-weight:750; color:var(--accent); }} .label {{ color:var(--muted); font-size:.9rem; }}
    .columns {{ display:grid; grid-template-columns:minmax(0,1.45fr) minmax(300px,.75fr); gap:18px; }}
    video,img {{ display:block; width:100%; border-radius:10px; background:#05080c; }} table {{ width:100%; border-collapse:collapse; }} td {{ padding:9px 4px; border-bottom:1px solid #263446; }} td:last-child {{ text-align:right; }}
    code {{ color:#b9e6ff; }} @media(max-width:800px) {{ .columns {{ grid-template-columns:1fr; }} }}
  </style>
</head>
<body><main>
  <h1>Offline movement report</h1>
  <p class="sub">{html.escape(Path(video['path']).name)} · {video['width']}×{video['height']} · {video['fps']:.2f} FPS · {video['duration_s']:.2f}s</p>
  <div class="notice"><strong>Prototype status:</strong> calibration is <code>{html.escape(calibration['status'])}</code>, player seed is <code>{html.escape(report['player_seed']['status'])}</code>, and movement metrics are <code>{html.escape(metrics['metric_status'])}</code>. Metrics are engineering diagnostics until setup is manually verified.</div>
  <section class="grid">
    <div class="card"><div class="metric">{metrics['distance_m']:.2f} m</div><div class="label">Tracked distance</div></div>
    <div class="card"><div class="metric">{metrics['average_moving_speed_mps']:.2f} m/s</div><div class="label">Average moving speed</div></div>
    <div class="card"><div class="metric">{metrics['speed_p90_mps']:.2f} m/s</div><div class="label">90th percentile speed</div></div>
    <div class="card"><div class="metric">{metrics['tracking_coverage'] * 100:.1f}%</div><div class="label">Tracking coverage</div></div>
  </section>
  <section class="columns">
    <div class="panel"><h2>Evidence overlay</h2><video controls preload="metadata" src="movement_overlay.mp4"></video></div>
    <div class="panel"><h2>Top-down path</h2><img src="court_path.png" alt="Top-down player path"></div>
  </section>
  <section class="columns" style="margin-top:18px">
    <div class="panel"><h2>Six-zone occupancy</h2><table>{zone_rows}</table></div>
    <div class="panel"><h2>Quality notes</h2><ul>{warning_items}</ul><p>Core tier: <code>{quality['core_movement_tier']}</code><br>Shuttle speed: <code>{quality['shuttle_speed_tier']}</code></p></div>
  </section>
</main></body></html>"""
    path.write_text(document)


def analyze(
    video_path: Path,
    calibration_path: Path,
    seed_path: Path,
    output_dir: Path,
    processing_width: int,
) -> dict[str, Any]:
    metadata = probe_video(video_path)
    calibration = load_calibration(calibration_path)
    seed_frame, seed_bbox, seed_status, seed_notes = load_player_seed(seed_path)
    if seed_frame != 0:
        raise ValueError("the current vertical slice requires a frame_index 0 player seed")

    output_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    ok, first_frame = cap.read()
    if not ok:
        cap.release()
        raise RuntimeError("could not read first frame")
    first_small, scale = resize_for_processing(first_frame, processing_width)
    scaled_seed = tuple(int(round(value * scale)) for value in seed_bbox)
    tracker = create_tracker()
    tracker.init(first_small, scaled_seed)

    outline = court_outline_image_points(calibration)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    overlay_path = output_dir / "movement_overlay.mp4"
    writer = cv2.VideoWriter(
        str(overlay_path),
        fourcc,
        metadata.fps,
        (first_small.shape[1], first_small.shape[0]),
    )
    if not writer.isOpened():
        cap.release()
        raise RuntimeError("could not create overlay video")

    bboxes: list[tuple[float, float, float, float] | None] = []
    foot_pixels: list[tuple[float, float]] = []
    valid: list[bool] = []
    frame = first_frame
    frame_index = 0
    while True:
        small, current_scale = resize_for_processing(frame, processing_width)
        if frame_index == 0:
            tracked_ok = True
            tracked_scaled = scaled_seed
        else:
            tracked_ok, tracked_scaled = tracker.update(small)
        if tracked_ok:
            x_s, y_s, w_s, h_s = (float(value) for value in tracked_scaled)
            bbox = (x_s / current_scale, y_s / current_scale, w_s / current_scale, h_s / current_scale)
            foot = (bbox[0] + bbox[2] / 2, bbox[1] + bbox[3])
        else:
            bbox = None
            foot = foot_pixels[-1] if foot_pixels else (float("nan"), float("nan"))
        bboxes.append(bbox)
        foot_pixels.append(foot)
        valid.append(bool(tracked_ok))
        frame_index += 1
        ok, frame = cap.read()
        if not ok:
            break
    cap.release()

    foot_array = np.asarray(foot_pixels, dtype=np.float64)
    valid_array = np.asarray(valid, dtype=bool) & np.isfinite(foot_array).all(axis=1)
    if not np.any(valid_array):
        writer.release()
        raise RuntimeError("player tracker produced no valid frames")
    first_valid = foot_array[valid_array][0]
    foot_array[~valid_array] = first_valid
    court_points = transform_points(foot_array, calibration.homography).astype(np.float64)
    metrics, smoothed_points, speeds = movement_metrics(court_points, metadata.fps, valid_array)
    occupancy = six_zone_occupancy(smoothed_points, valid_array)
    metrics["metric_status"] = (
        "verified"
        if calibration.status == "verified"
        and seed_status == "verified"
        and metrics["in_court_coverage"] >= 0.95
        else "diagnostic_only"
    )

    overlay_cap = cv2.VideoCapture(str(video_path))
    trail: list[tuple[float, float]] = []
    for index in range(len(bboxes)):
        frame_ok, overlay_frame = overlay_cap.read()
        if not frame_ok:
            break
        small, _ = resize_for_processing(overlay_frame, processing_width)
        if valid_array[index]:
            trail.append(tuple(foot_array[index]))
        if len(trail) > int(metadata.fps * 2.0):
            trail.pop(0)
        annotated = draw_overlay_frame(
            small,
            bboxes[index],
            outline,
            trail,
            float(speeds[index]),
            scale,
            calibration.status,
        )
        writer.write(annotated)
    overlay_cap.release()
    writer.release()

    draw_court_top_down(output_dir / "court_path.png", smoothed_points, valid_array, occupancy)

    with (output_dir / "trajectory.csv").open("w", newline="") as trajectory_file:
        fieldnames = [
            "frame_index", "timestamp_s", "tracking_valid", "bbox_x", "bbox_y", "bbox_width", "bbox_height",
            "foot_pixel_x", "foot_pixel_y", "court_x_m", "court_y_m", "smoothed_x_m", "smoothed_y_m", "speed_mps",
        ]
        writer_csv = csv.DictWriter(trajectory_file, fieldnames=fieldnames)
        writer_csv.writeheader()
        for index, bbox in enumerate(bboxes):
            bbox_values = bbox or (None, None, None, None)
            writer_csv.writerow({
                "frame_index": index,
                "timestamp_s": round(index / metadata.fps, 6),
                "tracking_valid": bool(valid_array[index]),
                "bbox_x": bbox_values[0],
                "bbox_y": bbox_values[1],
                "bbox_width": bbox_values[2],
                "bbox_height": bbox_values[3],
                "foot_pixel_x": foot_array[index, 0],
                "foot_pixel_y": foot_array[index, 1],
                "court_x_m": court_points[index, 0],
                "court_y_m": court_points[index, 1],
                "smoothed_x_m": smoothed_points[index, 0],
                "smoothed_y_m": smoothed_points[index, 1],
                "speed_mps": speeds[index],
            })

    report = {
        "schema_version": "0.1.0",
        "status": "provisional" if calibration.status != "verified" or seed_status != "verified" else "verified",
        "video": asdict(metadata),
        "capture_quality": capture_quality(metadata),
        "calibration": {
            "source": str(calibration_path),
            "status": calibration.status,
            "reprojection_rmse_m": round(calibration.reprojection_rmse_m, 6),
            "notes": calibration.notes,
        },
        "player_seed": {
            "source": str(seed_path),
            "frame_index": seed_frame,
            "bbox_xywh": list(seed_bbox),
            "status": seed_status,
            "notes": seed_notes,
        },
        "movement": metrics,
        "zone_occupancy": occupancy,
        "artifacts": {
            "overlay_video": "movement_overlay.mp4",
            "court_path": "court_path.png",
            "trajectory_csv": "trajectory.csv",
            "report_html": "report.html",
        },
        "limitations": [
            "Bounding-box bottom center is a provisional foot-point estimate, not pose-derived foot contact.",
            "CSRT is a single-object baseline and does not recover automatically after a track failure.",
            "No rally, shot, pose, shuttle, or split-step events are produced in this phase.",
            "Movement metrics are not coaching claims until calibration and player identity are verified.",
        ],
    }
    (output_dir / "report.json").write_text(json.dumps(report, indent=2))
    render_html(output_dir / "report.html", report)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate an offline calibrated movement report")
    parser.add_argument("video", type=Path)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--player-seed", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("data/processed/movement_report"))
    parser.add_argument("--processing-width", type=int, default=540)
    args = parser.parse_args(argv)
    for required in (args.video, args.calibration, args.player_seed):
        if not required.is_file():
            parser.error(f"file not found: {required}")
    if args.processing_width < 240:
        parser.error("--processing-width must be at least 240")
    try:
        report = analyze(args.video, args.calibration, args.player_seed, args.out_dir, args.processing_width)
    except (ValueError, RuntimeError, OSError, json.JSONDecodeError) as exc:
        print(f"Analysis failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps({"status": report["status"], "output": str(args.out_dir), "movement": report["movement"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
