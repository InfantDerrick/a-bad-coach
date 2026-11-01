#!/usr/bin/env python3
"""Generate canonical-court perspective and partial-view pretraining data.

Synthetic data teaches line topology and extreme geometry cheaply. It must be
followed by fine-tuning and evaluation on real, independently filmed venues.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pipeline.court_keypoints import (  # noqa: E402
    COURT_LENGTH_M,
    COURT_WIDTH_M,
    DOUBLES_LONG_SERVICE_OFFSET_M,
    LANDMARKS,
    NET_Y_M,
    SHORT_SERVICE_OFFSET_M,
)


SINGLES_INSET_M = 0.46


def court_segments() -> list[tuple[tuple[float, float], tuple[float, float]]]:
    left, right, length = 0.0, COURT_WIDTH_M, COURT_LENGTH_M
    near_short = NET_Y_M + SHORT_SERVICE_OFFSET_M
    far_short = NET_Y_M - SHORT_SERVICE_OFFSET_M
    near_long = length - DOUBLES_LONG_SERVICE_OFFSET_M
    far_long = DOUBLES_LONG_SERVICE_OFFSET_M
    segments = [
        ((left, 0), (right, 0)), ((left, length), (right, length)),
        ((left, 0), (left, length)), ((right, 0), (right, length)),
        ((SINGLES_INSET_M, 0), (SINGLES_INSET_M, length)),
        ((right - SINGLES_INSET_M, 0), (right - SINGLES_INSET_M, length)),
        ((left, far_long), (right, far_long)), ((left, near_long), (right, near_long)),
        ((left, far_short), (right, far_short)), ((left, near_short), (right, near_short)),
        ((left, NET_Y_M), (right, NET_Y_M)),
        ((right / 2, far_long), (right / 2, far_short)),
        ((right / 2, near_short), (right / 2, near_long)),
    ]
    return segments


def random_quad(width: int, height: int, rng: random.Random) -> np.ndarray:
    center = rng.uniform(-0.10, 1.10) * width
    far_y = rng.uniform(0.08, 0.52) * height
    near_y = rng.uniform(0.66, 1.18) * height
    far_width = rng.uniform(0.20, 0.78) * width
    near_width = rng.uniform(0.65, 1.55) * width
    skew = rng.uniform(-0.22, 0.22) * width
    roll_far = rng.uniform(-0.06, 0.06) * height
    roll_near = rng.uniform(-0.08, 0.08) * height
    return np.asarray(
        [
            [center - far_width / 2, far_y - roll_far],
            [center + far_width / 2, far_y + roll_far],
            [center + skew + near_width / 2, near_y + roll_near],
            [center + skew - near_width / 2, near_y - roll_near],
        ],
        dtype=np.float32,
    )


def transform_xy(point: tuple[float, float], homography: np.ndarray) -> tuple[float, float]:
    array = np.asarray([[point]], dtype=np.float32)
    transformed = cv2.perspectiveTransform(array, homography).reshape(2)
    return float(transformed[0]), float(transformed[1])


def background(width: int, height: int, rng: random.Random) -> np.ndarray:
    palette = [(48, 116, 82), (104, 126, 104), (124, 105, 66), (88, 128, 142), (72, 98, 118)]
    base = np.full((height, width, 3), rng.choice(palette), dtype=np.uint8)
    gradient = np.linspace(rng.uniform(0.78, 1.05), rng.uniform(0.85, 1.18), height, dtype=np.float32)[:, None, None]
    noise = np.random.default_rng(rng.randrange(2**32)).normal(0, rng.uniform(2, 10), base.shape).astype(np.float32)
    return np.clip(base.astype(np.float32) * gradient + noise, 0, 255).astype(np.uint8)


def render_scene(width: int, height: int, quad: np.ndarray, rng: random.Random) -> tuple[np.ndarray, dict[str, list[float] | None]]:
    image = background(width, height, rng)
    court_corners = np.asarray(
        [[0, 0], [COURT_WIDTH_M, 0], [COURT_WIDTH_M, COURT_LENGTH_M], [0, COURT_LENGTH_M]],
        dtype=np.float32,
    )
    homography = cv2.getPerspectiveTransform(court_corners, quad)
    line_color = rng.choice([(238, 238, 238), (210, 235, 238), (90, 220, 248)])
    thickness = rng.randint(2, 6)
    for start, end in court_segments():
        p0 = transform_xy(start, homography)
        p1 = transform_xy(end, homography)
        cv2.line(image, tuple(np.round(p0).astype(int)), tuple(np.round(p1).astype(int)), line_color, thickness, cv2.LINE_AA)

    # Net, players, equipment, and crop-edge clutter provide structured occlusion.
    if rng.random() < 0.85:
        net_left = transform_xy((0, NET_Y_M), homography)
        net_right = transform_xy((COURT_WIDTH_M, NET_Y_M), homography)
        net_offset = rng.randint(10, 45)
        cv2.line(image, (round(net_left[0]), round(net_left[1] - net_offset)), (round(net_right[0]), round(net_right[1] - net_offset)), (225, 225, 225), rng.randint(1, 4), cv2.LINE_AA)
    for _ in range(rng.randint(0, 4)):
        x = rng.randint(-width // 10, width)
        y = rng.randint(height // 4, height)
        body_w = rng.randint(max(8, width // 35), max(12, width // 10))
        body_h = rng.randint(max(20, height // 14), max(30, height // 4))
        color = tuple(rng.randint(15, 85) for _ in range(3))
        cv2.ellipse(image, (x, y - body_h // 2), (body_w // 2, body_h // 2), rng.randint(-20, 20), 0, 360, color, -1)
    if rng.random() < 0.7:
        kernel = rng.choice([1, 1, 3, 5])
        if kernel > 1:
            image = cv2.GaussianBlur(image, (kernel, kernel), 0)
    alpha = rng.uniform(0.75, 1.22)
    beta = rng.uniform(-24, 24)
    image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

    points: dict[str, list[float] | None] = {}
    for landmark in LANDMARKS:
        x, y = transform_xy((landmark.court_x_m, landmark.court_y_m), homography)
        points[landmark.name] = [round(x, 3), round(y, 3)] if 0 <= x < width and 0 <= y < height else None
    return image, points


def generate(dataset_dir: Path, samples: int, scenes: int, width: int, height: int, seed: int) -> dict:
    rng = random.Random(seed)
    images_dir = dataset_dir / "images"
    annotations_dir = dataset_dir / "annotations"
    images_dir.mkdir(parents=True, exist_ok=True)
    annotations_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = dataset_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {"schema_version": "1.0", "items": []}
    existing = {item["image"]: item for item in manifest["items"]}
    scene_quads = [random_quad(width, height, rng) for _ in range(scenes)]
    for index in range(samples):
        scene_index = index % scenes
        base_quad = scene_quads[scene_index]
        jitter = np.asarray([[rng.uniform(-12, 12), rng.uniform(-12, 12)] for _ in range(4)], dtype=np.float32)
        image, points = render_scene(width, height, base_quad + jitter, rng)
        filename = f"synthetic-s{scene_index:04d}-f{index:06d}.jpg"
        cv2.imwrite(str(images_dir / filename), image, [cv2.IMWRITE_JPEG_QUALITY, rng.randint(72, 96)])
        item = {
            "image": filename,
            "source_video": f"synthetic-scene-{scene_index:04d}",
            "source_path": "synthetic://canonical-court",
            "frame_index": index,
            "timestamp_s": None,
            "width": width,
            "height": height,
        }
        annotation = {
            **item,
            "image_width": width,
            "image_height": height,
            "points": points,
            "complete": True,
            "synthetic": True,
        }
        existing[filename] = item
        (annotations_dir / f"{Path(filename).stem}.json").write_text(json.dumps(annotation, indent=2))
    manifest["items"] = sorted(existing.values(), key=lambda item: (item["source_video"], item["frame_index"]))
    manifest_path.write_text(json.dumps(manifest, indent=2))
    return {"generated": samples, "scenes": scenes, "dataset_items": len(manifest["items"]), "dataset": str(dataset_dir)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate synthetic partial-view badminton courts")
    parser.add_argument("--dataset", type=Path, default=Path("data/court_dataset"))
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--scenes", type=int, default=100)
    parser.add_argument("--width", type=int, default=540)
    parser.add_argument("--height", type=int, default=960)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args(argv)
    if args.samples < 1 or args.scenes < 2 or args.scenes > args.samples:
        parser.error("require samples >= scenes >= 2")
    result = generate(args.dataset, args.samples, args.scenes, args.width, args.height, args.seed)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
