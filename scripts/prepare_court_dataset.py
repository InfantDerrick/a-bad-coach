#!/usr/bin/env python3
"""Sample a video into a grouped court-keypoint annotation dataset."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import cv2


def video_id(path: Path) -> str:
    digest = hashlib.sha256(str(path.resolve()).encode("utf-8")).hexdigest()[:10]
    return f"{path.stem.lower().replace(' ', '_')}-{digest}"


def prepare(video: Path, dataset_dir: Path, count: int, jpeg_quality: int) -> dict:
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video}")
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if frame_count <= 0:
        cap.release()
        raise RuntimeError("Video reports no frames")

    source_id = video_id(video)
    images_dir = dataset_dir / "images"
    annotations_dir = dataset_dir / "annotations"
    images_dir.mkdir(parents=True, exist_ok=True)
    annotations_dir.mkdir(parents=True, exist_ok=True)
    sample_count = min(count, frame_count)
    indices = sorted({round(index * (frame_count - 1) / max(1, sample_count - 1)) for index in range(sample_count)})

    manifest_path = dataset_dir / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
    else:
        manifest = {"schema_version": "1.0", "items": []}
    existing = {item["image"]: item for item in manifest["items"]}

    written = 0
    for frame_index in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = cap.read()
        if not ok:
            continue
        filename = f"{source_id}-f{frame_index:06d}.jpg"
        image_path = images_dir / filename
        cv2.imwrite(str(image_path), frame, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
        existing[filename] = {
            "image": filename,
            "source_video": source_id,
            "source_path": str(video),
            "frame_index": frame_index,
            "timestamp_s": round(frame_index / fps, 6) if fps else None,
            "width": int(frame.shape[1]),
            "height": int(frame.shape[0]),
        }
        written += 1
    cap.release()
    manifest["items"] = sorted(existing.values(), key=lambda item: (item["source_video"], item["frame_index"]))
    manifest_path.write_text(json.dumps(manifest, indent=2))
    return {"source_video": source_id, "sampled": written, "manifest_items": len(manifest["items"]), "dataset": str(dataset_dir)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Sample frames for court-keypoint annotation")
    parser.add_argument("video", type=Path)
    parser.add_argument("--dataset", type=Path, default=Path("data/court_dataset"))
    parser.add_argument("--count", type=int, default=24)
    parser.add_argument("--jpeg-quality", type=int, default=92)
    args = parser.parse_args(argv)
    if not args.video.is_file():
        parser.error(f"video not found: {args.video}")
    if args.count < 1:
        parser.error("--count must be positive")
    try:
        result = prepare(args.video, args.dataset, args.count, args.jpeg_quality)
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
