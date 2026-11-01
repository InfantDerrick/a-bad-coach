#!/usr/bin/env python3
"""Download a YouTube clip and sample frames into the court annotation dataset.

Requires ``yt-dlp`` on PATH (``pip install yt-dlp``). Prefer your own gym footage
for core training. Broadcast / third-party uploads need recorded usage rights
before you treat them as normal training data.

Example:
    python scripts/prepare_youtube_court_frames.py \\
      'https://www.youtube.com/watch?v=XXXX' \\
      --count 12 --mode random --name gym_phone_clip_12
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from prepare_court_dataset import prepare  # noqa: E402


def slugify(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9._-]+", "_", value.strip().lower()).strip("._-")
    return cleaned[:80] or "youtube_clip"


def require_yt_dlp() -> str:
    path = shutil.which("yt-dlp")
    if path:
        return path
    # Fall back to python -m yt_dlp if the module is installed but no console script.
    try:
        subprocess.run(
            [sys.executable, "-m", "yt_dlp", "--version"],
            check=True,
            capture_output=True,
            text=True,
        )
        return f"{sys.executable} -m yt_dlp"
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(
            "yt-dlp is required. Install with: python -m pip install yt-dlp"
        ) from exc


def yt_dlp_cmd(yt_dlp: str) -> list[str]:
    return yt_dlp.split() if yt_dlp.startswith(sys.executable) else [yt_dlp]


def fetch_metadata(yt_dlp: str, url: str) -> dict[str, Any]:
    completed = subprocess.run(
        [*yt_dlp_cmd(yt_dlp), "--dump-single-json", "--no-playlist", "--skip-download", url],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(completed.stdout)


def download_video(yt_dlp: str, url: str, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Prefer mp4; yt-dlp will merge if needed.
    template = str(output_path.with_suffix("")) + ".%(ext)s"
    subprocess.run(
        [
            *yt_dlp_cmd(yt_dlp),
            "--no-playlist",
            "--merge-output-format",
            "mp4",
            # Prefer <=720p for annotation frames; full 1080p match dumps are huge.
            "-f",
            "bv*[height<=720][ext=mp4]+ba[ext=m4a]/b[height<=720][ext=mp4]/bv*[height<=720]+ba/b[height<=720]/b",
            "-o",
            template,
            url,
        ],
        check=True,
    )
    matches = sorted(output_path.parent.glob(output_path.stem + ".*"))
    videos = [path for path in matches if path.suffix.lower() in {".mp4", ".mkv", ".webm", ".mov"}]
    if not videos:
        raise RuntimeError(f"Download finished but no video file found for {output_path.stem}")
    return videos[0]


def write_provenance(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("url", help="YouTube (or yt-dlp-supported) video URL")
    parser.add_argument("--name", default=None, help="Short local name for the downloaded file stem")
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--dataset", type=Path, default=Path("data/court_dataset"))
    parser.add_argument("--count", type=int, default=12)
    parser.add_argument("--mode", choices=("random", "uniform"), default="random")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--skip-start", type=float, default=0.08)
    parser.add_argument("--skip-end", type=float, default=0.08)
    parser.add_argument("--jpeg-quality", type=int, default=92)
    parser.add_argument(
        "--usage-rights",
        default="unspecified",
        help="Provenance note, e.g. 'own_upload', 'licensed', 'fair_use_review_needed'",
    )
    parser.add_argument(
        "--keep-video",
        action="store_true",
        help="Keep the downloaded video under data/raw (default: keep)",
    )
    parser.add_argument(
        "--delete-video",
        action="store_true",
        help="Delete the downloaded video after frame extraction",
    )
    args = parser.parse_args(argv)
    if args.count < 1:
        parser.error("--count must be positive")
    if args.delete_video and args.keep_video:
        parser.error("use only one of --keep-video / --delete-video")

    try:
        yt_dlp = require_yt_dlp()
        meta = fetch_metadata(yt_dlp, args.url)
    except (RuntimeError, subprocess.CalledProcessError, json.JSONDecodeError) as exc:
        print(f"YouTube metadata failed: {exc}", file=sys.stderr)
        return 1

    video_key = str(meta.get("id") or "unknown")
    title = str(meta.get("title") or "youtube_video")
    stem = slugify(args.name or f"yt_{video_key}")
    raw_video = args.raw_dir / f"{stem}.mp4"
    provenance_path = args.raw_dir / "youtube_sources" / f"{stem}.json"

    print(
        json.dumps(
            {
                "warning": (
                    "Only train on footage you have rights to use. "
                    "Set --usage-rights and keep the provenance JSON."
                ),
                "title": title,
                "id": video_key,
                "usage_rights": args.usage_rights,
            },
            indent=2,
        ),
        flush=True,
    )

    try:
        downloaded = download_video(yt_dlp, args.url, raw_video)
        if downloaded != raw_video and raw_video.suffix.lower() == downloaded.suffix.lower():
            downloaded.replace(raw_video)
            downloaded = raw_video
        elif downloaded != raw_video:
            # Keep whatever extension yt-dlp produced.
            raw_video = downloaded
        result = prepare(
            raw_video,
            args.dataset,
            args.count,
            args.jpeg_quality,
            mode=args.mode,
            seed=args.seed,
            skip_start_fraction=args.skip_start,
            skip_end_fraction=args.skip_end,
            extra_item_fields={
                "youtube_url": args.url,
                "youtube_id": video_key,
                "youtube_title": title,
                "usage_rights": args.usage_rights,
            },
        )
    except (RuntimeError, ValueError, subprocess.CalledProcessError, OSError) as exc:
        print(f"YouTube prepare failed: {exc}", file=sys.stderr)
        return 1

    provenance = {
        "downloaded_at": datetime.now(timezone.utc).isoformat(),
        "url": args.url,
        "youtube_id": video_key,
        "title": title,
        "channel": meta.get("channel") or meta.get("uploader"),
        "duration_s": meta.get("duration"),
        "webpage_url": meta.get("webpage_url") or args.url,
        "usage_rights": args.usage_rights,
        "local_video": str(raw_video),
        "dataset": str(args.dataset),
        "source_video": result["source_video"],
        "sample_mode": args.mode,
        "frame_count_requested": args.count,
        "frame_indices": result.get("frame_indices"),
        "notes": [
            "Frames are for offline court-keypoint annotation.",
            "Do not claim commercial/pro-trained rights without recorded permission.",
        ],
    }
    write_provenance(provenance_path, provenance)

    if args.delete_video and raw_video.exists():
        raw_video.unlink()
        provenance["local_video"] = None
        provenance["video_deleted_after_extract"] = True
        write_provenance(provenance_path, provenance)

    payload = {
        **result,
        "youtube_id": video_key,
        "title": title,
        "provenance": str(provenance_path),
        "local_video": None if args.delete_video else str(raw_video),
        "next_step": f"python scripts/court_annotation_server.py --dataset {args.dataset}",
    }
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
