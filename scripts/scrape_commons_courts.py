#!/usr/bin/env python3
"""Download attribution-preserving badminton court photos from Wikimedia Commons.

Only freely licensed Commons files are considered. Search results are filtered
for raster photos, minimum resolution, duplicate hashes, and visible long-line
evidence. Every image retains its description-page URL, creator, and license.
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
import ssl
import sys
import time
from html.parser import HTMLParser
from pathlib import Path
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import cv2
import certifi
import numpy as np


API_URL = "https://commons.wikimedia.org/w/api.php"
USER_AGENT = "BadmintonCoachDatasetBuilder/0.1 (offline research dataset; contact: local-user)"
DEFAULT_QUERIES = (
    '"badminton court"',
    'badminton match court',
    'badminton tournament court',
    'badminton players indoor court',
)
EXCLUDED_TITLE_TERMS = (
    "diagram", "dimensions", "layout", "logo", "icon", "map", "3d", "render", "schema", "plan view",
)
ALLOWED_LICENSE_MARKERS = ("cc0", "public domain", "cc by", "cc-by", "cc by-sa", "cc-by-sa")
TLS_CONTEXT = ssl.create_default_context(cafile=certifi.where())


class TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []

    def handle_data(self, data: str) -> None:
        self.parts.append(data)


def strip_html(value: str | None) -> str:
    if not value:
        return ""
    parser = TextExtractor()
    parser.feed(html.unescape(value))
    return re.sub(r"\s+", " ", " ".join(parser.parts)).strip()


def api_json(parameters: dict[str, Any]) -> dict[str, Any]:
    query = urlencode({"format": "json", "formatversion": 2, **parameters})
    request = Request(f"{API_URL}?{query}", headers={"User-Agent": USER_AGENT})
    with urlopen(request, timeout=30, context=TLS_CONTEXT) as response:
        return json.load(response)


def search_candidates(query: str, limit: int) -> list[dict[str, Any]]:
    payload = api_json(
        {
            "action": "query",
            "generator": "search",
            "gsrsearch": query,
            "gsrnamespace": 6,
            "gsrlimit": min(50, limit),
            "prop": "imageinfo",
            "iiprop": "url|size|mime|sha1|extmetadata",
            "iiurlwidth": 1600,
            "iiextmetadatalanguage": "en",
            "iiextmetadatafilter": "LicenseShortName|LicenseUrl|Artist|Credit|ImageDescription|UsageTerms",
        }
    )
    return payload.get("query", {}).get("pages", [])


def metadata_value(metadata: dict[str, Any], key: str) -> str:
    entry = metadata.get(key, {})
    return strip_html(entry.get("value") if isinstance(entry, dict) else str(entry))


def line_score(image: np.ndarray) -> int:
    height, width = image.shape[:2]
    scale = min(1.0, 900 / max(height, width))
    small = cv2.resize(image, (round(width * scale), round(height * scale)), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 55, 150)
    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 180,
        threshold=45,
        minLineLength=max(45, int(min(small.shape[:2]) * 0.16)),
        maxLineGap=18,
    )
    return 0 if lines is None else int(min(100, len(lines)))


def difference_hash(image: np.ndarray) -> int:
    gray = cv2.cvtColor(cv2.resize(image, (9, 8), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
    bits = gray[:, 1:] > gray[:, :-1]
    value = 0
    for bit in bits.flatten():
        value = (value << 1) | int(bit)
    return value


def hamming_distance(left: int, right: int) -> int:
    return (left ^ right).bit_count()


def download_bytes(url: str) -> bytes:
    request = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(request, timeout=45, context=TLS_CONTEXT) as response:
        return response.read()


def scrape(dataset_dir: Path, max_images: int, per_query: int, min_width: int, min_height: int) -> dict[str, Any]:
    images_dir = dataset_dir / "images"
    annotations_dir = dataset_dir / "annotations"
    images_dir.mkdir(parents=True, exist_ok=True)
    annotations_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = dataset_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {"schema_version": "1.0", "items": []}
    existing_items = {item["image"]: item for item in manifest["items"]}
    sources_path = dataset_dir / "sources.json"
    sources = json.loads(sources_path.read_text()) if sources_path.exists() else {"schema_version": "1.0", "sources": {}}
    known_sha1 = {source.get("commons_sha1") for source in sources["sources"].values() if source.get("commons_sha1")}
    known_hashes: list[int] = []
    for filename, source in sources["sources"].items():
        if source.get("difference_hash"):
            known_hashes.append(int(source["difference_hash"], 16))

    candidates: dict[str, dict[str, Any]] = {}
    for query in DEFAULT_QUERIES:
        for page in search_candidates(query, per_query):
            info_list = page.get("imageinfo") or []
            if not info_list:
                continue
            info = info_list[0]
            candidates.setdefault(page["title"], {"page": page, "info": info, "query": query})
        time.sleep(0.2)

    accepted = 0
    rejected: dict[str, int] = {}

    def reject(reason: str) -> None:
        rejected[reason] = rejected.get(reason, 0) + 1

    for title in sorted(candidates):
        if accepted >= max_images:
            break
        page = candidates[title]["page"]
        info = candidates[title]["info"]
        title_lower = title.lower()
        if any(term in title_lower for term in EXCLUDED_TITLE_TERMS):
            reject("title")
            continue
        mime = str(info.get("mime", ""))
        if mime not in {"image/jpeg", "image/png", "image/webp"}:
            reject("mime")
            continue
        if int(info.get("width", 0)) < min_width or int(info.get("height", 0)) < min_height:
            reject("resolution")
            continue
        sha1 = info.get("sha1")
        if sha1 in known_sha1:
            reject("duplicate_sha1")
            continue
        metadata = info.get("extmetadata", {})
        license_name = metadata_value(metadata, "LicenseShortName") or metadata_value(metadata, "UsageTerms")
        if not any(marker in license_name.lower() for marker in ALLOWED_LICENSE_MARKERS):
            reject("license")
            continue
        image_url = info.get("thumburl") or info.get("url")
        if not isinstance(image_url, str) or not image_url.startswith("https://"):
            reject("url")
            continue
        try:
            raw = download_bytes(image_url)
            array = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
        except (OSError, ValueError):
            reject("download")
            continue
        if array is None:
            reject("decode")
            continue
        height, width = array.shape[:2]
        if width < min_width or height < min_height:
            reject("thumbnail_resolution")
            continue
        score = line_score(array)
        if score < 3:
            reject("line_score")
            continue
        perceptual_hash = difference_hash(array)
        if any(hamming_distance(perceptual_hash, previous) <= 4 for previous in known_hashes):
            reject("near_duplicate")
            continue
        token = hashlib.sha256(title.encode("utf-8")).hexdigest()[:14]
        filename = f"commons-{token}.jpg"
        cv2.imwrite(str(images_dir / filename), array, [cv2.IMWRITE_JPEG_QUALITY, 92])
        description_url = info.get("descriptionurl") or f"https://commons.wikimedia.org/wiki/{title.replace(' ', '_')}"
        source_id = f"commons-{token}"
        item = {
            "image": filename,
            "source_video": source_id,
            "source_path": description_url,
            "frame_index": 0,
            "timestamp_s": None,
            "width": width,
            "height": height,
        }
        existing_items[filename] = item
        sources["sources"][filename] = {
            "provider": "Wikimedia Commons",
            "title": title,
            "description_page": description_url,
            "download_url": image_url,
            "creator": metadata_value(metadata, "Artist") or metadata_value(metadata, "Credit"),
            "description": metadata_value(metadata, "ImageDescription"),
            "license": license_name,
            "license_url": metadata_value(metadata, "LicenseUrl"),
            "commons_sha1": sha1,
            "difference_hash": f"{perceptual_hash:016x}",
            "court_line_score": score,
            "search_query": candidates[title]["query"],
        }
        known_sha1.add(sha1)
        known_hashes.append(perceptual_hash)
        accepted += 1

    manifest["items"] = sorted(existing_items.values(), key=lambda item: (item["source_video"], item["frame_index"]))
    manifest_path.write_text(json.dumps(manifest, indent=2))
    sources_path.write_text(json.dumps(sources, indent=2))
    return {
        "accepted": accepted,
        "candidates": len(candidates),
        "rejected": rejected,
        "manifest_items": len(manifest["items"]),
        "attribution_manifest": str(sources_path),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Download reusable badminton-court photos from Wikimedia Commons")
    parser.add_argument("--dataset", type=Path, default=Path("data/court_dataset"))
    parser.add_argument("--max-images", type=int, default=30)
    parser.add_argument("--per-query", type=int, default=40)
    parser.add_argument("--min-width", type=int, default=800)
    parser.add_argument("--min-height", type=int, default=600)
    args = parser.parse_args(argv)
    try:
        result = scrape(args.dataset, args.max_images, args.per_query, args.min_width, args.min_height)
    except (OSError, json.JSONDecodeError, KeyError, ValueError) as exc:
        print(f"Commons scrape failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
