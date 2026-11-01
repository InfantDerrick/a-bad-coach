#!/usr/bin/env python3
"""Summarize court dataset diversity and annotation coverage."""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("data/court_dataset"))
    parser.add_argument("--held-out", type=Path, default=Path("data/held_out_sources.json"))
    args = parser.parse_args()
    manifest = json.loads((args.dataset / "manifest.json").read_text())
    annotations = {path.stem for path in (args.dataset / "annotations").glob("*.json")}
    by_source: dict[str, dict[str, int]] = defaultdict(lambda: {"frames": 0, "annotated": 0})
    for item in manifest["items"]:
        source = item["source_video"]
        by_source[source]["frames"] += 1
        if Path(item["image"]).stem in annotations:
            by_source[source]["annotated"] += 1
    real = [name for name in by_source if not name.startswith("synthetic")]
    synthetic = [name for name in by_source if name.startswith("synthetic")]
    held = json.loads(args.held_out.read_text()) if args.held_out.exists() else {}
    report = {
        "dataset": str(args.dataset),
        "manifest_items": len(manifest["items"]),
        "annotated_items": sum(1 for item in manifest["items"] if Path(item["image"]).stem in annotations),
        "real_sources": len(real),
        "synthetic_sources": len(synthetic),
        "real_source_target_tier2": "25-40",
        "held_out": held,
        "sources": {
            name: by_source[name]
            for name in sorted(by_source, key=lambda key: (key.startswith("synthetic"), key))
        },
    }
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
