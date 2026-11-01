"""
Generate a YOLOv8 `data.yaml` config for training given dataset paths.
Usage:
  python scripts/prepare_yolo_config.py --images_dir data/processed/phase1/images --labels_dir data/processed/phase1/labels --out data/processed/phase1/yolo_data.yaml
"""
from __future__ import annotations
import argparse
from pathlib import Path
import yaml


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images_dir", required=True)
    parser.add_argument("--labels_dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--nc", type=int, default=1, help="number of classes (person=1)")
    parser.add_argument("--names", nargs='+', default=["player"], help="class names")
    args = parser.parse_args()

    images = Path(args.images_dir)
    labels = Path(args.labels_dir)
    out = Path(args.out)

    # expect structure: images/train images/val labels/train labels/val
    d = {
        'path': str(images.resolve()),
        'train': 'train',
        'val': 'val',
        'nc': args.nc,
        'names': {i: n for i, n in enumerate(args.names)}
    }

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(yaml.safe_dump(d))
    print(f"Wrote YOLO config to {out}")


if __name__ == '__main__':
    main()
