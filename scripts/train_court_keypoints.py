#!/usr/bin/env python3
"""Train an offline court-landmark heatmap model from completed annotations.

The split is grouped by source video by default to prevent adjacent frames from
leaking into validation. Use ``--allow-frame-split`` only for plumbing tests.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pipeline.court_keypoints import LANDMARKS, validate_annotation  # noqa: E402


def choose_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def letterbox(image: np.ndarray, width: int, height: int) -> tuple[np.ndarray, float, int, int]:
    source_h, source_w = image.shape[:2]
    scale = min(width / source_w, height / source_h)
    resized_w = max(1, int(round(source_w * scale)))
    resized_h = max(1, int(round(source_h * scale)))
    resized = cv2.resize(image, (resized_w, resized_h), interpolation=cv2.INTER_AREA)
    left = (width - resized_w) // 2
    top = (height - resized_h) // 2
    output = np.full((height, width, 3), 32, dtype=np.uint8)
    output[top : top + resized_h, left : left + resized_w] = resized
    return output, scale, left, top


def gaussian_heatmap(height: int, width: int, x: float, y: float, sigma: float) -> np.ndarray:
    yy, xx = np.mgrid[0:height, 0:width]
    return np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * sigma**2)).astype(np.float32)


def load_items(dataset_dir: Path) -> list[dict[str, Any]]:
    manifest = json.loads((dataset_dir / "manifest.json").read_text())
    items: list[dict[str, Any]] = []
    for item in manifest["items"]:
        annotation_path = dataset_dir / "annotations" / f"{Path(item['image']).stem}.json"
        if not annotation_path.exists():
            continue
        annotation = json.loads(annotation_path.read_text())
        errors = validate_annotation(annotation)
        if errors:
            raise ValueError(f"Invalid annotation {annotation_path}: {'; '.join(errors)}")
        items.append({**item, "annotation_path": str(annotation_path)})
    return items


def split_items(
    items: list[dict[str, Any]],
    validation_fraction: float,
    seed: int,
    allow_frame_split: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rng = random.Random(seed)
    if allow_frame_split:
        shuffled = items.copy()
        rng.shuffle(shuffled)
        val_count = max(1, round(len(shuffled) * validation_fraction))
        return shuffled[val_count:], shuffled[:val_count]

    groups: dict[str, list[dict[str, Any]]] = {}
    for item in items:
        groups.setdefault(item["source_video"], []).append(item)
    if len(groups) < 2:
        raise ValueError(
            "Need completed annotations from at least two source videos for a grouped train/validation split. "
            "Use --allow-frame-split only to test the training plumbing."
        )
    group_names = sorted(groups)
    rng.shuffle(group_names)
    target_validation = max(1, round(len(items) * validation_fraction))
    validation: list[dict[str, Any]] = []
    training: list[dict[str, Any]] = []
    for name in group_names:
        if len(validation) < target_validation:
            validation.extend(groups[name])
        else:
            training.extend(groups[name])
    if not training:
        last_group = validation[-len(groups[group_names[-1]]) :]
        training.extend(last_group)
        del validation[-len(last_group) :]
    return training, validation


class CourtKeypointDataset(Dataset):
    def __init__(
        self,
        dataset_dir: Path,
        items: list[dict[str, Any]],
        input_width: int,
        input_height: int,
        output_stride: int,
        augment: bool,
    ) -> None:
        self.dataset_dir = dataset_dir
        self.items = items
        self.input_width = input_width
        self.input_height = input_height
        self.output_stride = output_stride
        self.augment = augment

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        item = self.items[index]
        image = cv2.imread(str(self.dataset_dir / "images" / item["image"]))
        if image is None:
            raise RuntimeError(f"Could not read image: {item['image']}")
        annotation = json.loads(Path(item["annotation_path"]).read_text())
        image, scale, pad_x, pad_y = letterbox(image, self.input_width, self.input_height)
        if self.augment:
            alpha = random.uniform(0.82, 1.18)
            beta = random.uniform(-18, 18)
            image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

        output_w = self.input_width // self.output_stride
        output_h = self.input_height // self.output_stride
        heatmaps = np.zeros((len(LANDMARKS), output_h, output_w), dtype=np.float32)
        target_points = np.full((len(LANDMARKS), 2), -1.0, dtype=np.float32)
        visibility = np.zeros(len(LANDMARKS), dtype=np.float32)
        for channel, landmark in enumerate(LANDMARKS):
            point = annotation["points"][landmark.name]
            if point is None:
                continue
            x = (float(point[0]) * scale + pad_x) / self.output_stride
            y = (float(point[1]) * scale + pad_y) / self.output_stride
            heatmaps[channel] = gaussian_heatmap(output_h, output_w, x, y, sigma=1.7)
            target_points[channel] = (x, y)
            visibility[channel] = 1.0
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        image_tensor = torch.from_numpy(image_rgb.transpose(2, 0, 1))
        return image_tensor, torch.from_numpy(heatmaps), torch.from_numpy(target_points), torch.from_numpy(visibility)


def conv_block(input_channels: int, output_channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(input_channels, output_channels, 3, padding=1, bias=False),
        nn.BatchNorm2d(output_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(output_channels, output_channels, 3, padding=1, bias=False),
        nn.BatchNorm2d(output_channels),
        nn.ReLU(inplace=True),
    )


class CourtKeypointNet(nn.Module):
    """Small U-Net-like heatmap network with output stride four."""

    def __init__(self, keypoint_count: int = len(LANDMARKS)) -> None:
        super().__init__()
        self.block1 = conv_block(3, 24)
        self.block2 = conv_block(24, 48)
        self.block3 = conv_block(48, 96)
        self.pool = nn.MaxPool2d(2)
        self.head = nn.Sequential(
            nn.Conv2d(96, 96, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(96, keypoint_count, 1),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features = self.block1(inputs)
        features = self.block2(self.pool(features))
        features = self.block3(self.pool(features))
        return self.head(features)


def weighted_heatmap_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    weights = 1.0 + targets * 40.0
    return nn.functional.binary_cross_entropy_with_logits(logits, targets, weight=weights)


@torch.no_grad()
def keypoint_error(logits: torch.Tensor, points: torch.Tensor, visibility: torch.Tensor, output_stride: int) -> tuple[float, int]:
    batch, channels, height, width = logits.shape
    flattened = torch.sigmoid(logits).reshape(batch, channels, -1)
    indices = flattened.argmax(dim=-1)
    predicted_y = torch.div(indices, width, rounding_mode="floor").float()
    predicted_x = (indices % width).float()
    predicted = torch.stack((predicted_x, predicted_y), dim=-1)
    mask = visibility.bool()
    if not mask.any():
        return 0.0, 0
    error = torch.linalg.vector_norm(predicted[mask] - points[mask], dim=-1) * output_stride
    return float(error.sum().item()), int(mask.sum().item())


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    output_stride: int,
    optimizer: torch.optim.Optimizer | None,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_error = 0.0
    visible_count = 0
    batch_count = 0
    for images, heatmaps, points, visibility in loader:
        images, heatmaps = images.to(device), heatmaps.to(device)
        points, visibility = points.to(device), visibility.to(device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            logits = model(images)
            loss = weighted_heatmap_loss(logits, heatmaps)
            if training:
                loss.backward()
                optimizer.step()
        error_sum, count = keypoint_error(logits.detach(), points, visibility, output_stride)
        total_loss += float(loss.item())
        total_error += error_sum
        visible_count += count
        batch_count += 1
    return {
        "loss": total_loss / max(1, batch_count),
        "visible_keypoint_mae_input_px": total_error / max(1, visible_count),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train the court-keypoint heatmap model")
    parser.add_argument("--dataset", type=Path, default=Path("data/court_dataset"))
    parser.add_argument("--output", type=Path, default=Path("data/models/court_keypoints.pt"))
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--input-width", type=int, default=288)
    parser.add_argument("--input-height", type=int, default=512)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--allow-frame-split", action="store_true")
    args = parser.parse_args(argv)
    if args.input_width % 4 or args.input_height % 4:
        parser.error("input dimensions must be divisible by four")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    try:
        items = load_items(args.dataset)
        if len(items) < 8:
            raise ValueError(f"Need at least 8 completed annotations; found {len(items)}")
        train_items, validation_items = split_items(items, args.validation_fraction, args.seed, args.allow_frame_split)
    except (OSError, json.JSONDecodeError, KeyError, ValueError) as exc:
        print(f"Dataset error: {exc}", file=sys.stderr)
        return 2

    output_stride = 4
    train_dataset = CourtKeypointDataset(args.dataset, train_items, args.input_width, args.input_height, output_stride, True)
    validation_dataset = CourtKeypointDataset(args.dataset, validation_items, args.input_width, args.input_height, output_stride, False)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    validation_loader = DataLoader(validation_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    device = choose_device(args.device)
    model = CourtKeypointNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    best_error = math.inf
    history: list[dict[str, Any]] = []
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(model, train_loader, device, output_stride, optimizer)
        validation_metrics = run_epoch(model, validation_loader, device, output_stride, None)
        row = {"epoch": epoch, "train": train_metrics, "validation": validation_metrics}
        history.append(row)
        print(json.dumps(row))
        if validation_metrics["visible_keypoint_mae_input_px"] < best_error:
            best_error = validation_metrics["visible_keypoint_mae_input_px"]
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "landmarks": [landmark.name for landmark in LANDMARKS],
                    "input_width": args.input_width,
                    "input_height": args.input_height,
                    "output_stride": output_stride,
                    "validation_mae_input_px": best_error,
                    "train_source_videos": sorted({item["source_video"] for item in train_items}),
                    "validation_source_videos": sorted({item["source_video"] for item in validation_items}),
                },
                args.output,
            )
    history_path = args.output.with_suffix(".history.json")
    history_path.write_text(json.dumps(history, indent=2))
    print(json.dumps({"checkpoint": str(args.output), "best_validation_mae_input_px": best_error, "device": str(device)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
