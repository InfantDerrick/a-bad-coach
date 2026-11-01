#!/usr/bin/env python3
"""
Evaluate estimated court corners / overlay against ground-truth corners.

GT format: JSON with keys as image filenames mapping to list of 4 [x,y] corners in tl,tr,br,bl order.

Outputs per-image pixel RMSE and mean-meter error if `scale_m_per_pixel` provided.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import math

import numpy as np


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--gt', type=Path, required=True, help='GT JSON mapping image->corners')
    parser.add_argument('--pred', type=Path, required=True, help='Pred JSON mapping image->corners')
    parser.add_argument('--scale', type=float, default=None, help='meters per pixel scale to convert to meters')
    args = parser.parse_args(argv)

    gt = json.loads(args.gt.read_text())
    pred = json.loads(args.pred.read_text())
    rows = []
    for k, v in gt.items():
        if k not in pred:
            continue
        gt_pts = np.array(v, dtype=float)
        pr_pts = np.array(pred[k], dtype=float)
        if gt_pts.shape != pr_pts.shape:
            continue
        px_rmse = rmse(gt_pts.reshape(-1, 2), pr_pts.reshape(-1, 2))
        meters = px_rmse * args.scale if args.scale else None
        rows.append({'image': k, 'px_rmse': px_rmse, 'meters': meters})

    # summary
    pxs = [r['px_rmse'] for r in rows]
    if not pxs:
        print('No overlap between GT and predictions')
        return 2
    print(f'Images evaluated: {len(rows)}')
    print(f'Pixel RMSE mean: {np.mean(pxs):.2f}, median: {np.median(pxs):.2f}, max: {np.max(pxs):.2f}')
    if args.scale:
        meters = [r['meters'] for r in rows]
        print(f'Mean error in meters: {np.mean(meters):.3f} m')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
