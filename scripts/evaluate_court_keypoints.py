"""Compare model landmarks with labeled images, separately from fit residuals."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from infer_court_keypoints import predict
from train_court_keypoints import choose_device
from pipeline.court_keypoints import fit_homography_from_landmarks, validate_annotation


def evaluate(annotation: dict, points: dict) -> dict:
    """Missing predictions reduce coverage; they are never counted as zero error."""
    visible = {name: point for name, point in annotation['points'].items() if point is not None}
    errors = {name: float(np.linalg.norm(np.asarray(point) - points[name]))
              for name, point in visible.items() if points.get(name) is not None}
    mean_error = float(np.mean(list(errors.values()))) if errors else None
    result = {
        'annotated_visible': len(visible), 'matched_visible': len(errors),
        'visible_coverage': len(errors) / len(visible) if visible else None,
        'missed_visible': [name for name in visible if points.get(name) is None],
        'predicted_on_invisible': [name for name, point in points.items() if point is not None and name not in visible],
        'mae_source_px': mean_error,
        'normalized_mae_diagonal': mean_error / np.hypot(annotation['image_width'], annotation['image_height']) if errors else None,
        'per_landmark_error_px': errors,
    }
    try:
        _, result['fit'] = fit_homography_from_landmarks(points)
        result['status'] = 'model_prediction_unverified'
    except ValueError as exc:
        result.update(status='abstained', reason=str(exc))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, default=Path('data/court_dataset'))
    parser.add_argument('--annotation-glob', default='bad_1*.json')
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--confidence', type=float, default=.55)
    parser.add_argument('--device', default='auto')
    args = parser.parse_args()
    paths = sorted((args.dataset / 'annotations').glob(args.annotation_glob))
    if not paths:
        parser.error('No matching annotations')
    device = choose_device(args.device)
    rows = []
    for path in paths:
        annotation = json.loads(path.read_text())
        errors = validate_annotation(annotation)
        if errors:
            parser.error(f'{path}: {errors}')
        frame = cv2.imread(str(args.dataset / 'images' / annotation['image']))
        if frame is None or frame.shape[:2] != (annotation['image_height'], annotation['image_width']):
            parser.error(f'{path}: image missing or dimensions disagree with annotation')
        points, _, _ = predict(frame, args.checkpoint, args.confidence, device)
        rows.append({'annotation': path.name, **evaluate(annotation, points)})
    result = {
        'checkpoint': str(args.checkpoint), 'confidence_threshold': args.confidence,
        'scope': 'Annotation comparison only; verify source-video split before claiming generalization.',
        'frames': len(rows),
        'abstention_rate': sum(row['status'] == 'abstained' for row in rows) / len(rows),
        'results': rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
