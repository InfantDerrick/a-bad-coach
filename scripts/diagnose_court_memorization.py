"""Memorize one annotated image; this is a training diagnostic, never validation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from train_court_keypoints import (
    CourtKeypointDataset, CourtKeypointNet, LANDMARKS, choose_device,
    letterbox, weighted_heatmap_loss,
)
from infer_court_keypoints import predict


def peaks(logits: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    probabilities = logits.sigmoid()[0].detach().cpu().numpy()
    width = probabilities.shape[-1]
    indices = probabilities.reshape(len(LANDMARKS), -1).argmax(axis=1)
    return np.stack((indices % width, indices // width), axis=1), probabilities


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, default=Path('data/court_dataset'))
    parser.add_argument('--annotation', default='bad_1-3507bbab2c-f000000.json')
    parser.add_argument('--steps', type=int, default=1000)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--output-dir', type=Path, default=Path('data/processed/one_frame_diagnostic'))
    parser.add_argument('--figure', type=Path)
    args = parser.parse_args()
    if args.steps < 1:
        parser.error('steps must be positive')
    torch.manual_seed(7)
    np.random.seed(7)
    device = choose_device(args.device)
    print(json.dumps({'device': str(device), 'gpu': torch.cuda.get_device_name(device) if device.type == 'cuda' else None}), flush=True)
    annotation_path = args.dataset / 'annotations' / args.annotation
    annotation = json.loads(annotation_path.read_text())
    dataset = CourtKeypointDataset(args.dataset, [{'image': annotation['image'], 'annotation_path': str(annotation_path)}], 288, 512, 4, False)
    image, target, points, visibility = dataset[0]
    image, target = image[None].to(device), target[None].to(device)
    visible = visibility.numpy().astype(bool)
    expected = points.numpy()
    # Verify actual tensor target coordinates against independently transformed labels.
    frame = cv2.imread(str(args.dataset / 'images' / annotation['image']))
    prepared, scale, pad_x, pad_y = letterbox(frame, 288, 512)
    independently_transformed = np.array([
        [(annotation['points'][landmark.name][0] * scale + pad_x) / 4,
         (annotation['points'][landmark.name][1] * scale + pad_y) / 4]
        for landmark in LANDMARKS if annotation['points'][landmark.name] is not None])
    np.testing.assert_allclose(expected[visible], independently_transformed, atol=1e-5)
    target_peaks, _ = peaks(torch.logit(target.clamp(1e-6, 1 - 1e-6)))
    assert np.max(np.abs(target_peaks[visible] - expected[visible])) <= .5
    model = CourtKeypointNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.output_dir / 'memorization_only.pt'
    history = []
    best = float('inf')
    best_step = 0
    for step in range(1, args.steps + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = model(image)
        loss = weighted_heatmap_loss(logits, target)
        loss.backward()
        optimizer.step()
        if step == 1 or step % 50 == 0 or step == args.steps:
            model.eval()
            with torch.no_grad():
                coordinates, probabilities = peaks(model(image))
            errors = np.linalg.norm(coordinates[visible] - expected[visible], axis=1) * 4
            row = {'step': step, 'loss': float(loss.detach()), 'mae_input_px': float(errors.mean()),
                   'max_error_input_px': float(errors.max()),
                   'visible_above_055': int((probabilities.max(axis=(1, 2))[visible] >= .55).sum())}
            history.append(row)
            print(json.dumps(row), flush=True)
            if errors.mean() < best:
                best = float(errors.mean())
                best_step = step
                torch.save({'model_state': model.state_dict(), 'landmarks': [x.name for x in LANDMARKS],
                            'input_width': 288, 'input_height': 512, 'output_stride': 4,
                            'diagnostic_only': True, 'memorized_annotation': args.annotation}, checkpoint_path)
    state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state['model_state'])
    model.eval()
    with torch.no_grad():
        coordinates, probabilities = peaks(model(image))
    deployed, confidence, _ = predict(frame, checkpoint_path, .55, device)
    rows = []
    for index, landmark in enumerate(LANDMARKS):
        if not visible[index]:
            continue
        location = coordinates[index] * 4
        rows.append({'landmark': landmark.name, 'target_input': (expected[index] * 4).tolist(),
                     'prediction_input': location.tolist(),
                     'error_input_px': float(np.linalg.norm(location - expected[index] * 4)),
                     'confidence': confidence[landmark.name], 'deployed_source_point': deployed[landmark.name],
                     'deployed_error_source_px': float(np.linalg.norm(np.asarray(deployed[landmark.name]) - annotation['points'][landmark.name])) if deployed[landmark.name] is not None else None})
    result = {'scope': 'ONE TRAINING IMAGE ONLY; NOT A GENERALIZATION TEST', 'steps': args.steps,
              'augmentation': False, 'initialized_from': 'random seed 7', 'best_step': best_step,
              'best_mae_input_px': best, 'label_transform_check': 'passed',
              'nearest_grid_mae_input_px': float((np.linalg.norm(expected[visible] - np.round(expected[visible]), axis=1) * 4).mean()),
              'inference_offset_input_px': [0, 0], 'landmarks': rows, 'history': history}
    (args.output_dir / 'diagnostic.json').write_text(json.dumps(result, indent=2))
    if args.figure:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(14, 8))
        for ax in axes[:2]:
            ax.imshow(cv2.cvtColor(prepared, cv2.COLOR_BGR2RGB))
            ax.set_xlim(15, 275); ax.set_ylim(365, 205)
        axes[0].set_title('Exact resized training image + labels')
        axes[1].set_title(f'Same image after {best_step} training steps')
        for number, row in enumerate(rows, 1):
            x, y = row['target_input']; px, py = row['prediction_input']
            for ax in axes[:2]:
                ax.scatter(x, y, s=85, facecolors='none', edgecolors='lime', linewidths=2)
            axes[0].annotate(str(number), (x, y), xytext=(4, -9 if number == 1 else 4), textcoords='offset points', color='white', fontsize=10, bbox=dict(facecolor='black', alpha=.6, pad=1))
            axes[1].scatter(px, py, marker='x', c='red', s=65)
        axes[2].plot([r['step'] for r in history], [r['mae_input_px'] for r in history])
        axes[2].axhline(result['nearest_grid_mae_input_px'], color='gray', linestyle='--', label='nearest grid point')
        axes[2].set(xlabel='Repeated training steps on this photo', ylabel='Mean landmark error (input pixels)', yscale='log', title='Learning this one example')
        axes[2].legend()
        fig.suptitle('One-photo memorization diagnostic — green = label, red = predicted heatmap peak', fontsize=14)
        fig.text(.05, .04, f'Transformed labels checked. Best error: {best:.2f} input pixels. This photo was used for training; no claim about other photos.', fontsize=11)
        fig.tight_layout(rect=(0, .08, 1, .94))
        args.figure.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.figure, dpi=150)
    print(json.dumps({k: v for k, v in result.items() if k not in ('history', 'landmarks')}), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
