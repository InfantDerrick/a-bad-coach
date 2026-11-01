"""Controlled real/synthetic training diagnostic, not unseen-real validation."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np
import torch

from train_court_keypoints import (
    CourtKeypointDataset, CourtKeypointNet, LANDMARKS, choose_device,
    load_items, split_items, weighted_heatmap_loss,
)
from evaluate_court_keypoints import evaluate


@torch.no_grad()
def measure(model, tensors, items, dataset_dir):
    images, _, targets, visible = tensors
    model.eval()
    rows = []
    total_error, count = 0., 0
    for start in range(0, len(items), 16):
        probs = model(images[start:start + 16]).sigmoid()
        batch, channels, height, width = probs.shape
        flat = probs.reshape(batch, channels, -1)
        confidence, indices = flat.max(dim=-1)
        xy = torch.stack((indices % width, indices // width), dim=-1).float()
        errors = torch.linalg.vector_norm(xy - targets[start:start + batch], dim=-1) * 4
        mask = visible[start:start + batch].bool()
        total_error += errors[mask].sum().item()
        count += mask.sum().item()
        for index in range(batch):
            item = items[start + index]
            ann = json.loads(Path(item['annotation_path']).read_text())
            scale = min(288 / ann['image_width'], 512 / ann['image_height'])
            padding = np.array([(288 - round(ann['image_width'] * scale)) // 2,
                                (512 - round(ann['image_height'] * scale)) // 2])
            positions = (xy[index].cpu().numpy() * 4 - padding) / scale
            points = {landmark.name: point.tolist() if confidence[index, channel] >= .55
                      and 0 <= point[0] < ann['image_width'] and 0 <= point[1] < ann['image_height'] else None
                      for channel, (landmark, point) in enumerate(zip(LANDMARKS, positions))}
            rows.append({'annotation': Path(item['annotation_path']).name, 'points': points, **evaluate(ann, points)})
    matched = sum(row['matched_visible'] for row in rows)
    return {
        'raw_mae_input_px': total_error / count,
        'visible_coverage': matched / sum(row['annotated_visible'] for row in rows),
        'confident_mae_source_px': sum(sum(row['per_landmark_error_px'].values()) for row in rows) / matched if matched else None,
        'abstention_rate': sum(row['status'] == 'abstained' for row in rows) / len(rows),
        'predicted_on_invisible_count': sum(len(row['predicted_on_invisible']) for row in rows),
        'frames': rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, default=Path('data/court_dataset'))
    parser.add_argument('--steps', type=int, default=1000)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--output-dir', type=Path, default=Path('data/processed/training_comparison'))
    parser.add_argument('--figure', type=Path)
    args = parser.parse_args()
    if args.steps <= 0:
        parser.error('steps must be positive')
    device = choose_device(args.device)
    print(json.dumps({'device': str(device), 'gpu': torch.cuda.get_device_name(device) if device.type == 'cuda' else None}), flush=True)
    items = load_items(args.dataset)
    training, validation = split_items(items, .2, 7, False)
    real = [item for item in training if not item['image'].startswith('synthetic-')]
    synthetic = [item for item in training if item['image'].startswith('synthetic-')]
    assert len(real) == 12 and len(synthetic) > 0
    ordered = real + synthetic
    def cache(selected):
        dataset = CourtKeypointDataset(args.dataset, selected, 288, 512, 4, False)
        return tuple(torch.stack([dataset[i][column] for i in range(len(dataset))]).to(device) for column in range(4))
    # Materialize once; no random augmentation or I/O differences between arms.
    dataset = CourtKeypointDataset(args.dataset, ordered, 288, 512, 4, False)
    samples = [dataset[i] for i in range(len(dataset))]
    tensors = tuple(torch.stack([sample[column] for sample in samples]).to(device) for column in range(4))
    real_tensors = tuple(tensor[:len(real)] for tensor in tensors)
    validation_tensors = cache(validation)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result = {'scope': 'All real evaluation frames are training examples from ONE video; this is not generalization evidence.',
              'seed': 7, 'steps_per_arm': args.steps, 'batch_size': 16, 'augmentation': False,
              'real_frames': len(real), 'synthetic_training_frames': len(synthetic),
              'training_sources': sorted({item['source_video'] for item in ordered}),
              'synthetic_validation_sources': sorted({item['source_video'] for item in validation}), 'arms': {}}
    for arm in ('real_only', 'natural_mix', 'balanced_mix'):
        random.seed(7); np.random.seed(7); torch.manual_seed(7)
        rng = np.random.default_rng(7)
        model = CourtKeypointNet().to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
        history = []
        real_draws = 0
        for step in range(args.steps + 1):
            if step:
                if arm == 'real_only':
                    indices = rng.integers(0, len(real), 16)
                elif arm == 'natural_mix':
                    indices = rng.integers(0, len(ordered), 16)
                else:
                    indices = np.concatenate((rng.integers(0, len(real), 8), rng.integers(len(real), len(ordered), 8)))
                real_draws += int((indices < len(real)).sum())
                index_tensor = torch.as_tensor(indices, device=device)
                model.train()
                optimizer.zero_grad(set_to_none=True)
                loss = weighted_heatmap_loss(model(tensors[0][index_tensor]), tensors[1][index_tensor])
                loss.backward(); optimizer.step()
            if step % 200 == 0 or step == args.steps:
                real_result = measure(model, real_tensors, real, args.dataset)
                row = {'step': step, 'raw_mae_input_px': real_result['raw_mae_input_px'],
                       'visible_coverage': real_result['visible_coverage'], 'abstention_rate': real_result['abstention_rate']}
                history.append(row)
                print(json.dumps({'arm': arm, **row}), flush=True)
        final = measure(model, real_tensors, real, args.dataset)
        synthetic_result = measure(model, validation_tensors, validation, args.dataset)
        result['arms'][arm] = {'real_examples_sampled': real_draws, 'history': history,
                               'real_training_evaluation': final,
                               'held_out_synthetic_raw_mae_input_px': synthetic_result['raw_mae_input_px']}
        torch.save({'model_state': model.state_dict(), 'landmarks': [x.name for x in LANDMARKS],
                    'input_width': 288, 'input_height': 512, 'output_stride': 4,
                    'diagnostic_only': True, 'arm': arm, 'steps': args.steps}, args.output_dir / f'{arm}.pt')
        (args.output_dir / 'comparison.json').write_text(json.dumps(result, indent=2))
    if args.figure:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(15, 7))
        for ax, (name, arm) in zip(axes, result['arms'].items()):
            row = arm['real_training_evaluation']['frames'][0]
            ann = json.loads(Path(real[0]['annotation_path']).read_text())
            frame = cv2.cvtColor(cv2.imread(str(args.dataset / 'images' / ann['image'])), cv2.COLOR_BGR2RGB)
            ax.imshow(frame); ax.set_xlim(180, 2050); ax.set_ylim(2550, 1500); ax.axis('off')
            for landmark, target in ann['points'].items():
                if target is not None:
                    ax.scatter(*target, s=70, facecolors='none', edgecolors='lime', linewidths=2)
                predicted = row['points'].get(landmark)
                if predicted is not None:
                    ax.scatter(*predicted, s=70, marker='x', c='red')
            metrics = arm['real_training_evaluation']
            ax.set_title(name.replace('_', ' ') + f"\n12-frame mean: {metrics['raw_mae_input_px']:.2f} input px\nCoverage: {metrics['visible_coverage']:.0%}")
        fig.suptitle(f'{args.steps} updates each | green: labels | red: confident predictions | frame 0', fontsize=15)
        fig.text(.05, .07, 'Training-fit diagnostic: all 12 real frames were used for learning. This does not measure accuracy on new videos.', fontsize=12)
        fig.tight_layout(rect=(0, .1, 1, .92))
        args.figure.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.figure, dpi=150)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
