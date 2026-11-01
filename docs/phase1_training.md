Phase 1 — Detection training guide

Goal: achieve robust player detection and reliable target assignment (which player to track).

1) Data collection
- Collect 200–1000 images from your videos: sample frames across sessions, cameras, lighting, and player positions.
- Aim for diversity: closeups, far court, occlusions, different jerseys/backgrounds.

2) Labeling
- Use LabelImg, Label Studio, or Roboflow to draw bounding boxes around players.
- Save labels in YOLO format (one .txt per image: class x_center y_center width height normalized).
- Ensure class 0 = `player`.

3) Dataset layout (recommended)
data/phase1_dataset/
  images/
    train/
    val/
  labels/
    train/
    val/

Split: 80/20 train/val by frames, keep videos grouped if possible.

4) Prepare YOLOv8 config
- Use `scripts/prepare_yolo_config.py` to generate a `data.yaml` pointing at the `images/` folder.

5) Training (quick start)
```bash
python3 -m pip install -r requirements.txt
# optionally install 'nvidia-cuda-toolkit' or use CPU (slow)
# from repo root:
python3 -c "from ultralytics import YOLO; print('ok')"
# train (example):
python3 -m ultralytics.yolo.train data=data/processed/phase1/yolo_data.yaml model=yolov8n.pt epochs=50 imgsz=640
```

6) Fine-tuning tips
- Start with `yolov8n.pt` (nano) then scale to `yolov8s`/`m` if needed.
- Use `batch` sized for your GPU memory.
- Use simple augmentations: horizontal flip, small scale/brightness.

7) Evaluation
- Use the built-in YOLO metrics (mAP@0.5, precision, recall).
- Inspect failure cases and add more labeled images for those scenarios.

8) Re-ID / Target selection
- If automatic selection between two players is often wrong, collect short tracklets of the target player and others.
- Create a small dataset where crops are labeled `target` vs `other` (or player IDs) and train a small classifier (ResNet18) for re-id.
- Alternatively, use embedding + k-NN on embeddings to match identities.

9) If you want, I can scaffold the training scripts and a small notebook that:
- samples frames, generates labeling buckets
- converts COCO ↔ YOLO
- runs an example YOLOv8 training run (dry-run)


Good resources:
- YOLOv8 docs: https://docs.ultralytics.com/
- Roboflow/LabelImg for labeling

