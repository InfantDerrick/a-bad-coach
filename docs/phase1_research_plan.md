Phase 1 Research Plan — Full Court Detection

Goal: reliable full-court detection (baselines, service lines, sidelines, back boundaries, net) across cameras and environments.

1) Data
- Collect a dataset of ~500–2000 annotated frames across videos, camera positions, and lighting.
- Annotations: full court polygon/corners (4 pts), plus line masks (binary mask of court lines) and net line/post markers.
- Split: 80% train, 10% val, 10% test.

2) Baseline heuristics (fast):
- Player-anchor + Hough + color mask + angle clustering (already implemented prototype).
- Multi-frame RANSAC + vanishing point estimation to stabilize corners.

3) Trainable models (robust):
- Court-line segmentation (U-Net/DeepLabv3) predicting a binary mask for court lines.
- Corner regression head: predict 4 corner points as a regression output from backbone features.
- Net detection head: detect the net line for anchor.

4) Hybrid pipeline:
- Run segmentation to get clean line masks.
- Fit lines via Hough on mask to extract robust lines.
- RANSAC cluster intersections across frames for final corners.
- Temporal smoothing (Kalman) to produce stable homography for video.

5) Metrics
- Pixel RMSE for corners.
- IoU for line segmentation.
- In/out classification accuracy for a set of test ball/shuttle landing points.

6) Tools & annotation
- Use Label Studio or CVAT for masks and corner points.
- Scripts provided to sample frames, convert corners->mask, and evaluate.

7) Compute & training
- Start with small U-Net on 256x256 crops, then scale to 512.
- Use AdamW, Cosine LR, early stopping on val IoU.

8) Timeline
- Data collection + tool setup: 3–7 days
- Baseline heuristics + aggregation: 2–3 days
- Segmentation training & eval: 3–5 days
- Integration + smoothing: 2–3 days


Contact me which subtask to begin: annotation tooling, segmentation model scaffold, or net detection.