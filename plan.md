# Badminton AI Coach — Master Plan

> **Goal:** Build an application where you upload badminton match footage, select a player (court side), and receive actionable coaching feedback — form critique, shot suggestions, movement analysis.

---

## Architecture Overview

```
┌──────────────────────────────────────────────────────────┐
│                      Frontend (React)                    │
│  Upload video → Select court side → View feedback report │
└──────────────────────┬───────────────────────────────────┘
                       │ HTTP / WebSocket
┌──────────────────────▼───────────────────────────────────┐
│                   Backend API (FastAPI)                   │
│  Job queue → Processing pipeline → Results storage       │
└──────────────────────┬───────────────────────────────────┘
                       │
┌──────────────────────▼───────────────────────────────────┐
│                  ML Pipeline (Python)                     │
│                                                          │
│  1. Frame Extraction (OpenCV)                            │
│  2. Court Detection & Homography                         │
│  3. Player Detection & Tracking                          │
│  4. Pose Estimation (MediaPipe / MoveNet)                │
│  5. Shuttle Tracking                                     │
│  6. Feature Extraction (angles, velocity, positions)     │
│  7. Shot Classification (LSTM / Temporal CNN)            │
│  8. Form Analysis (rule-based + ML scoring)              │
│  9. Tactical Analysis (shot selection evaluation)        │
│  10. Feedback Generation                                 │
└──────────────────────────────────────────────────────────┘
```

---

## Phase 0 — Environment & Foundations

*Get the workspace ready and learn the absolute basics. No ML yet — just tooling.*
### Checklist

- [ ] **0.1** Set up Python environment (pyenv / conda, Python 3.11+)
- [ ] **0.2** Create project structure:
  ```
  badminton-coach/
  ├── plan.md
  ├── notebooks/          # Jupyter experiments
  ├── src/
  │   ├── pipeline/       # ML pipeline modules
  │   ├── api/            # FastAPI backend
  │   └── frontend/       # React app
  ├── data/
  │   ├── raw/            # Raw videos
  │   ├── processed/      # Extracted frames, labels
  │   └── models/         # Trained model weights
  ├── tests/
  ├── requirements.txt
  └── README.md
  ```
- [ ] **0.3** Install core dependencies: `opencv-python`, `mediapipe`, `torch`, `numpy`, `matplotlib`, `jupyter`
- [ ] **0.4** Collect 3–5 badminton rally videos (YouTube singles matches, your own footage)
- [ ] **0.5** Write a script that loads a video with OpenCV and iterates frames — confirm you can read, display, and save frames
- [ ] **0.6** Learn tensor basics: load an image as a NumPy array, resize, normalize, convert color spaces (BGR ↔ RGB)

### Milestone
> You can load any badminton video, extract frames, and manipulate them as arrays.

---

## Phase 1 — Court Detection & Player Isolation

*Before analyzing a player, you need to know where the court is and which blob is your player.*
### Checklist

- [ ] **1.1** **Court line detection** — use Canny edge detection + Hough line transform to find court lines in a frame
- [ ] **1.2** **Homography mapping** — compute a homography matrix that maps the camera view to a top-down court diagram (bird's-eye view)
  - Input: 4+ court corner points (manual annotation at first)
  - Output: a transform that maps any (x, y) pixel to a court coordinate in meters
- [ ] **1.3** **Player detection** — use a pretrained object detector (YOLOv8 `yolov8n.pt`) to detect people in each frame
- [ ] **1.4** **Player assignment** — given the user's selected court side (near/far), assign the correct bounding box to the target player using court position + homography
- [ ] **1.5** **Player tracking across frames** — use a simple tracker (SORT / ByteTrack / OpenCV CSRT) so the player ID persists across frames
- [ ] **1.6** Build a **visualization notebook** that overlays bounding boxes and court lines on video frames

### Milestone
> Given a video and a court side selection, you can isolate and track the target player across the entire rally.

---

## Phase 2 — Pose Estimation & Feature Extraction

*Extract the skeleton of the player and compute meaningful biomechanical features.*
### Checklist

- [ ] **2.1** Run **MediaPipe Pose** on cropped player frames — extract 33 landmark (x, y, z, visibility) per frame
- [ ] **2.2** Handle failure cases: landmarks not detected, player occluded, low resolution — implement interpolation / smoothing
- [ ] **2.3** Build a **`PoseFeatureExtractor`** class that computes per-frame features:
  - `elbow_angle` (racket arm) — angle(shoulder, elbow, wrist)
  - `shoulder_rotation` — angle between shoulder line and net
  - `knee_angle` (both legs)
  - `hip_angle`
  - `trunk_lean` — angle of spine relative to vertical
  - `wrist_height` — y-coordinate of racket wrist relative to shoulder
  - `foot_spread` — distance between ankles
  - `center_of_mass` — approximate from hip midpoint
- [ ] **2.4** Build **temporal features** (frame-to-frame):
  - `racket_hand_velocity` — pixel displacement of wrist per frame → convert to m/s using homography
  - `foot_velocity` — how fast feet are moving
  - `com_velocity` — center of mass movement speed
  - `recovery_speed` — speed returning to base position after a shot
- [ ] **2.5** Build a **pose visualization** that draws skeleton + feature values on each frame
- [ ] **2.6** Export feature sequences as NumPy arrays / CSV for downstream models

### Milestone
> For every frame of the target player, you have a feature vector like `[elbow_angle, knee_angle, shoulder_rotation, wrist_height, racket_velocity, ...]`.

---

## Phase 3 — Shot Detection & Classification

*Detect when a shot happens and classify what kind of shot it is.*

### Checklist

- [ ] **3.1** **Shot event detection** — detect the moment of contact:
  - Rule-based first: peak in `racket_hand_velocity` (the wrist accelerates, peaks at contact, decelerates)
  - Refine: combine with `wrist_height` to distinguish overhead vs underhand
- [ ] **3.2** **Segment rallies into shots** — each rally becomes a sequence of shot events with timestamps
- [ ] **3.3** **Label training data** — for each detected shot, label the shot type:
  - `smash` — steep downward, high contact, high velocity
  - `clear` — high arc, overhead, deep court
  - `drop` — soft, overhead, short landing
  - `drive` — flat, fast, mid-height
  - `net_shot` — near net, low contact, gentle
  - `lift` — underhand, high trajectory, deep
  - `block` — short swing, defensive
  - Use **Label Studio** or a simple custom Jupyter widget for labeling
- [ ] **3.4** **Build shot classifier model:**
  - Input: window of ~30 frames of pose features centered on contact point
  - Architecture: start with a small **1D CNN** or **LSTM** (2 layers, 64 hidden units)
  - Output: shot type (7 classes)
  - Framework: PyTorch
- [ ] **3.5** Train on your labeled data — aim for **>80% accuracy** on a validation set
- [ ] **3.6** Evaluate with confusion matrix — understand which shots get confused (smash vs clear is common)
- [ ] **3.7** (Optional) Try **shuttle tracking** to improve classification:
  - Detect shuttlecock using small object detection or background subtraction
  - Shuttle trajectory (angle, speed) is a strong signal for shot type

### Milestone
> Given a rally video, you can output: `Shot 1: smash @ 2.3s, Shot 2: net shot @ 4.1s, Shot 3: clear @ 5.8s, ...`

---

## Phase 4 — Form & Technique Analysis (Rule-Based + ML)

*This is where the coaching starts. Analyze each shot's technique and flag issues.*

### Checklist

- [ ] **4.1** Define **coaching rules** for each shot type. Start with the most impactful ones:

  **Split Step:**
  - [ ] **4.1a** Detect split step: both feet leave ground briefly before opponent's shot — look for `foot_y` dip pattern
  - [ ] **4.1b** Rule: if no split step detected before opponent hits, flag `"Missing split step"`
  - [ ] **4.1c** Rule: if split step timing is late (> 200ms after opponent contact), flag `"Split step too late"`

  **Overhead shots (smash, clear, drop):**
  - [ ] **4.1d** Rule: `contact_height` — if wrist_y at contact is below forehead height, flag `"Contact point too low — reach higher"`
  - [ ] **4.1e** Rule: `preparation` — if elbow_angle > 160° at preparation phase (arm not cocked back), flag `"Insufficient racket preparation"`
  - [ ] **4.1f** Rule: `body_rotation` — if shoulder_rotation < threshold at contact, flag `"Not rotating body into the shot"`
  - [ ] **4.1g** Rule: `base leg` — if knee_angle of back leg > 160° (too straight), flag `"Bend your base leg more"`

  **Net shots:**
  - [ ] **4.1h** Rule: if lunge `knee_angle` < 80° (too deep) or front knee past toes, flag `"Overcommitting on lunge"`
  - [ ] **4.1i** Rule: if `racket_preparation` happens after arriving at net (not during), flag `"Prepare racket earlier at the net"`

  **Recovery:**
  - [ ] **4.1j** Rule: measure `recovery_speed` — time from contact to returning to court center; if > threshold, flag `"Slow recovery to base"`
  - [ ] **4.1k** Rule: if player stays planted after shot for > N frames, flag `"Move back to base after your shot"`

- [ ] **4.2** Build a **`FormAnalyzer`** class that takes a shot event + surrounding pose features and returns a list of feedback strings with severity (info / warning / critical)
- [ ] **4.3** Test rules on real footage — tune thresholds by watching video and comparing feedback to what a coach would say
- [ ] **4.4** (Later) Train an **ML form scorer** — collect ratings (1–5) for shot quality and train a regression model on the pose features

### Milestone
> For each shot in a rally, you get feedback like: `"Shot 3 (smash @ 5.8s): Contact point too low. Rotate body more. Good racket preparation."`

---

## Phase 5 — Tactical / Shot Selection Analysis

*Was the right shot played given the situation?*

### Checklist

- [ ] **5.1** Build a **rally state representation** for each shot:
  - Player position on court (from homography)
  - Opponent position on court (detect + track opponent too)
  - Incoming shot type
  - Player's body orientation / readiness
  - Score context (if available)
- [ ] **5.2** Define **tactical rules** (start simple):
  - [ ] **5.2a** If opponent is at the net and you're deep → `clear` or `lift` is usually better than `drop`
  - [ ] **5.2b** If opponent is off-balance / out of position → `smash` or `fast drop` is recommended
  - [ ] **5.2c** If you're off-balance / late → `clear` to buy time is better than attacking
  - [ ] **5.2d** If opponent keeps returning to center → vary shot placement, flag `"Predictable shot pattern"`
  - [ ] **5.2e** Repeated same shot > 3 times → flag `"Too repetitive — mix up your shots"`
- [ ] **5.3** Build a **`TacticalAnalyzer`** class that takes the rally state and the shot played, then evaluates whether the shot choice was good
- [ ] **5.4** Output tactical feedback: `"Shot 5 (drop @ 8.2s): Opponent was behind you at rear court — a clear would have been safer here"`
- [ ] **5.5** (Advanced) Train a **shot recommendation model:**
  - Dataset: professional match rallies with labeled shots + positions
  - Model: given state → predict probability distribution over shot types
  - Compare player's actual shot to model's recommendation
  - Use **ShuttleSet** or similar research datasets

### Milestone
> The system not only critiques form, but also tells you: `"You played a smash here, but a cross-court drop would've been more effective given your opponent's position."`

---

## Phase 6 — Feedback Report Generation

*Aggregate all analysis into a coherent, readable coaching report.*

### Checklist

- [ ] **6.1** Design the **feedback report schema:**
  ```
  Report:
    video_metadata: { duration, resolution, fps }
    player_side: "near" | "far"
    rallies: [
      {
        rally_id: 1,
        start_time: 0.0,
        end_time: 12.5,
        shots: [
          {
            shot_id: 1,
            timestamp: 2.3,
            type: "smash",
            confidence: 0.92,
            form_feedback: [
              { message: "Contact point too low", severity: "warning", detail: "..." }
            ],
            tactical_feedback: [
              { message: "Good shot selection — opponent off balance", severity: "info" }
            ]
          },
          ...
        ],
        movement_feedback: [
          { message: "Slow recovery after shot 2", severity: "warning" }
        ]
      },
      ...
    ]
    summary: {
      strengths: ["Good smash power", "Consistent split step"],
      weaknesses: ["Late contact on clears", "Slow recovery to base"],
      drills: ["Shadow footwork drill for recovery speed", "Wall practice for overhead contact point"]
    }
  ```
- [ ] **6.2** Build a **`ReportGenerator`** class that takes all analysis outputs and produces the structured report
- [ ] **6.3** Generate a **summary** using heuristics:
  - Count frequency of each feedback type
  - Strengths = things done correctly most of the time
  - Weaknesses = most frequent issues
  - Drills = mapped from weaknesses to a predefined drill bank
- [ ] **6.4** (Optional) Use an **LLM** (local or API) to generate natural-language summaries from the structured data
- [ ] **6.5** Build a **highlight reel**: for each piece of feedback, output the relevant video clip timestamp so the frontend can show it

### Milestone
> The pipeline produces a complete JSON report from a video, ready for the frontend to render.

---

## Phase 7 — Backend API

*Wrap the pipeline in a web service.*

### Checklist

- [ ] **7.1** Set up **FastAPI** project with proper structure
- [ ] **7.2** Implement endpoints:
  - `POST /upload` — upload video, returns a `job_id`
  - `POST /analyze` — start analysis for a video with `{ video_id, player_side }`
  - `GET /status/{job_id}` — poll for progress (frame count, current step)
  - `GET /report/{job_id}` — get the finished feedback report
- [ ] **7.3** Implement **background processing** — video analysis is slow (minutes), so use a task queue:
  - Simple: `BackgroundTasks` in FastAPI for MVP
  - Production: Celery + Redis or `arq`
- [ ] **7.4** **File storage** — save uploaded videos and generated reports
  - Local filesystem for dev
  - S3-compatible for production
- [ ] **7.5** Add **progress tracking** — the pipeline emits progress events so the frontend can show a progress bar
- [ ] **7.6** Add basic error handling, input validation (file type, file size, duration limits)
- [ ] **7.7** Dockerize the backend (Dockerfile + docker-compose with any dependencies like Redis)

### Milestone
> You can `curl` the API: upload a video, start analysis, poll for completion, and get back a full coaching report as JSON.

---

## Phase 8 — Frontend Application

*Build the user-facing app.*

### Checklist

- [ ] **8.1** Set up **React** project (Vite + TypeScript)
- [ ] **8.2** **Upload page:**
  - Drag-and-drop video upload with progress bar
  - Video preview after upload
  - Court side selector: show a court diagram, let user click near or far side
  - "Analyze" button
- [ ] **8.3** **Processing page:**
  - Progress bar showing pipeline stages (extracting frames → detecting poses → classifying shots → generating report)
  - Estimated time remaining
- [ ] **8.4** **Report page — Shot Timeline:**
  - Video player with shot markers on the timeline (click to jump)
  - Shot-by-shot breakdown: timestamp, shot type, form feedback, tactical feedback
  - Color-coded severity (green = good, yellow = warning, red = critical)
- [ ] **8.5** **Report page — Summary:**
  - Strengths and weaknesses cards
  - Suggested drills
  - Overall score / rating (optional)
- [ ] **8.6** **Report page — Court Visualization:**
  - Top-down court diagram showing player movement path
  - Shot placement markers
  - Heatmap of court coverage
- [ ] **8.7** **Report page — Video Overlay:**
  - Play video with skeleton overlay drawn on the player
  - Highlight moments where feedback was triggered (e.g., flash red on late split step)
- [ ] **8.8** Make it responsive (works on mobile for quick upload from phone)
- [ ] **8.9** Polish UI/UX — loading states, error handling, empty states

### Milestone
> A complete web app: upload video → pick court side → wait for analysis → view a rich coaching report with video playback, timeline, court diagrams, and actionable feedback.

---

## Phase 9 — Iteration, Data, & Improvement

*The system works. Now make it good.*

### Checklist

- [ ] **9.1** **Collect feedback** — use the app yourself and with friends; note where the AI is wrong
- [ ] **9.2** **Expand the dataset** — label more shots, more videos, more playing styles
- [ ] **9.3** **Improve shot classifier** — try better architectures (Transformer, Video Swin), more data augmentation
- [ ] **9.4** **Calibrate thresholds** — form analysis thresholds will vary by player level; consider beginner / intermediate / advanced presets
- [ ] **9.5** **Add doubles support** — detect 2 players per side, more complex positioning
- [ ] **9.6** **Shuttle trajectory tracking** — improves shot classification and enables placement analysis
- [ ] **9.7** **Opponent analysis** — "your opponent tends to drop when you clear to their backhand"
- [ ] **9.8** **Session history** — track improvement over time; compare metrics across uploaded videos
- [ ] **9.9** **Model optimization** — quantize models, use ONNX runtime or TensorRT for faster inference
- [ ] **9.10** **Deploy** — cloud deployment (AWS/GCP), GPU instance for inference, CDN for frontend

### Milestone
> A polished, accurate, continuously improving badminton coaching tool.

---

## Quick Reference: Tech Stack

| Layer              | Technology                              |
| ------------------ | --------------------------------------- |
| Pose Estimation    | MediaPipe Pose / MoveNet                |
| Object Detection   | YOLOv8 (ultralytics)                    |
| Tracking           | ByteTrack / SORT                        |
| Feature Extraction | NumPy, SciPy                            |
| Shot Classifier    | PyTorch (LSTM or 1D CNN)                |
| Form Analysis      | Rule-based (Python) + ML scorer         |
| Tactical Analysis  | Rule-based → ML (later)                 |
| Backend            | FastAPI, Celery/arq, Redis              |
| Frontend           | React, TypeScript, Vite                 |
| Video Processing   | OpenCV, FFmpeg                          |
| Data Labeling      | Label Studio / CVAT                     |
| Deployment         | Docker, AWS/GCP                         |

---

## Quick Reference: Key Datasets

| Dataset                     | What it provides                           | Link / Source                              |
| --------------------------- | ------------------------------------------ | ------------------------------------------ |
| **ShuttleSet**              | Rally-level shot labels, player positions  | Search "ShuttleSet badminton dataset"      |
| **Badminton Olympic videos**| High-quality match footage                 | YouTube (BWF channel)                      |
| **Your own footage**        | Realistic amateur data for your use case   | Film at your court with a tripod           |
