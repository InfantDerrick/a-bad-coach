# Badminton AI Coach — Offline-First MVP Plan

> **Product promise:** Import a fixed-camera badminton clip, process it locally with the user's GPU, and receive a trustworthy, evidence-linked performance dashboard: rallies, shots, movement, split-step timing, recovery, shuttle trajectory when measurable, and a short list of specific things to train next.

## 1. Product decision

The MVP is **post-session, local, and singles-first**.

- The video stays on the user's computer. After models are installed, analysis does not require the internet.
- "Upload" means selecting or dragging a local video into the application; there is no mandatory cloud upload.
- The application uses available hardware acceleration: CUDA on NVIDIA, Metal/MPS on Apple silicon, and CPU fallback. ONNX Runtime is a later portability option.
- Start with one fixed camera behind a baseline, the full court visible, minimal camera movement, and one selected player.
- Build the processing engine and a local dashboard before mobile capture, GoPro control, accounts, cloud sync, or live coaching.
- Every metric links back to the relevant timestamp/clip and carries a confidence or an "unavailable" reason.

This replaces the earlier backend/GoPro-first direction. Capture integrations can return after the analysis itself is useful and validated.

## 2. What the MVP should answer

For an advanced player, counts alone are not coaching. The dashboard should help answer:

1. **Am I prepared on time?** Split-step timing relative to opponent contact, landing timing, stance width, first-step reaction, and initial movement direction.
2. **Do I reach the shuttle efficiently?** Distance, peak/average speed, acceleration and deceleration, directness of path, extra steps, arrival margin, lunge depth, and balance at contact.
3. **Do I recover well?** Time and distance back toward a context-dependent base, recovery completeness before the next opponent contact, and repeated late-recovery patterns by court zone.
4. **What am I actually playing?** Shot type and attributes, contact zone, forehand/backhand, overhead/underarm, straight/cross, court origin and destination, plus confidence.
5. **Are my shots effective?** Opponent displacement, time removed from the opponent, depth/width, net clearance and trajectory when observable, and whether the next ball leaves the player attacking, neutral, or defending.
6. **What breaks down under pressure?** Metrics split by rally length, late-rally phase, movement direction, incoming shot, and high-pressure/low-time situations.
7. **What should I train next?** The top two or three recurring, high-confidence issues, each supported by clips and mapped to a drill. Compare with the player's own history first and an appropriate pro cohort second.

Research on elite play supports making temporal load, stroke frequency, movement speed/variability, acceleration/deceleration, distance, split-step reaction, and positional/shot sequences central rather than generating a vague overall technique score. See [Research basis](#12-research-basis).

## 3. MVP report and dashboard

### 3.1 Input quality and calibration

Before showing performance data, show:

- resolution, FPS, shutter/motion-blur warning, duration, camera stability, and court visibility;
- detected court with editable anchors and homography quality;
- selected player and opponent tracks with correction controls;
- which metric tiers the clip supports.

Recommended capture:

- **Core movement/shot report:** 1080p at 60 FPS or better.
- **Shuttle trajectory and speed beta:** 4K or tightly framed 1080p at 120 FPS or better, a fast shutter, stable camera, and the entire flight visible.
- Manual court-point correction and player selection are acceptable MVP steps. Five seconds of correction is better than an entire polished but wrong report.

### 3.2 Session overview

- analyzed minutes, rallies, active/rest time, shots, shots per rally, rally-duration distribution, and shot interval;
- movement distance, speed distribution, acceleration/deceleration count, six-zone workload, and movement heatmap;
- shot mix and outcome/context splits;
- three evidence-backed findings and suggested drills;
- model version, confidence coverage, and percentage of events requiring review.

### 3.3 Rally explorer

- video with player pose, court coordinates, and shuttle trail overlays that can be toggled;
- rally and shot markers on the timeline;
- top-down court path and shot map;
- per-shot event card: timestamp, hitter, shot class/attributes, origin/destination, player positions, split-step/reaction/recovery timing, and confidence;
- filters for shot, court zone, forehand/backhand, rally length, confidence, and detected issue;
- one-click correction of rally boundaries, contacts, shot label, and player identity. Corrections become labeled training data only with explicit user opt-in.

### 3.4 Advanced-player views

| View | Metrics | Coaching use |
| --- | --- | --- |
| Readiness | split-step start/landing vs opponent contact, stance width normalized by height, first-step latency and direction | Find late or mistimed preparation by incoming shot and zone |
| Movement | distance, speed percentiles, accelerations/decelerations, directness, extra steps, six-corner transitions | Identify inefficient routes and costly directions |
| Arrival/contact | arrival margin, contact height normalized by body height, body balance, lunge/rear-court position | Separate slow recognition from slow movement or poor spacing |
| Recovery | recovery time/distance, position at next opponent contact, dynamic-base error | Find shots and zones that prevent readiness for the next ball |
| Shot profile | type, attributes, origin/destination, straight/cross, depth/width, error/winner context | Reveal predictability, weak zones, and missing options |
| Rally pressure | tempo, strokes, duration, workload, metric change by rally phase/length | Show what degrades when pace or fatigue rises |
| Effectiveness | opponent displacement, response time, next-shot pressure state, terminal three-shot patterns | Judge shot value, not just shot appearance |
| Progress | personal baseline, rolling trend, matched pro reference distribution | Turn a report into a training loop |

Do not ship a single opaque "pro score." Show distributions, sample sizes, uncertainty, and the clips that caused each conclusion.

## 4. Measurement truth: what court lines can and cannot do

Known court geometry provides a real-world scale on the **court plane**. A homography can therefore measure player foot positions, planar distance, planar speed, zones, and approximate shuttle landing points.

It does **not** by itself recover shuttle height or true 3D speed. The shuttle is often above the court plane, loses speed rapidly to drag, is tiny, blurred, and may be visible in only a few frames. From one ordinary camera:

- report `player_ground_speed` confidently after calibration;
- report `shuttle_2d_apparent_speed` only as a diagnostic, never label it smash speed;
- report `shuttle_initial_speed_estimate` only when contact is detected, enough post-contact observations exist, the camera is calibrated, and a validated 3D/physics fit passes quality gates;
- otherwise show "speed unavailable for this clip" and explain why;
- validate speed against radar or synchronized high-speed reference footage before product release.

Monocular 3D reconstruction is possible with court dimensions, player pose, hit recognition, shuttle tracking, and a drag-aware flight model, but published work describes it as an ill-posed, engineering-heavy problem. It belongs in a qualified beta, not the first promise.

Video also cannot provide heart rate, internal training load, grip pressure, racket-head speed hidden by blur, impact location on the strings, or ground-reaction force. Call the output **video-derived telemetry**, not "all telemetry."

## 5. Local architecture

```text
Local video import
      |
      v
Media probe + quality gate ----> calibration/player correction UI
      |
      v
Frame decode (FFmpeg/OpenCV)
      |
      +--> court/net detection + camera calibration
      +--> player detection/tracking + pose
      +--> shuttle tracking + trajectory confidence
      +--> rally/contact/shot event detection
      |
      v
Court-coordinate event model
      |
      +--> deterministic metric engine
      +--> confidence/abstention engine
      +--> feedback rules + drill mapping
      |
      v
Local report store (JSON + SQLite + derived clips)
      |
      v
Local dashboard and video overlays
```

Implementation direction:

- Python pipeline, PyTorch for research/training, FFmpeg/OpenCV for video, and Polars/Pandas plus NumPy/SciPy for metrics.
- Start with a local CLI and generated HTML or localhost dashboard. Package as a desktop app only after the vertical slice is stable.
- Keep model inference, metric calculation, and feedback generation separate. An optional local LLM may rewrite structured findings, but it must not invent detections or measurements.
- Cache each stage by video hash and model/config version so users can correct calibration or rerun one model without repeating everything.
- Preserve raw detections, transformed coordinates, derived events, and final metrics as separate versioned artifacts.

## 6. Canonical event schema

The application needs one auditable representation before it needs many models.

```yaml
session:
  video: {hash, fps, resolution, duration, capture_quality}
  calibration: {court_points, homography, reprojection_error, confidence}
  models: [{name, version, dataset_version}]
players:
  - {track_id, selected, side, handedness_optional, height_optional}
rallies:
  - rally_id: 1
    start_s: 2.10
    end_s: 11.82
    outcome: unknown
    shots:
      - shot_id: 1
        contact_s: 2.34
        hitter_track_id: 4
        type: clear
        attributes: {hand: forehand, height: overhead, direction: straight}
        origin_m: [x, y]
        destination_m: [x, y]
        shuttle: {observations: 18, speed_mps: null, confidence: 0.61}
        readiness: {split_land_offset_ms: 22, first_step_ms: 247}
        recovery: {time_ms: 610, base_error_m_at_next_contact: 0.42}
        confidence: {contact: 0.91, type: 0.84, destination: 0.70}
        evidence: {start_s: 1.9, end_s: 3.2}
```

Unknown is a valid value. Every derived metric must retain provenance and confidence from its required inputs.

## 7. Model and algorithm plan

### 7.1 Court and camera

- Treat the regulation court as a canonical geometric template. Detect whatever named intersections/line segments are visible, fit them to the template with RANSAC, and extrapolate boundaries that are outside the frame.
- Pretrain on synthetic court projections with randomized camera angle, perspective, roll, crop, partial visibility, blur, floor color, and occlusion; then fine-tune on labeled real footage from the target camera domain.
- Use landmark heatmaps first and add court-line segmentation as a complementary signal for views with too few visible intersections. Retain the existing Hough/homography prototype only as a baseline.
- Detect camera cuts or movement and recalibrate by segment.
- Use net/post evidence and line identity to reject mirrored or geometrically impossible court fits.
- Once a static-camera calibration passes quality gates, stabilize/reuse it across the clip instead of independently redetecting the court on every frame.

### 7.2 Players and pose

- Detect both players, assign near/far court identities, and track through occlusion.
- Estimate pose on player crops; smooth without erasing fast foot contacts.
- Derive court position primarily from foot contact/mid-ankle geometry rather than bounding-box center.
- Treat far-player technique metrics as lower confidence when pixel height is insufficient.

### 7.3 Rally, contact, and shot events

- Segment active play using player motion, audio peaks, shuttle evidence, and temporal rules.
- Detect contacts using fused evidence: shuttle trajectory change, wrist/racket-side motion, audio, alternating hitter logic, and legal timing.
- Classify an MVP coarse taxonomy: `serve`, `clear_or_lift`, `drop`, `net`, `drive_or_push`, `smash`, `block`, `unknown`.
- Store independent attributes (`forehand/backhand`, `overhead/underarm`, `straight/cross`, contact zone) instead of forcing every distinction into one class.
- Expand toward ShuttleSet's finer taxonomy only when the coarse classes meet validation gates.

### 7.4 Shuttle and trajectory

- Benchmark TrackNetV3-style temporal heatmaps against a small-object detector on the actual fixed-camera domain.
- Constrain candidates using court volume, alternating contacts, plausible flight continuity, and player pose.
- Keep original observations and rectified/interpolated points distinct.
- Estimate landing and net crossing in 2D first. Add drag-aware monocular 3D reconstruction and speed only after ground-truth validation.

### 7.5 Metrics and feedback

- Calculate metrics deterministically from versioned events.
- Establish each player's personal baseline; compare like-for-like contexts before comparing with pros.
- Generate feedback only when an issue repeats, has sufficient sample size, exceeds uncertainty, and has a useful intervention.
- Each recommendation contains: finding, magnitude, context, clips, confidence, and one drill/test for the next session.

## 8. Training on professionals without fooling ourselves

"Train on current pros" should mean a governed data program, not scraping a few highlight reels.

### 8.1 Three complementary datasets

1. **Public structured data:** use ShuttleSet/ShuttleSet22 for shot sequences, hitter/player locations, tactical forecasting baselines, and a fine-grained label vocabulary. ShuttleSet contains 44 matches, 3,685 rallies, and 36,492 strokes from 27 high-ranking players, but its footage is from 2018–2021; it is a starting point, not a current-pro ground truth.
2. **Current professional benchmark set:** obtain permission or confirm usable rights for full-match footage from recent seasons. Sample men and women, left/right handed players, multiple styles, venues, camera views, and tournament rounds. Label contacts, shot type/attributes, positions, outcomes, and a smaller coach-reviewed technique/tactical subset. Refresh on a scheduled cadence.
3. **Target-domain footage:** collect fixed-behind-baseline clips from the cameras, frame rates, venues, and advanced amateur players the product will actually see. This is essential because broadcast footage and user footage have different viewpoints, compression, cuts, and player scale.

Professional footage supplies reference distributions and tactical patterns; it should not define one universal "perfect" motion. Technique varies with body, handedness, intent, incoming shuttle, and tactical context.

### 8.2 Annotation and quality

- Write a versioned label guide with examples and an explicit `uncertain/unknown` option.
- Have badminton-knowledgeable annotators label all events; double-label the validation/test subsets and adjudicate disagreements.
- Use model-assisted annotation, but never allow predictions to silently become ground truth.
- Capture source, rights/consent, competition, player metadata, camera view, FPS, and label provenance.
- Keep a coach-reviewed challenge set of deceptive shots, dives, occlusions, motion blur, net exchanges, and long rallies.

### 8.3 Evaluation discipline

- Split by **match, player, venue, and video**, never by adjacent frames. Hold out unseen players and at least one target camera/venue.
- Report per-class precision/recall/F1, confusion matrices, calibration error, and performance by near/far player, camera quality, sex/discipline, and movement zone.
- Test public-dataset pretraining against target-domain fine-tuning; do not assume more pro footage automatically improves user-video performance.
- Version datasets, weights, thresholds, and model cards. The dashboard records exactly which versions produced a report.
- Review subgroup gaps and abstain where the model is not supported.

## 9. Validation and release gates

Create a manually annotated gold set before calling a metric complete.

| Component | Validation | Initial release gate |
| --- | --- | --- |
| Court calibration | landmark reprojection error in court coordinates | median error at marked landmarks ≤ 0.10 m on supported clips |
| Player location | foot-point error vs manual court coordinate | median ≤ 0.25 m |
| Tracking | IDF1/identity switches by rally | no uncorrected identity switch in ≥ 95% of supported rallies |
| Rally segmentation | boundary precision/recall | F1 ≥ 0.90 within 0.5 s |
| Contact detection | event precision/recall | F1 ≥ 0.85 within 100 ms |
| Shot classification | macro F1 and per-class recall | macro F1 ≥ 0.80; no shipped class below 0.65 recall |
| Split/first step | timing error vs frame annotation | median absolute error ≤ 2 frames on qualified views |
| Shuttle tracking | detection F1 plus trajectory completeness | ≥ 85% observed/valid trajectory frames on clips labeled supported |
| Shuttle speed | error vs radar/high-speed reference | median absolute percentage error ≤ 10% before showing km/h/mph |
| Feedback | blind coach agreement and usefulness rating | ≥ 80% of surfaced findings judged correct and actionable |

These are product targets, not claims about the current prototype. If a clip fails the input or confidence gate, the relevant metric must abstain.

## 10. Build order

### Phase A — Offline vertical slice (current priority)

- [x] Read local video and extract frames.
- [x] Prototype court lines, court corners/homography, and a player box on the sample clip.
- [x] Define the initial movement report schema and artifact folders.
- [x] Add media probe and capture-quality report.
- [ ] Make court calibration editable and persist it.
- [x] Track a manually seeded player in court meters and render an overlay/top-down path (provisional CSRT baseline).
- [x] Produce a local JSON/CSV report plus a minimal HTML dashboard.

**Exit:** import one supported clip and see a verified court, selected player path, distance/speed, heatmap, and evidence video without a server.

### Phase B — Useful movement and rally MVP

- [ ] Track both players and segment rallies.
- [ ] Add pose with visibility-aware smoothing.
- [ ] Detect split-step, first step, lunges, contacts, and recovery events.
- [ ] Calculate readiness, movement, recovery, rally tempo, and workload metrics.
- [ ] Build rally explorer, filters, confidence, and correction UI.
- [ ] Validate against a manually labeled target-domain gold set.

**Exit:** reliably explain where and why the selected player loses time, with clips and measurable trends.

### Phase C — Shot intelligence MVP

- [ ] Build fused contact detection.
- [ ] Label and train the coarse shot taxonomy plus independent attributes.
- [ ] Add shot origin/destination and zone-transition views.
- [ ] Add effectiveness measures using opponent movement and next-shot state.
- [ ] Generate the top recurring findings and map them to a curated drill bank.

**Exit:** the report connects movement to shot selection and shot effect instead of listing isolated statistics.

### Phase D — Shuttle trajectory and speed beta

- [ ] Fine-tune shuttle tracking on supported target-camera footage.
- [ ] Add trajectory rectification, landing/net-crossing estimates, and quality gates.
- [ ] Build and validate drag-aware 3D reconstruction.
- [ ] Validate initial speed with radar or synchronized high-speed cameras.
- [ ] Ship speed only for clips that pass the qualified-input and trajectory gates.

**Exit:** trajectory/depth/placement are useful and speed is measured with known error, never guessed.

### Phase E — Pro reference and longitudinal coaching

- [ ] License/approve and annotate a diverse current-pro benchmark set.
- [ ] Build context-matched reference distributions, not a universal pro score.
- [ ] Add session history, personal baselines, trend detection, and before/after drill tests.
- [ ] Add coach review workflow and active-learning queue for opted-in corrections.
- [ ] Expand fine shot classes and tactical sequence analysis when validation supports them.

**Exit:** advanced players can test whether targeted training changes the exact metric and match context that triggered the recommendation.

### Explicitly after MVP

- doubles and team-formation analysis;
- live/realtime feedback;
- GoPro/iOS camera control;
- cloud accounts, sharing, or team dashboards;
- automatic scoring from arbitrary broadcast footage;
- physiological load without a wearable integration.

## 11. Immediate next milestone

Build the smallest end-to-end artifact from the existing sample video:

1. probe video quality and FPS;
2. load/edit the existing court homography;
3. track one selected player and map the foot point to meters;
4. calculate path, distance, speed distribution, and six-zone occupancy;
5. render an overlay video, top-down trace, and local report JSON/HTML;
6. manually inspect errors and turn them into the first gold-set annotations.

Do not start shot-speed modeling until this coordinate pipeline is quantitatively validated.

## 12. Research basis

- [ShuttleSet paper](https://arxiv.org/abs/2306.04948): 104 sets, 3,685 rallies, 36,492 strokes, 18 shot classes, hit locations, and both player locations across 44 elite matches from 2018–2021.
- [ShuttleSet22 / stroke forecasting benchmark](https://www.ijcai.org/proceedings/2024/1042.pdf): adds high-ranking 2022 match data and benchmark tasks/code for future-stroke forecasting.
- [TrackNetV3](https://people.cs.nycu.edu.tw/~yushuen/data/TrackNetV3.pdf): a temporal shuttle-tracking baseline with augmentation and trajectory rectification; useful to benchmark, not assume it transfers unchanged to user footage.
- [MonoTrack (CVPR Workshops 2022)](https://openaccess.thecvf.com/content/CVPR2022W/CVSports/html/Liu_MonoTrack_Shuttle_Trajectory_Reconstruction_From_Monocular_Badminton_Video_CVPRW_2022_paper.html): combines court dimensions, shuttle tracking, hit recognition, pose, and physics for monocular 3D trajectory reconstruction while documenting why the problem is difficult.
- [Characteristics of split-step skills of the world's top athletes](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0316632): reports split-step position, stance width, and reaction-time measurements in top players and supports context-specific readiness analysis.
- [Elite footwork indicators using markerless measurement](https://www.frontiersin.org/journals/sports-and-active-living/articles/10.3389/fspor.2026.1753118/full): uses movement speed, speed variability, accelerations/decelerations, and distance as performance/load indicators and compares world-class with high-school players.
- [Long rallies and next-rally performance](https://pmc.ncbi.nlm.nih.gov/articles/PMC7053721/): supports rally time, rest time, density, stroke count, interval, outcome, and contextual analysis rather than isolated averages.
- [ShuttleNet](https://ojs.aaai.org/index.php/AAAI/article/view/20341): demonstrates position-, rally-, and player-style-aware stroke forecasting, relevant to later tactical baselines.
- [BWF Laws of Badminton](https://system.bwfbadminton.com/documents/folder_1_81/Statutes/CHAPTER-4---RULES-OF-THE-GAME/SECTION%204.1-%20Laws%20of%20Badminton.pdf): authoritative court/net geometry for calibration constraints.

## 13. MVP success criteria

The MVP is successful when an advanced singles player can import a supported clip, correct setup mistakes quickly, and within one local workflow receive:

- accurate rally and movement telemetry;
- a trustworthy coarse shot breakdown;
- split-step, first-step, arrival, and recovery evidence;
- court and rally-context views that expose a repeated weakness;
- no more than three prioritized training recommendations with supporting clips;
- transparent confidence and abstention wherever the video cannot support a measurement.

The goal is not the largest dashboard. It is the smallest report an advanced player trusts enough to change the next practice session.
