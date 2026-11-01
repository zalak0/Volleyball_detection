# Volleyball Player Detection & Court Mapping — System Design

*AMME5710 project · Zahi Al-Aker · prototype validated on IMG_4838.MOV*

## 1. Goal

Given match video from fixed cameras behind each baseline, output for every frame:
each player's bounding box, a persistent ID, their team, and their **position on the court in metres**,
rendered as a top-down minimap (like the RF-DETR + RTMPose demo by Selçuk Görgülü).

## 2. What the footage looks like (and why it matters)

Findings from the sample clip (1920×1080, 59.94 fps, HEVC, 2 min):

| Observation | Consequence for the design |
|---|---|
| Camera is **static** (tripod, elevated about 4 m, behind the near baseline, iPhone 0.5× ultra-wide) | One homography per camera setup is enough. No per-frame line detection is needed. |
| Ultra-wide lens, but iPhone distortion correction leaves court lines straight | A plain 3×3 homography fits. Projected net line lands on the post bases; posts map to 0.9 m and 1.2 m outside the sidelines (regulation is 0.5–1 m). |
| Strong perspective: near players are about 250 px tall, far players about 60–80 px | **The main problem.** Downscaling the full court to 1280 px loses the far team (0 of 6 detected). It needs a magnified far-court tile. |
| Multi-sport floor (white, red and yellow lines). The volleyball court is **yellow**. | Calibrate on the yellow lines. Automatic line extraction is unreliable because of floor glare, so use manual click calibration. |
| A second court with other players on the right, plus spectators and subs near the walls | Gate detections in **court coordinates**, not pixels. The adjacent court sits at x > 12 m. |
| Near players overlap far players in the image; the net mesh covers far players' legs | Expect lower recall and noisier foot points on the far side. Ideally the far side is covered by **the other camera**. |

## 3. Architecture

```
          ┌───────────── once per camera placement ─────────────┐
          │ calibrate.py: click ≥4 court landmarks → H (3×3),    │
          │ detection ROI, far-court tile, playing-area gate     │
          └──────────────────────────────────────────────────────┘
frame ─► [1] Tiled detection ─► [2] Ground point ─► [3] Image→court (H⁻¹)
          full-court ROI          ankle midpoint        metres (x across, y along)
        + magnified far tile      (fallback: bbox
          → merge with NMS         bottom-centre)
                                                          │
        [6] Outputs ◄── [5] Tracking ◄── [4] Gate + team ◄┘
        annotated video   ByteTrack IDs     keep −1.5≤x≤10.5, −2.5≤y≤20.5
        minimap + trails  + court-space EMA team = near if y<9 else far
        positions.csv
```

### [1] Detection — YOLO11m-pose, two-scale tiling
- A pose model gives person boxes **and** 17 keypoints in one pass, so there is no separate detector.
- **Tile A**: the whole playing area (ROI crop, inferred at 1280).
- **Tile B**: only the far half of the court, magnified about 1.35×. On the same frame, far players went from undetected to 0.87–0.91 confidence.
- Merge: drop boxes cut by an internal tile edge, then NMS at IoU 0.5. This is SAHI-style slicing, specialised to the known court geometry.
- Upgrade path: fine-tune on 300–500 labelled frames from your own gym (Roboflow export to YOLO format). Volleyball-specific failures, such as jumping players, arms above the net and players in dark clothing against the wall, drop sharply after fine-tuning. RF-DETR, used in the reference video, is a drop-in alternative.

### [2] Ground contact point
The homography is only valid for points **on the floor plane**, so the choice of image point decides accuracy.
- Use the midpoint of the ankle keypoints with confidence > 0.5 (lowest ankle for v), plus a small offset down to the sole.
- Fall back to the bbox bottom-centre. That is worse when legs are occluded, because the box ends at the knee.
- **Jumps**: a jumping player's feet leave the plane and the projection pushes them away from the camera. Handle this by freezing the court position while the ankle rises faster than a threshold, or by using the Kalman prediction. This is not implemented in the prototype yet.

### [3] Homography (your Assignment 1 Q2 rectification, applied to a court)
`H` maps court metres to pixels, estimated by DLT from point correspondences (`cv2.findHomography`, with RANSAC when there are more than 4 points).
Court frame: x ∈ [0, 9] across, y ∈ [0, 18] along (0 = near baseline, 9 = net).
The prototype uses 4 near-court points (baseline corners and attack-line intersections). The projected far attack line and far baseline land on the painted lines, which validates the fit.
**Resolution check:** near court is about 1 px ≈ 1 cm; at the far baseline about 1 px ≈ 6–8 cm along y. Far-side depth is therefore inherently less precise from one camera.

### [4] Gating and team assignment
- Keep a detection only if its court position is inside the court plus a free-zone margin. This alone removes the adjacent court, spectators, the bench and people walking past.
- **Team assignment is free in volleyball**: players never cross the net, so team = sign(y − 9). No jersey-colour clustering is needed. That is only needed to separate refs or liberos, or to identify individuals.
- Optional: at most 6 per side. If more than 6 remain, keep the 6 with the highest track age and confidence.

### [5] Tracking
- ByteTrack on the merged detections. It also associates low-confidence boxes, which matters for occluded far players.
- Exponential smoothing of each track's court position. Upgrade: a constant-velocity Kalman filter **in court metres**, with Hungarian matching on Mahalanobis distance. That is more stable than image-IoU matching because motion is linear on the floor, not in perspective pixels.
- ID persistence through long occlusions: add appearance ReID (e.g. BoT-SORT/OSNet), or exploit the fact that the roster is fixed at 6 per side and re-bind lost IDs to unmatched tracks on the same side.

### [6] Two-camera fusion (for footage with a camera behind each team)
This is the answer to the shallow-angle problem. **Each camera owns the half of the court nearest it**, where players are large, feet are visible and depth resolution is about 1 cm/px.
1. Calibrate each camera to the **same** court frame. For the camera behind the far team, click the same landmarks with x and y mirrored: (x, y) → (9 − x, 18 − y).
2. Time-sync the two videos: audio cross-correlation of a whistle or ball contact (`ffmpeg` + `scipy.signal.correlate`), or a clap at the start.
3. Per frame, take camera A's detections with y < 9 + 1 and camera B's with y > 9 − 1. In the overlap band near the net, associate across cameras by court distance (< 0.7 m) and average positions weighted by each camera's local resolution.
4. A single cross-camera tracker then runs in court space. This also fixes occlusion, since a player hidden behind a teammate in one view is usually clear in the other.

## 4. Prototype results (IMG_4838.MOV, t = 40–80 s, processed at 10 fps on CPU)

| Metric | Near half | Far half |
|---|---|---|
| Players placed per frame (mean) | 4.8 | 2.8 |
| Foot point from ankles (vs bbox fallback) | 98 % | 100 % |
| Track IDs created over 40 s | 11 | 5 |
| Tracks lasting ≥10 s | 7 | 3 |

- **Single-scale vs tiled detection on the same frame:** the far team went from 0 to 3 detected. Tiling is the single most important design decision for this camera angle.
- The gate correctly ignored the adjacent court and the people near the walls (grey boxes in the video).
- **Not yet measured:** true recall. These counts are detections, not accuracy against ground truth, and a casual game can have fewer than 6 per side. Section 5 describes how to label and score it.
- **Known failure modes seen:** ID switches when near players cross or overlap (11 IDs for about 6 people). A player chasing the ball past the 1.5 m gate is dropped. Far-team positions jitter more in y than near-team ones, as the resolution analysis predicts.
- **Speed:** about 2.5 s/frame on a 2-core CPU with both tiles. A laptop GPU should reach real time at 1280 px.

## 5. Evaluation plan (for the report)
- **Detection**: hand-label boxes on about 100 frames spread over the clip with CVAT or Roboflow. Report precision, recall and AP50 separately for the near and far halves, comparing single-scale vs tiled detection (the main ablation).
- **Tracking**: IDF1, ID switches and HOTA via TrackEval, from the same labels with IDs.
- **Position accuracy**: put cones or tape at known court positions (e.g. the 9 zone centres), stand on each, and compare the projected foot point to the truth. Report error vs distance from the camera.
- **Calibration**: reprojection error of held-out landmarks. Use 4 points to fit and the far-court lines to test, which is the same check used here.

## 6. Links to your AMME5710 work
- **Homography / rectification**: A1 Q2 board localisation and rectification is the same operation, applied to the court plane.
- **Outlier rejection and RANSAC**: A2 Q1's chained geometric tests are the same mindset as the court-space gate and the RANSAC homography fit.
- **Pre-processing**: CLAHE (A2) is worth testing on the far tile, since the glossy floor and backlighting wash out the far-court players.
- **Classical baseline**: a median-background model of the static camera (computed here from 60 frames) plus foreground blobs is a cheap baseline to compare against YOLO in the report.

## 7. Files
- `vb_tracker.py` — the full pipeline (detection → court mapping → tracking → video, minimap and CSV)
- `calibrate.py` — click-to-calibrate tool that writes `calib.json` (H, ROI, far tile, gate)
- `calib.json` — calibration for this camera placement
- `positions.csv` — per-frame output: frame, t_s, track_id, team, x_m, y_m, u_px, v_px, conf, foot_src

Run:
```
pip install ultralytics supervision==0.26.1
python calibrate.py --video IMG_4838.MOV --out calib.json        # once per setup
python vb_tracker.py --video IMG_4838.MOV --calib calib.json --start 40 --dur 40 --fps 10
```
On a laptop GPU, use `--fps 30` or full frame rate. On CPU it runs about 2.5 s per frame with both tiles.

## 8. Stage 2 — Ball tracking, touch attribution and rule checks (Option B)

### 8.1 Ball detection: colour + foreground + 3-D gate (classical, full 60 fps)
COCO's "sports ball" class (YOLO11m, 4 tiles) found the ball in only about half of the frames, at about 5 s/frame. It also fired on a static wall object and the next court's ball. The ball is a yellow/blue Mikasa about 15 px across, and it is the most saturated yellow object in the hall. Candidates are pixels that are:
- **yellow** (HSV 14–36°, S and V > 110), **AND**
- **foreground** (max channel difference > 35 against a median background of the static camera), **AND**
- a **compact blob** (15–700 px², fill > 0.4, aspect 0.45–2.2).

Grayscale frame differencing fails here because the ball's luminance matches the pale back wall.
This step gives about 7 clutter blobs per frame (skin, clothing, wall items). They are removed by:
1. **Static-cell suppression**: 12 px cells that fire in more than 8 % of frames.
2. **A 3-D gate from apparent size.** The camera K, R and t are recovered from the court homography (plane-based calibration: f ≈ 1040 px, camera ≈ 3.9 m high, 5.2 m behind the baseline; the projected net top lands within about 10 px of the real tape). A 21 cm ball's pixel diameter gives its depth, so its pixel position gives a 3-D point. Candidates outside the court volume are rejected, which removes the adjacent court's ball. From this camera, image position alone cannot separate that court's far half from our near half.
3. **Tracklet linking** (constant-velocity prediction, 50 px/frame gate). Only tracklets that are long, moving (< 30 % still steps) and free of teleports are kept, non-overlapping in time.

**Size bias:** the yellow-only mask under-measures a yellow/blue ball, which put rolling balls about 0.8 m *below* the floor. A factor of 1.22, calibrated on frames where the ball rolls on the floor, fixes it.

Result on 40 s: the ball is tracked in 55 % of frames, with no jumps over 40 px. Most untracked time is dead ball between rallies, plus occlusion behind players.

### 8.2 Contact events
Contacts are found with a Savitzky–Golay smoothed track, then the velocity change across ±4 frames, with gravity removed as the segment median. Peaks over 5 px/frame are kept when the direction turns by more than 25° or the speed changes by more than 40 %.

### 8.3 Who touched it: top-down RTMPose-m on YOLO boxes
At each event (±2 frames) players are detected (two-scale tiles) and RTMPose-m (ONNX, about 30 ms/person on CPU) runs on each box. The **contact distance** is the minimum distance from the ball to:
- the hand (wrist + 0.35 × forearm extension),
- the forearm segment (elbow → wrist, for digs and passes),
- the head (nose).

That distance is normalised by box height, and a contact needs < 0.22. Team and track ID come from the toucher's ankle point through the homography, matched to the tracker's court positions. A **block** is a contact within 1.2 m of the net with both wrists above the head (not counted toward the 3).
An impulse with nobody in reach whose size is floor-consistent is a **bounce**.

### 8.4 Ballistic 3-D flights (monocular, gravity-constrained)
Between contacts, X(t) = X0 + V0·t + ½·g·t² with g = 9.81 m/s². Because gravity is known in metres, the 6 unknowns are recoverable from **one** camera by least-squares reprojection, initialised from the size-depth estimate and bounded to the hall.
- Good fits (RMS 0.6–1.2 px) give a far serve from y = 19 m to 8 m with a 3.9 m apex, near-side passes, and a set crossing the net.
- Short or degenerate flights are flagged invalid (RMS > 4 px or below the floor).
These give an independent **net-crossing** signal: a valid flight with y crossing 9 m.

### 8.5 Dead-ball detection and rally logic
- **Dead ball:** low (z < 0.35 m), slow, floor-consistent size (the ball's lowest pixel mapped through H must predict its apparent size within 0.7–1.4×), for **≥ 0.4 s**. This threshold matters: a ball dropping to a player passes low heights for about 16 frames, while a rolling ball stays low for more than 60.
- A **rally** ends on a dead ball, a floor-consistent bounce, or more than 4 s without contacts.
- A contact made when the ball was already dead counts as **dead-ball handling** (picking up or chasing the ball), not a touch.
- **Counting:** the count resets when possession changes team. It is marked *uncertain* when the ball was untracked for more than 0.35 s between touches, or a valid flight says the ball crossed and came back unseen. Fouls are raised only on certain counts:
  - `FOUR_TOUCHES`: more than 3 touches by one team; blocks are excluded.
  - `DOUBLE_CONTACT`: the same player twice in a row, unless the first was a block.

### 8.6 Results on t = 40–80 s (checked frame by frame against the video)
| Event | System | Ground truth (by eye) |
|---|---|---|
| Rally 1 | far serve (unattributed: server occluded or too far), near touches 1-2-3 (#3, #5, #6) | near pass, set and attack ✔ |
| 47.3 s, 48.3 s | *dead-ball handling* | ball rolling on the floor, player picks it up ✔ (an earlier version wrongly flagged FOUR_TOUCHES + DOUBLE_CONTACT here, which motivated 8.5) |
| Rally 3 | near 62.8 s (pre-serve handling ✘), near serve 63.2 s, far #24 touch 65.0 s, near #35 touch 65.8 s, floor 66.7 s | serve → far return → near touch → ball down ✔ (except the pre-serve toss) |
| Bounces | 58.8, 59.4, 66.7, 67.3 s floor | ball on the floor ✔ |
| Fouls | none flagged | none occurred ✔ (detection of real fouls is still **untested**) |

**Known failure modes:**
- Pre-serve ball handling counts as a touch. Fix: a rally starts with a serve, i.e. a contact from behind the baseline.
- A loose-ball chase after a bounce appears as a one-touch rally.
- Track IDs at a contact can be wrong when two players stand within about 1 m.
- Far-court ball depth from size is weak (±1 px on a 10 px ball ≈ ±10 % depth). A second camera would triangulate this properly (stereo + DLT as in your A2).
- Touches hidden behind players are missed. The 0.35 s gap rule then makes the count uncertain rather than wrong.

**To validate fouls:** record a short staged clip with deliberate 4-touch and double-contact plays. Then hand-label every contact in 2–3 minutes of footage and report precision and recall of touch attribution plus the foul confusion matrix.

### 8.7 Files (stage 2)
- `ball/ball_cands.py`: candidate detector
- `ball_track.py`: tracking, outputs `ball_track.csv`
- `camera_model.py`: K, R and t from H, plus size → 3-D
- `ball_events.py`: impulses, outputs `ball_events.csv`
- `touches.py`: RTMPose attribution, outputs `touches.csv`
- `flights.py`: ballistic fits, outputs `flights.csv`
- `rules.py`: rallies and fouls, outputs `rallies.csv`
- `render_overlay.py`: outputs `volleyball_touches_demo.mp4`

Run order:
```
pip install rtmlib onnxruntime scipy
# unzip rtmpose-m_simcc-body7_pt-body7_420e-256x192 ONNX into rtmpose_m/
python ball_track.py --video IMG_4838.MOV --start 40 --dur 40     # ~5 min CPU, cached
python ball_events.py && python touches.py && python flights.py && python rules.py
python render_overlay.py
```
