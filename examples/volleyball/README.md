# Volleyball tactical-view MVP

This local CLI turns a fixed-camera volleyball recording into tracked player boxes,
metric court positions, a synchronized top-down view, and structured CSV data. Manual
calibration is deliberately the reliable default; no cloud API, web server, ball model,
or automatic court model is involved. The input may be an existing local file or one
public YouTube video downloaded to a persistent local cache before analysis starts.

## Current scope

The MVP:

- collects or reloads any four or more visible named court landmarks (6+ recommended);
- maps player bounding-box bottom centers into a 9 m × 18 m court;
- detects only the COCO person class using an official auto-resolved Ultralytics model
  or a trusted local compatible checkpoint;
- assigns temporary ByteTrack IDs;
- retains positions inside configurable side/baseline analysis margins;
- selects at most six temporally stable active tracks on each canonical court side;
- writes annotated, tactical, and side-by-side MP4 files plus `player_tracks.csv`.

It does **not** identify teams or people, detect the ball, segment rallies, recognize
actions, estimate 3D positions, preserve audio, compensate for a moving camera, or run
a web service. Tracker IDs are temporary and may change after long occlusions.

## Architecture

```text
Video
  |
  v
Player Detector
  |
  v
ByteTrack
  |
  v
Raw tracks
  |
  v
Court homography / analysis-area filter
  |
  v
Temporal active-player selector (max 6 FAR + max 6 NEAR)
  |
  +------------------------+
  |                        |
  v                        v
Annotated Video        Bottom-Center Point
                           |
                           v
                    Court Homography
                           |
                           v
                      (x_m, y_m)
                           |
                +----------+----------+
                |                     |
                v                     v
          Tracking CSV          Tactical View
```

The generic `sports.common.view.ViewTransformer` owns the homography. Volleyball
dimensions live in `sports.configs.volleyball`, while rendering lives in
`sports.annotators.volleyball`. The example detector is behind a small `PlayerDetector`
interface so a differently licensed local detector can replace Ultralytics without
changing geometry, tracking export, or rendering.

`CameraView.ENDLINE` and `CameraView.SIDELINE` affect only calibration guidance and
the tactical canvas. The homography always produces the same canonical 9 m × 18 m
coordinates, so CSV coordinates remain comparable across camera positions.

`examples.volleyball.source` is a narrow input adapter. It resolves a local path or
downloads one YouTube video, then hands the same ordinary local `Path` to calibration
and the existing processing pipeline. No YouTube-specific behavior exists downstream.

The volleyball extra currently constrains Supervision below 0.30 because its bundled
ByteTrack API is deprecated for removal in 0.30. Replacing that adapter with the new
standalone tracker package is a future compatibility task; court geometry and rendering
do not depend on the tracker implementation.

## Installation

Use Python 3.8 or newer in a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[volleyball,tests]'
```

Use Python 3.10 or newer for YouTube input so pip can install a current yt-dlp
release. Local-file processing retains the repository's Python 3.8 minimum; on older
Python versions pip may select an older yt-dlp release that no longer tracks YouTube
changes reliably.

The default `--player-model auto` uses the installed Ultralytics version and hardware.
Ultralytics 8.4.0 or newer selects YOLO26: `yolo26s.pt` on CUDA or MPS and
`yolo26n.pt` on CPU. Older compatible installations fall back to the corresponding
mature YOLO11 model. Official `yolo26n.pt`, `yolo26s.pt`, and `yolo26m.pt` detection
weights may be requested explicitly; automatic selection never chooses medium, large,
or extra-large. Weights are stored under `models/ultralytics/` (ignored by Git).

Prepare and perform an actual inference smoke test before processing video:

```bash
python examples/volleyball/setup_models.py
```

This reports Torch/Ultralytics versions, available CUDA/MPS hardware, the selected
device and exact model path, then runs inference on a synthetic image and verifies
that the model is a detection model with COCO class 0 named `person`. Use
`--player-model yolo26n.pt`, `yolo26s.pt`, or `yolo26m.pt` for an explicit official
choice. A custom checkpoint must be an existing local path. Only load custom
PyTorch-compatible model files from a trusted source.

Inference resolution and the visible confidence threshold are controlled explicitly:

```text
--player-imgsz 640|960|1280   default: 640
--player-conf FLOAT           default: 0.25
```

Higher resolution can help distant players but increases latency and memory use.
The legacy aliases `--image-size` and `--confidence` remain accepted.

### Reproducible detector benchmark

Compare official models on the same sequential frames, excluding model load and one
warm-up prediction from timing:

```bash
python examples/volleyball/benchmark_player_models.py \
  --source-video /path/to/match.mov \
  --calibration-file output/calibration.json \
  --device mps \
  --start-time 10 \
  --frames 300 \
  --models yolo26n.pt yolo26s.pt yolo26m.pt \
  --player-imgsz 640 960 \
  --output-csv output/player-model-benchmark.csv
```

The report includes mean/median inference latency, FPS, raw person counts,
confidence, geometrically plausible analysis-area counts, and frames below/above 12
plausible people. These are diagnostics rather than an accuracy leaderboard: detection
count alone cannot establish which model found the correct players. The script uses
official Ultralytics acquisition and never downloads `l`/`x` weights.

Ultralytics software and its default trained weights are currently offered under
AGPL-3.0 or a separate Enterprise license. That is distinct from this repository's MIT
code license. Review [the license notes](../../docs/volleyball-third-party.md) before
distribution or commercial use.

## Input sources

Exactly one of `--source-video` and `--youtube-url` is required.

### Local file

Local paths are expanded (including `~`), resolved, validated, and used in place; they
are not copied into the download cache.

```bash
python examples/volleyball/main.py \
  --source-video match.mp4 \
  --output-dir output \
  --player-model auto \
  --max-frames 500
```

### YouTube

YouTube input uses the `yt-dlp` Python API, disables playlists, and defaults to the best
video up to 1080p plus audio. The maximum can be changed with
`--youtube-max-height`. The downloaded media is resolved before calibration begins.

```bash
python examples/volleyball/main.py \
  --youtube-url "https://www.youtube.com/watch?v=VIDEO_ID" \
  --download-dir data/youtube \
  --output-dir output \
  --player-model auto \
  --max-frames 500
```

Files remain visible and reusable after processing:

```text
data/youtube/VIDEO_ID/
├── video.mp4
└── metadata.json
```

The actual media extension may differ when MP4 is unavailable. A valid cached
`video.*` is reused automatically; pass `--redownload` to replace only that video's
download files. Titles are stored only in the sanitized metadata and are never used as
paths.

FFmpeg is strongly recommended and must be available as the `ffmpeg` executable on
`PATH` to merge separate high-quality video and audio streams. Without it, the adapter
tries a compatible single-file stream and reports an actionable error if none can be
used. It never installs system packages or loads browser cookies.

Only download and process media when you have permission to do so and when doing so
complies with the source platform's terms and applicable law. Automated tests mock
yt-dlp and never access YouTube.

## Manual calibration

Choose a frame with as many visible line/sideline intersections as practical. The UI
walks through the ten canonical landmarks in court order, shows the landmark name,
metric coordinate, semantic description and a highlighted reference diagram. Never
guess an off-screen point: press **S** to skip it.

Select the camera position explicitly when creating a calibration:

```text
--camera-view endline   camera primarily behind a baseline (portrait reference)
--camera-view sideline  camera primarily beside a long sideline (landscape reference)
```

The backward-compatible default for a new calibration is `endline`. In endline view,
the reference shows the far baseline at the top and camera/near baseline at the bottom.
In sideline view it is rotated clockwise, with the camera side at the bottom. Landmark
names and metric coordinates do not rotate: `far_left_corner` is always `(0, 0)`, the
net is always `y=9`, and `near_right_corner` is always `(9, 18)`.

For a camera beside the long sideline, use:

```bash
python examples/volleyball/main.py \
  --source-video match.mp4 \
  --output-dir output-sideline \
  --camera-view sideline \
  --player-model auto \
  --max-frames 500
```

Controls:

- click — assign the current landmark;
- **S** — skip an invisible landmark;
- **U** — undo the previous click or skip;
- **R** — restart;
- **Enter** — fit once at least four non-degenerate points are present;
- **Escape** — cancel.

Six or more well-distributed landmarks are strongly recommended. The solver uses
RANSAC with a default `0.15 m` court-space threshold (configurable with
`--calibration-ransac-threshold`). After fitting, it projects both baselines, both
sidelines, both attack lines, the net line and every landmark back into the camera
frame—even where the inferred court lies off screen. Review the inlier count, rejected
points, court-space and image-space residuals, then press **Enter** to accept or **R**
to recalibrate.

The displayed good/warning/poor bands use median inlier court error below 0.10 m,
0.10–0.25 m, and above 0.25 m. These are diagnostics, not accuracy guarantees.
Concentrated landmarks also trigger an extrapolation warning. Calibration assumes the
camera remains fixed for the whole processed segment.

Create calibration and process a short development sample:

```bash
python examples/volleyball/main.py \
  --source-video match.mp4 \
  --output-dir output \
  --camera-view endline \
  --calibration manual \
  --calibration-file output/calibration.json \
  --player-model auto \
  --device auto \
  --max-frames 500 \
  --show-calibration \
  --show-tactical-view
```

On later runs, the existing file passed to `--calibration-file` is loaded without a
clicking step. Both legacy v1 four-corner files and v2 arbitrary-landmark files are
accepted; files without `camera_view` default to `endline`, and outputs are saved as
v2. A supplied `--camera-view` must match the saved value or the command stops with a
clear error. To process a time slice:

```bash
python examples/volleyball/main.py \
  --source-video match.mp4 \
  --output-dir output-slice \
  --calibration-file output/calibration.json \
  --player-model auto \
  --device mps \
  --start-time 60 \
  --end-time 90 \
  --max-frames 500
```

For a complete video, omit `--max-frames`, `--start-time`, and `--end-time`:

```bash
python examples/volleyball/main.py \
  --source-video match.mp4 \
  --output-dir output-full \
  --calibration-file output/calibration.json \
  --player-model auto \
  --device cuda:0
```

Use `--show-detections` / `--hide-detections` and `--show-track-ids` /
`--hide-track-ids` to control source-video overlays. `--show-tactical-view` opens a
live combined preview; press `q` to stop. Processing and output generation do not
otherwise require a display when a calibration file already exists.

The tactical and combined outputs show selected active players by default. Add
`--show-all-tracks` to draw rejected but geometrically eligible tracks as smaller gray
markers. Active players remain the larger colored markers. Side stability can be tuned
with `--net-hysteresis` (default `0.5 m`) and `--side-switch-frames` (default `5`).

## Outputs

Every output directory contains:

- `calibration.json` — v2 named correspondences and recomputed fit diagnostics;
- `player_tracks.csv` — every raw tracked person with frame, timestamp, temporary ID,
  confidence, box, pixel foot point, metric court point, court/analysis-area flags,
  canonical `side`, `active_player`, `active_rank`, and `active_score`;
- `annotated.mp4` — camera view with calibrated lines, boxes, foot points, and IDs;
- `tactical.mp4` — full tactical court with optional trajectory tails;
- `combined.mp4` — synchronized annotated and tactical frames side by side.

All generated videos use source FPS and frame order. Audio is not copied in this MVP.
The default tactical canvas is 450 × 900 for endline view and 900 × 450 for sideline
view. `--tactical-width` and `--tactical-height` still override these values. The
combined output scales and letterboxes the tactical panel without changing its aspect
ratio or allowing a landscape sideline panel to dominate the source frame.

## Coordinates and filtering

Coordinates are metres on the floor plane:

```text
far-left (0,0) -------- (9,0) far-right
       |       y=6        |
       |       y=9 net    |
       |       y=12       |
near-left (0,18) ------ (9,18) near-right
```

The default analysis area extends 3 m past each sideline and 5 m past each baseline,
so `x` may be `-3..12` and `y` may be `-5..23`. Adjust with `--side-margin` and
`--baseline-margin`. Servers behind either baseline and pursuit players beyond a
sideline remain eligible. Raw tracking makes no assumption that exactly 12 people are
visible.

The active-player layer uses canonical court coordinates, never screen position:
`FAR` is primarily `y < 9 m`, and `NEAR` is primarily `y > 9 m`. A per-track hysteresis
state prevents noisy foot anchors near the net from flipping side every frame. Within
each side, at most six current tracks are selected using named weights for detection
confidence, track age, recent active membership, selection history, and proximity to
the playable court. Missing players are not invented; recent membership is remembered
briefly so a returning established track can displace a transient official. This is a
domain plausibility filter, not team or player identity classification.

## Hardware and known limitations

- CPU works but is usually slow; use CUDA or MPS only if supported by the installed
  PyTorch/Ultralytics build.
- Generic person weights can miss crouched, partially occluded, or distant players and
  can detect officials or spectators. Geometry removes only people whose projected foot
  point falls outside the configured area; the max-six selector cannot determine which
  six are correct when several plausible people occupy one side.
- If `yolo26s @ 960` or `yolo26m @ 960` still misses real players, a small
  volleyball-specific fine-tune is preferable to continuing through larger generic COCO
  weights. Model size alone does not close a domain gap.
- A single planar homography is valid for floor contact points, not airborne bodies or
  balls. The bottom-center box anchor is only an approximation of foot contact.
- Calibration becomes invalid after camera pan, zoom, stabilization crop, or relocation.
- Four-point fits have no redundancy for outlier rejection; use 6+ distributed points
  whenever possible. Projection into an unseen end remains an extrapolation and should
  be checked carefully in the full-court preview.
- MP4 encoding depends on the local OpenCV `mp4v` codec build. Outputs are video-only.

## Roadmap

1. Validate and benchmark the Phase 1 geometry/tracking pipeline on representative
   fixed-camera footage.
2. Add team classification using upper-body appearance and the existing team-classifier
   architecture where licensing/offline behavior permits.
3. Add an optional 10-keypoint volleyball court model while retaining manual fallback.
4. Evaluate ReID-assisted persistent identities.
5. Add volleyball-specific ball tracking in the same metric coordinate system.
6. Add simple rally segmentation, then geometry-assisted action recognition.
7. Build tactical metrics only after the underlying tracks are measured and reliable.
