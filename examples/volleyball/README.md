# Volleyball tactical-view MVP

This local CLI turns a fixed-camera volleyball recording into tracked player boxes,
metric court positions, a synchronized top-down view, and structured CSV data. Manual
calibration is deliberately the reliable default; no cloud API, web server, ball model,
or automatic court model is involved.

## Current scope

The MVP:

- collects or reloads four ordered outer-court corners;
- maps player bounding-box bottom centers into a 9 m × 18 m court;
- detects only a configurable person class from local Ultralytics-compatible weights;
- assigns temporary ByteTrack IDs;
- retains positions inside configurable side/baseline analysis margins;
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

The command requires a local Ultralytics-compatible detection checkpoint. It will not
accept a bare model name such as `yolo11n.pt`, because Ultralytics can download missing
named weights automatically. Obtain weights under terms appropriate for your use,
store them locally, and pass their exact path. Only load model files from a trusted
source: PyTorch-compatible checkpoints can be unsafe when untrusted.

Ultralytics software and its default trained weights are currently offered under
AGPL-3.0 or a separate Enterprise license. That is distinct from this repository's MIT
code license. Review [the license notes](../../docs/volleyball-third-party.md) before
distribution or commercial use.

## Manual calibration

Choose a frame where all four outer corners are visible. Click exactly:

1. far-left baseline corner;
2. far-right baseline corner;
3. near-right baseline corner;
4. near-left baseline corner.

Each click is labelled immediately. Once four points exist, the boundary, attack
lines, and net/center line are projected back into the image. Press **Enter** to
confirm, **R** to restart, or **Escape** to cancel. Calibration assumes the camera is
fixed for the whole processed segment.

Create calibration and process a short development sample:

```bash
python examples/volleyball/main.py \
  --source-video match.mp4 \
  --output-dir output \
  --calibration manual \
  --calibration-file output/calibration.json \
  --player-model /absolute/path/to/trusted-person-model.pt \
  --device cpu \
  --max-frames 500 \
  --show-calibration \
  --show-tactical-view
```

On later runs, the existing file passed to `--calibration-file` is loaded without a
clicking step. To process a time slice:

```bash
python examples/volleyball/main.py \
  --source-video match.mp4 \
  --output-dir output-slice \
  --calibration-file output/calibration.json \
  --player-model /absolute/path/to/trusted-person-model.pt \
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
  --player-model /absolute/path/to/trusted-person-model.pt \
  --device cuda:0
```

Use `--show-detections` / `--hide-detections` and `--show-track-ids` /
`--hide-track-ids` to control source-video overlays. `--show-tactical-view` opens a
live combined preview; press `q` to stop. Processing and output generation do not
otherwise require a display when a calibration file already exists.

## Outputs

Every output directory contains:

- `calibration.json` — self-contained image-to-court calibration;
- `player_tracks.csv` — frame, timestamp, temporary ID, confidence, box, pixel foot
  point, metric court point, and court/analysis-area membership;
- `annotated.mp4` — camera view with calibrated lines, boxes, foot points, and IDs;
- `tactical.mp4` — full tactical court with optional trajectory tails;
- `combined.mp4` — synchronized annotated and tactical frames side by side.

All generated videos use source FPS and frame order. Audio is not copied in this MVP.

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
`--baseline-margin`. No assumption is made that exactly 12 people are visible.

## Hardware and known limitations

- CPU works but is usually slow; use CUDA or MPS only if supported by the installed
  PyTorch/Ultralytics build.
- Generic person weights can miss crouched, partially occluded, or distant players and
  can detect officials or spectators. Geometry removes only people whose projected foot
  point falls outside the configured area.
- A single planar homography is valid for floor contact points, not airborne bodies or
  balls. The bottom-center box anchor is only an approximation of foot contact.
- Calibration becomes invalid after camera pan, zoom, stabilization crop, or relocation.
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
