# Volleyball MVP third-party and license notes

This file separates the repository code license from optional software, model, and
dataset licenses. It is an engineering inventory, not legal advice.

## Repository code

This repository retains the upstream `roboflow/sports` MIT license in `LICENSE`. The
new volleyball configuration, calibration, renderer, pipeline, tests, and documentation
are original additions under that repository license.

## Runtime software

| Component | Use | License status |
|---|---|---|
| OpenCV | video I/O, GUI, drawing, perspective transforms | Apache-2.0 |
| NumPy | arrays and numeric operations | BSD-3-Clause |
| Supervision | detections and ByteTrack integration | MIT |
| Ultralytics | optional local YOLO detector adapter | AGPL-3.0 or separate Ultralytics commercial terms; verify current terms |

Ultralytics is optional in `setup.py` because its AGPL-3.0 terms are materially
different from this repository's MIT license. Its official licensing page states that
Ultralytics software and default trained models use AGPL-3.0 unless covered by separate
commercial terms: <https://www.ultralytics.com/license>.

## Model weights

No weights are included or downloaded by this repository. The CLI requires an existing
local path and rejects missing paths/bare model names before constructing the model.
Users must separately verify:

- the license of the exact checkpoint;
- whether redistribution and commercial use are allowed;
- the provenance and integrity of the file.

Model files are executable-data attack surfaces in parts of the Python ML ecosystem.
Only load checkpoints from trusted sources. A repository's code license does not imply
that weights trained by or distributed with it have the same license.

## Datasets

No dataset is included, downloaded, or required by the MVP. Dataset licenses govern
training images and annotations separately from both application code and resulting
model weights. Do not redistribute a future volleyball dataset until its source,
consent/privacy posture, attribution requirements, and derivative-model terms have
been reviewed.

## Volleyball-Metrics reference

The architecture of
[`kairsato/Volleyball-Metrics`](https://github.com/kairsato/Volleyball-Metrics) was
reviewed, particularly `Backend/Analysis/CourtDefinition/court.py` and its documented
court/player pipeline. The project identifies its code license as MIT.

This MVP used only high-level concepts from that project: explicit court dimensions,
ordered manual points, saved calibration, planar homography, and deriving court lines
from known geometry. No source code, UI, server, model, checkpoint, or dataset was
copied or adapted. Consequently, no additional source notice is embedded in the code.
The reference's FastAPI/React/Caddy/network stack, automatic court model, ball and
action systems, and checkpoints were intentionally excluded.

## Specifically excluded assets

The MVP does not include the Volleyball-Metrics court weights/dataset, its action or
ball weights/datasets, or the third-party VideoMAE rally checkpoint. In particular,
non-commercial, attribution, share-alike, or unknown terms on any such asset must be
reviewed independently before a later phase adopts it.
