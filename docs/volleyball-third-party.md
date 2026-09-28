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
| yt-dlp | optional public YouTube video download | Unlicense; packaged distributions can include separately licensed components |
| FFmpeg | optional stream merging/container handling | License depends on the installed build and configuration |

Ultralytics is optional in `setup.py` because its AGPL-3.0 terms are materially
different from this repository's MIT license. Its official licensing page states that
Ultralytics software and default trained models use AGPL-3.0 unless covered by separate
commercial terms: <https://www.ultralytics.com/license>.

The volleyball extra uses yt-dlp through its Python API and does not bundle a yt-dlp
binary. The upstream project is released under the Unlicense and documents separately
licensed bundled components in its own `THIRD_PARTY_LICENSES.txt`:
<https://github.com/yt-dlp/yt-dlp>. FFmpeg is not bundled or installed by this project.
Users must review the license of their chosen FFmpeg build independently.

Downloading software does not grant rights to downloaded media. Users are responsible
for ensuring that they have permission to download and process each source video and
that their use complies with the platform terms and applicable law.

## Ultralytics software

The optional `ultralytics>=8.4.0` Python dependency supplies the official Python API
used for model loading and inference. Ultralytics documents its software under
AGPL-3.0 or separate Enterprise terms. It is not relicensed by this repository's MIT
license. See <https://www.ultralytics.com/license>.

## Ultralytics model weights

No weights are committed or redistributed by this repository. In `auto` mode the
application asks the official Ultralytics Python API for one allow-listed COCO
detection checkpoint (`yolo26n.pt`, `yolo26s.pt`, or the YOLO11 compatibility
equivalent) and stores it in the ignored `models/ultralytics/` directory. It does not
construct arbitrary model URLs or use third-party mirrors. Ultralytics documents
YOLO26 code and models under AGPL-3.0 or Enterprise terms:
<https://docs.ultralytics.com/models/yolo26/>.

Automatic acquisition does not make the weights MIT-licensed. Users must verify:

- the license of the exact checkpoint;
- whether redistribution and commercial use are allowed;
- the provenance and integrity of the file.

Model files are executable-data attack surfaces in parts of the Python ML ecosystem.
Only the fixed official names are auto-downloadable. Custom checkpoints must already
exist locally and should only be loaded from trusted sources. A repository code license
does not imply that weights trained or distributed with it have the same license.

## Future custom model weights

Future volleyball-specific court, ball, ReID, OCR, or action models require their own
provenance and license review. None are downloaded by the current setup command.

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
