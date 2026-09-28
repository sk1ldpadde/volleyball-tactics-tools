from pathlib import Path

import cv2
import numpy as np
import supervision as sv

from examples.volleyball.benchmark_player_models import benchmark_combination
from sports.common.calibration import CourtCalibration


class _FixedDetector:
    def detect(self, _frame):
        return sv.Detections(
            xyxy=np.asarray([[130.0, 70.0, 190.0, 160.0]], dtype=np.float32),
            confidence=np.asarray([0.8], dtype=np.float32),
            class_id=np.asarray([0], dtype=int),
        )


def _write_video(path: Path) -> None:
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (320, 240))
    assert writer.isOpened()
    try:
        for value in range(4):
            writer.write(np.full((240, 320, 3), value * 20, dtype=np.uint8))
    finally:
        writer.release()


def test_benchmark_uses_requested_frames_and_geometric_counts(tmp_path) -> None:
    video = tmp_path / "source.mp4"
    _write_video(video)
    calibration = CourtCalibration(landmarks={
        "far_left_corner": (40.0, 30.0),
        "far_right_corner": (280.0, 30.0),
        "near_right_corner": (280.0, 210.0),
        "near_left_corner": (40.0, 210.0),
    })
    result = benchmark_combination(
        source_video=video,
        detector=_FixedDetector(),
        model_name="fake.pt",
        image_size=640,
        start_frame=0,
        frames=3,
        calibration=calibration,
        side_margin=3.0,
        baseline_margin=5.0,
    )
    assert result.frames == 3
    assert result.mean_raw_persons == 1
    assert result.mean_plausible_persons == 1
    assert result.frames_under_12 == 3
    assert result.frames_over_12 == 0
