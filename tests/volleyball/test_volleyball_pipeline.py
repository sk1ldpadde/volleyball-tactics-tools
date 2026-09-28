import csv
import json
from pathlib import Path

import cv2
import numpy as np
import supervision as sv
import pytest

from examples.volleyball.pipeline import PipelineOptions, run_pipeline
from sports.common.calibration import CourtCalibration
from sports.configs.volleyball import CameraView


class _FixedDetector:
    def detect(self, _frame):
        return sv.Detections(
            xyxy=np.asarray([[130.0, 70.0, 190.0, 160.0]], dtype=np.float32),
            confidence=np.asarray([0.95], dtype=np.float32),
            class_id=np.asarray([0], dtype=int),
        )


def _write_video(path: Path, frame_count: int = 6, fps: float = 12.0) -> None:
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (320, 240))
    assert writer.isOpened()
    try:
        for index in range(frame_count):
            frame = np.full((240, 320, 3), 25 + index, dtype=np.uint8)
            writer.write(frame)
    finally:
        writer.release()


def _probe(path: Path):
    capture = cv2.VideoCapture(str(path))
    assert capture.isOpened()
    try:
        return (
            int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
            float(capture.get(cv2.CAP_PROP_FPS)),
            int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )
    finally:
        capture.release()


@pytest.mark.parametrize(
    ("camera_view", "tactical_resolution"),
    [
        (CameraView.ENDLINE, (240, 360)),
        (CameraView.SIDELINE, (360, 240)),
    ],
)
def test_synthetic_video_runs_through_complete_pipeline(
    tmp_path, camera_view, tactical_resolution,
) -> None:
    source = tmp_path / "source.mp4"
    output = tmp_path / "output"
    _write_video(source)
    calibration = CourtCalibration(
        landmarks={
            "far_left_corner": (40.0, 30.0),
            "far_right_corner": (280.0, 30.0),
            "near_right_corner": (280.0, 210.0),
            "near_left_corner": (40.0, 210.0),
        },
        camera_view=camera_view,
    )

    written = run_pipeline(
        source_video=source,
        output_dir=output,
        detector=_FixedDetector(),
        calibration=calibration,
        config=calibration.configuration(),
        options=PipelineOptions(
            max_frames=5,
            tactical_resolution=tactical_resolution,
            tactical_padding=20,
            camera_view=camera_view,
        ),
    )

    assert written == 5
    for filename in ("annotated.mp4", "tactical.mp4", "combined.mp4"):
        frame_count, fps, width, height = _probe(output / filename)
        assert frame_count == 5
        assert fps == 12.0
        if filename == "tactical.mp4":
            assert (width, height) == tactical_resolution
    with (output / "player_tracks.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert rows
    assert all(np.isfinite(float(row["court_x_m"])) for row in rows)
    assert all(np.isfinite(float(row["court_y_m"])) for row in rows)
    assert {int(row["frame_index"]) for row in rows}.issubset(set(range(5)))
    assert all(row["side"] in {"far", "near", "unknown"} for row in rows)
    assert all(row["active_player"] in {"True", "False"} for row in rows)
    assert all(row["logical_player_track_id"] for row in rows)
    assert all(row["raw_track_id"] == row["track_id"] for row in rows)
    assert all(row["visible"] == "True" for row in rows)
    assert all(row["position_observed"] == "True" for row in rows)
    assert all(row["visibility_state"] == "visible" for row in rows)
    active_by_frame_side = {}
    for row in rows:
        if row["active_player"] == "True":
            key = (row["frame_index"], row["side"])
            active_by_frame_side[key] = active_by_frame_side.get(key, 0) + 1
    assert all(count <= 6 for count in active_by_frame_side.values())


def test_pipeline_records_raw_id_reconnection_without_changing_logical_id(
    tmp_path, monkeypatch,
) -> None:
    class _SwitchingTracker:
        def __init__(self, **_kwargs):
            self.calls = 0

        def update_with_detections(self, detections):
            raw_id = 17 if self.calls < 3 else 42
            self.calls += 1
            return sv.Detections(
                xyxy=detections.xyxy,
                confidence=detections.confidence,
                class_id=detections.class_id,
                tracker_id=np.asarray([raw_id], dtype=int),
            )

    monkeypatch.setattr("examples.volleyball.pipeline.sv.ByteTrack", _SwitchingTracker)
    source = tmp_path / "source.mp4"
    output = tmp_path / "output"
    _write_video(source, frame_count=6)
    calibration = CourtCalibration(landmarks={
        "far_left_corner": (40.0, 30.0),
        "far_right_corner": (280.0, 30.0),
        "near_right_corner": (280.0, 210.0),
        "near_left_corner": (40.0, 210.0),
    })
    run_pipeline(
        source_video=source,
        output_dir=output,
        detector=_FixedDetector(),
        calibration=calibration,
        config=calibration.configuration(),
        options=PipelineOptions(max_frames=5, tactical_resolution=(240, 360)),
    )
    with (output / "player_tracks.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert {int(row["raw_track_id"]) for row in rows} == {17, 42}
    assert len({int(row["logical_player_track_id"]) for row in rows}) == 1
    reconnect_rows = [row for row in rows if row["reconnected"] == "True"]
    assert len(reconnect_rows) == 1
    assert reconnect_rows[0]["reconnection_from_raw_track_id"] == "17"
    events = [
        json.loads(line)
        for line in (output / "reconnection_events.jsonl").read_text().splitlines()
    ]
    assert len(events) == 1
    assert events[0]["old_raw_track_id"] == 17
    assert events[0]["new_raw_track_id"] == 42
