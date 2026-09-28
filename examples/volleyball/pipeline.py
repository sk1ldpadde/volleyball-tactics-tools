"""Streaming volleyball player tracking and tactical-view pipeline."""

from collections import defaultdict, deque
import csv
from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Deque, Dict, Optional, Tuple

import cv2
import numpy as np
import supervision as sv

from sports.annotators.volleyball import draw_player_tracks_on_volleyball_court
from sports.common.calibration import CourtCalibration
from sports.common.view import get_bottom_center_points
from sports.configs.volleyball import Point, VolleyballCourtConfiguration

try:
    from .calibration import draw_projected_court
    from .detection import PlayerDetector
except ImportError:  # Support direct execution from examples/volleyball.
    from calibration import draw_projected_court
    from detection import PlayerDetector


LOGGER = logging.getLogger(__name__)
CSV_COLUMNS = (
    "frame_index",
    "timestamp_s",
    "track_id",
    "confidence",
    "bbox_x1",
    "bbox_y1",
    "bbox_x2",
    "bbox_y2",
    "foot_x_px",
    "foot_y_px",
    "court_x_m",
    "court_y_m",
    "inside_court",
    "inside_analysis_area",
)


@dataclass(frozen=True)
class PipelineOptions:
    start_time: float = 0.0
    end_time: Optional[float] = None
    max_frames: Optional[int] = None
    tactical_resolution: Tuple[int, int] = (600, 900)
    tactical_padding: int = 40
    trajectory_length: int = 30
    show_detections: bool = True
    show_track_ids: bool = True
    preview_tactical_view: bool = False


def _video_writer(path: Path, fps: float, size: Tuple[int, int]) -> cv2.VideoWriter:
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    if not writer.isOpened():
        writer.release()
        raise RuntimeError(f"Could not open video writer for {path}")
    return writer


def _annotate_source(
    frame: np.ndarray,
    detections: sv.Detections,
    calibration: CourtCalibration,
    config: VolleyballCourtConfiguration,
    show_detections: bool,
    show_track_ids: bool,
) -> np.ndarray:
    annotated = draw_projected_court(frame, calibration, config)
    tracker_ids = detections.tracker_id
    for index, box in enumerate(detections.xyxy):
        x1, y1, x2, y2 = np.rint(box).astype(int)
        if show_detections:
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 120, 20), 2)
            cv2.circle(
                annotated, (int(round((x1 + x2) / 2)), y2), 4,
                (0, 255, 255), thickness=-1)
        if show_track_ids and tracker_ids is not None:
            cv2.putText(
                annotated, f"ID {int(tracker_ids[index])}", (x1, max(20, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
    return annotated


def _combine_frames(source: np.ndarray, tactical: np.ndarray) -> np.ndarray:
    source_height = source.shape[0]
    tactical_width = max(1, int(round(tactical.shape[1] * source_height / tactical.shape[0])))
    resized_tactical = cv2.resize(
        tactical, (tactical_width, source_height), interpolation=cv2.INTER_AREA)
    return np.hstack((source, resized_tactical))


def run_pipeline(
    source_video: Path,
    output_dir: Path,
    detector: PlayerDetector,
    calibration: CourtCalibration,
    config: VolleyballCourtConfiguration,
    options: PipelineOptions,
) -> int:
    """Process a video stream and return the number of written frames."""
    output_dir.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(source_video))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open source video: {source_video}")

    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if fps <= 0 or frame_width <= 0 or frame_height <= 0:
        capture.release()
        raise RuntimeError("Source video has invalid FPS or dimensions.")

    first_frame_index = max(0, int(round(options.start_time * fps)))
    capture.set(cv2.CAP_PROP_POS_FRAMES, first_frame_index)
    tactical_size = options.tactical_resolution
    combined_tactical_width = max(
        1, int(round(tactical_size[0] * frame_height / tactical_size[1])))
    writers = []
    try:
        writers.append(_video_writer(
            output_dir / "annotated.mp4", fps, (frame_width, frame_height)))
        writers.append(_video_writer(
            output_dir / "tactical.mp4", fps, tactical_size))
        writers.append(_video_writer(
            output_dir / "combined.mp4", fps,
            (frame_width + combined_tactical_width, frame_height)))
    except Exception:
        capture.release()
        for writer in writers:
            writer.release()
        raise
    tracker = sv.ByteTrack(frame_rate=max(1, int(round(fps))))
    transformer = calibration.create_transformer()
    trajectories: Dict[int, Deque[Point]] = defaultdict(
        lambda: deque(maxlen=max(1, options.trajectory_length)))
    written_frames = 0

    try:
        with (output_dir / "player_tracks.csv").open(
            "w", newline="", encoding="utf-8") as csv_file:
            csv_writer = csv.DictWriter(csv_file, fieldnames=CSV_COLUMNS)
            csv_writer.writeheader()
            while True:
                ok, frame = capture.read()
                if not ok or frame is None:
                    break
                frame_index = first_frame_index + written_frames
                timestamp_s = frame_index / fps
                if options.end_time is not None and timestamp_s >= options.end_time:
                    break
                if options.max_frames is not None and written_frames >= options.max_frames:
                    break

                detections = detector.detect(frame)
                detected_feet = get_bottom_center_points(detections.xyxy)
                detected_court = transformer.transform_points(detected_feet)
                analysis_mask = np.asarray(
                    [config.is_inside_analysis_area(tuple(point)) for point in detected_court],
                    dtype=bool,
                )
                detections = detections[analysis_mask]
                detections = tracker.update_with_detections(detections)

                foot_points = get_bottom_center_points(detections.xyxy)
                court_points = transformer.transform_points(foot_points)
                current_players: Dict[int, Point] = {}
                tracker_ids = detections.tracker_id
                confidences = detections.confidence
                if tracker_ids is None:
                    tracker_ids = np.empty((0,), dtype=int)

                for index, tracker_id_value in enumerate(tracker_ids):
                    track_id = int(tracker_id_value)
                    court_point = (float(court_points[index, 0]), float(court_points[index, 1]))
                    foot_point = foot_points[index]
                    inside_court = config.is_inside_court(court_point)
                    inside_analysis = config.is_inside_analysis_area(court_point)
                    if not inside_analysis:
                        continue
                    current_players[track_id] = court_point
                    trajectories[track_id].append(court_point)
                    confidence = (
                        float(confidences[index]) if confidences is not None else float("nan"))
                    box = detections.xyxy[index]
                    csv_writer.writerow({
                        "frame_index": frame_index,
                        "timestamp_s": f"{timestamp_s:.6f}",
                        "track_id": track_id,
                        "confidence": f"{confidence:.6f}",
                        "bbox_x1": f"{box[0]:.3f}",
                        "bbox_y1": f"{box[1]:.3f}",
                        "bbox_x2": f"{box[2]:.3f}",
                        "bbox_y2": f"{box[3]:.3f}",
                        "foot_x_px": f"{foot_point[0]:.3f}",
                        "foot_y_px": f"{foot_point[1]:.3f}",
                        "court_x_m": f"{court_point[0]:.6f}",
                        "court_y_m": f"{court_point[1]:.6f}",
                        "inside_court": inside_court,
                        "inside_analysis_area": inside_analysis,
                    })

                annotated = _annotate_source(
                    frame, detections, calibration, config,
                    options.show_detections, options.show_track_ids)
                tactical = draw_player_tracks_on_volleyball_court(
                    config=config,
                    player_points=current_players,
                    trajectories=trajectories,
                    trajectory_length=options.trajectory_length,
                    resolution_wh=tactical_size,
                    padding=options.tactical_padding,
                    include_free_zone=True,
                )
                combined = _combine_frames(annotated, tactical)
                writers[0].write(annotated)
                writers[1].write(tactical)
                writers[2].write(combined)
                written_frames += 1

                if options.preview_tactical_view:
                    cv2.imshow("Volleyball tactical view", combined)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        LOGGER.info("Preview stopped by user after %d frames", written_frames)
                        break
    finally:
        capture.release()
        for writer in writers:
            writer.release()
        if options.preview_tactical_view:
            cv2.destroyAllWindows()

    return written_frames
