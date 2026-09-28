"""Streaming volleyball player tracking and tactical-view pipeline."""

from collections import defaultdict, deque
import csv
from dataclasses import dataclass, field
import json
import logging
from pathlib import Path
from typing import Deque, Dict, List, Optional, Set, Tuple

import cv2
import numpy as np
import supervision as sv

from sports.annotators.volleyball import draw_player_tracks_on_volleyball_court
from sports.common.calibration import CourtCalibration, ImageOcclusionZone
from sports.common.view import get_bottom_center_points
from sports.configs.volleyball import CameraView, Point, VolleyballCourtConfiguration

try:
    from .calibration import draw_projected_court
    from .detection import PlayerDetector
    from .selection import (
        ActivePlayerSelector,
        ActivePlayerSelectorConfig,
        PlayerSelection,
        ReconnectionConfig,
        SelectionResult,
        TrackCandidate,
        TrackSide,
        VisibilityState,
    )
except ImportError:  # Support direct execution from examples/volleyball.
    from calibration import draw_projected_court
    from detection import PlayerDetector
    from selection import (
        ActivePlayerSelector,
        ActivePlayerSelectorConfig,
        PlayerSelection,
        ReconnectionConfig,
        SelectionResult,
        TrackCandidate,
        TrackSide,
        VisibilityState,
    )


LOGGER = logging.getLogger(__name__)
CSV_COLUMNS = (
    "frame_index",
    "timestamp_s",
    "track_id",
    "raw_track_id",
    "logical_player_track_id",
    "confidence",
    "bbox_x1",
    "bbox_y1",
    "bbox_x2",
    "bbox_y2",
    "foot_x_px",
    "foot_y_px",
    "court_x_m",
    "court_y_m",
    "court_x_m_predicted",
    "court_y_m_predicted",
    "last_known_court_x_m",
    "last_known_court_y_m",
    "inside_court",
    "inside_analysis_area",
    "side",
    "active_player",
    "active_rank",
    "active_score",
    "visible",
    "occluded",
    "position_observed",
    "visibility_state",
    "occlusion_age_frames",
    "occlusion_evidence",
    "reconnected",
    "reconnection_from_raw_track_id",
    "reconnection_score",
    "reconnection_gap_frames",
    "logical_reconnection_count",
)


@dataclass(frozen=True)
class PipelineOptions:
    start_time: float = 0.0
    end_time: Optional[float] = None
    max_frames: Optional[int] = None
    tactical_resolution: Tuple[int, int] = (450, 900)
    tactical_padding: int = 40
    trajectory_length: int = 30
    show_detections: bool = True
    show_track_ids: bool = True
    preview_tactical_view: bool = False
    camera_view: CameraView = CameraView.ENDLINE
    show_all_tracks: bool = False
    net_hysteresis_m: float = 0.5
    side_switch_frames: int = 5
    active_occlusion_grace_frames: Optional[int] = None
    show_occluded_players: bool = False
    show_occlusion_zones: bool = False
    reconnect_max_gap_frames: Optional[int] = None
    reconnect_max_distance_m: float = 3.0
    reconnect_max_speed_mps: float = 8.0
    reconnect_min_score: float = 0.68
    reconnect_ambiguity_margin: float = 0.12


@dataclass
class _PipelineDiagnostics:
    raw_counts: List[int]
    candidate_counts: List[int]
    active_counts: List[int]
    capped_far_frames: int = 0
    capped_near_frames: int = 0
    six_far_frames: int = 0
    six_near_frames: int = 0
    under_six_far_frames: int = 0
    under_six_near_frames: int = 0
    removed_candidate_instances: int = 0
    unique_raw_ids: Set[int] = field(default_factory=set)
    unique_logical_ids: Set[int] = field(default_factory=set)
    temporary_occlusion_events: int = 0
    restored_after_occlusion: int = 0
    active_slots_replaced_during_grace: int = 0
    disappearances_in_zones: int = 0
    completed_occlusion_durations: List[int] = field(default_factory=list)
    _previous_occluded: Set[int] = field(default_factory=set)
    _last_occlusion_age: Dict[int, int] = field(default_factory=dict)

    def observe(
        self,
        raw_ids: Set[int],
        result: SelectionResult,
    ) -> None:
        decisions = result.members
        far_candidates = sum(
            decision.side is TrackSide.FAR and decision.position_observed
            for decision in result.visible.values())
        near_candidates = sum(
            decision.side is TrackSide.NEAR and decision.position_observed
            for decision in result.visible.values())
        active_far = sum(
            decision.side is TrackSide.FAR and decision.active
            for decision in decisions.values())
        active_near = sum(
            decision.side is TrackSide.NEAR and decision.active
            for decision in decisions.values())
        self.raw_counts.append(len(raw_ids))
        self.candidate_counts.append(sum(
            decision.position_observed for decision in result.visible.values()))
        self.active_counts.append(active_far + active_near)
        self.capped_far_frames += int(far_candidates > 6)
        self.capped_near_frames += int(near_candidates > 6)
        self.six_far_frames += int(active_far == 6)
        self.six_near_frames += int(active_near == 6)
        self.under_six_far_frames += int(active_far < 6)
        self.under_six_near_frames += int(active_near < 6)
        visible_active = sum(
            decision.active and decision.position_observed
            for decision in decisions.values())
        self.removed_candidate_instances += (
            self.candidate_counts[-1] - visible_active)
        self.unique_raw_ids.update(raw_ids)
        self.unique_logical_ids.update(
            logical_id for logical_id, decision in decisions.items() if decision.active)
        current_occluded = {
            track_id for track_id, decision in decisions.items()
            if decision.active and decision.occluded
        }
        entered = current_occluded - self._previous_occluded
        self.temporary_occlusion_events += len(entered)
        self.disappearances_in_zones += sum(
            decisions[track_id].occlusion_evidence == "static_zone"
            for track_id in entered)
        restored = {
            track_id for track_id in self._previous_occluded
            if track_id in decisions and decisions[track_id].visible
        }
        self.restored_after_occlusion += len(restored)
        for track_id in current_occluded:
            self._last_occlusion_age[track_id] = decisions[track_id].occlusion_age
        ended = self._previous_occluded - current_occluded
        self.completed_occlusion_durations.extend(
            self._last_occlusion_age.pop(track_id, 0) for track_id in ended)
        self._previous_occluded = current_occluded

    def log(self, frames: int, selector: ActivePlayerSelector) -> None:
        median_raw = float(np.median(self.raw_counts)) if self.raw_counts else 0.0
        median_candidates = (
            float(np.median(self.candidate_counts)) if self.candidate_counts else 0.0)
        median_active = (
            float(np.median(self.active_counts)) if self.active_counts else 0.0)
        LOGGER.info("Player-selection diagnostics (%d frames)", frames)
        LOGGER.info("Median raw tracks/frame: %.1f", median_raw)
        LOGGER.info("Median analysis candidates/frame: %.1f", median_candidates)
        LOGGER.info("Median active players/frame: %.1f", median_active)
        LOGGER.info(
            "Frames capped before selection: FAR=%d, NEAR=%d",
            self.capped_far_frames, self.capped_near_frames)
        LOGGER.info(
            "Frames with exactly 6 active: FAR=%d, NEAR=%d",
            self.six_far_frames, self.six_near_frames)
        LOGGER.info(
            "Frames with fewer than 6 active: FAR=%d, NEAR=%d",
            self.under_six_far_frames, self.under_six_near_frames)
        LOGGER.info(
            "Unique IDs: raw=%d, logical active=%d",
            len(self.unique_raw_ids), len(self.unique_logical_ids))
        LOGGER.info(
            "Candidate instances removed by selector: %d",
            self.removed_candidate_instances)
        durations = [
            *self.completed_occlusion_durations,
            *self._last_occlusion_age.values(),
        ]
        LOGGER.info("Temporary occlusion events: %d", self.temporary_occlusion_events)
        LOGGER.info(
            "Occlusion duration frames: average=%.1f, maximum=%d",
            float(np.mean(durations)) if durations else 0.0,
            max(durations, default=0),
        )
        LOGGER.info(
            "Players restored after occlusion: %d", self.restored_after_occlusion)
        LOGGER.info(
            "Active slots replaced during grace: %d",
            self.active_slots_replaced_during_grace)
        LOGGER.info(
            "Tracks disappearing inside static occlusion zones: %d",
            self.disappearances_in_zones)
        reconnect = selector.diagnostics
        events = selector.events
        raw_counts_by_logical = [
            len(player.previous_raw_track_ids)
            for player in selector.logical_players.values()
            if player.was_ever_active
        ]
        fragmented = sum(count > 1 for count in raw_counts_by_logical)
        fragmentation_ratio = (
            sum(raw_counts_by_logical) / len(raw_counts_by_logical)
            if raw_counts_by_logical else 0.0
        )
        LOGGER.info(
            "Reconnections: accepted=%d, ambiguous=%d, side=%d, distance=%d, "
            "speed=%d, score=%d, referee=%d",
            reconnect.accepted, reconnect.ambiguous_rejected,
            reconnect.rejected_side, reconnect.rejected_distance,
            reconnect.rejected_speed, reconnect.rejected_score,
            reconnect.rejected_referee,
        )
        LOGGER.info(
            "Reconnection gap frames: mean=%.1f, maximum=%d",
            float(np.mean([event.gap_frames for event in events])) if events else 0.0,
            max((event.gap_frames for event in events), default=0),
        )
        LOGGER.info(
            "Reconnection distance meters: mean=%.3f",
            float(np.mean([event.distance_m for event in events])) if events else 0.0,
        )
        LOGGER.info(
            "Identity fragmentation: raw IDs/logical active IDs=%.3f; "
            "logical players with >1 raw ID=%d; max raw IDs/logical player=%d",
            fragmentation_ratio, fragmented, max(raw_counts_by_logical, default=0),
        )


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
    result: SelectionResult,
    show_occluded_players: bool,
    show_occlusion_zones: bool,
) -> np.ndarray:
    annotated = draw_projected_court(frame, calibration, config)
    if show_occlusion_zones:
        for zone in calibration.image_occlusion_zones:
            polygon = np.rint(zone.points).astype(np.int32)
            cv2.polylines(annotated, [polygon], True, (255, 0, 255), 2, cv2.LINE_AA)
            anchor = tuple(polygon[0])
            cv2.putText(
                annotated, f"OCCLUSION: {zone.name}", anchor,
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 0, 255), 2, cv2.LINE_AA)
    tracker_ids = detections.tracker_id
    for index, box in enumerate(detections.xyxy):
        x1, y1, x2, y2 = np.rint(box).astype(int)
        if show_detections:
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 120, 20), 2)
            cv2.circle(
                annotated, (int(round((x1 + x2) / 2)), y2), 4,
                (0, 255, 255), thickness=-1)
        if show_track_ids and tracker_ids is not None:
            track_id = int(tracker_ids[index])
            decision = result.visible.get(track_id)
            state = (
                "ACTIVE" if decision is not None and decision.active
                else "REJECTED" if decision is not None
                else "RAW"
            )
            cv2.putText(
                annotated,
                (
                    f"L{decision.logical_player_track_id} R{track_id} {state}"
                    if decision is not None and decision.logical_player_track_id is not None
                    else f"R{track_id} {state}"
                ),
                (x1, max(20, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
            if decision is not None and decision.reconnected:
                cv2.putText(
                    annotated,
                    f"RECONNECTED R{decision.reconnection_from_raw_track_id} -> R{track_id}",
                    (x1, min(annotated.shape[0] - 10, y2 + 22)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2, cv2.LINE_AA)
    if show_occluded_players:
        for logical_id, decision in result.members.items():
            if not decision.active or decision.position_observed:
                continue
            if decision.last_known_bbox is None:
                continue
            x1, y1, x2, y2 = np.rint(decision.last_known_bbox).astype(int)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 255), 2)
            cv2.putText(
                annotated,
                f"L{logical_id} {decision.visibility.value.upper()} last R"
                f"{decision.last_known_raw_track_id or '-'}",
                (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (0, 255, 255), 2, cv2.LINE_AA)
    return annotated


def _bbox_touches_zone(box: np.ndarray, zone: ImageOcclusionZone) -> bool:
    """Return whether representative points of a bbox intersect a static polygon."""
    x1, y1, x2, y2 = (float(value) for value in box)
    polygon = np.asarray(zone.points, dtype=np.float32)
    samples = (
        (x1, y1), (x2, y1), (x2, y2), (x1, y2),
        ((x1 + x2) / 2.0, (y1 + y2) / 2.0),
        ((x1 + x2) / 2.0, y2),
    )
    return any(cv2.pointPolygonTest(polygon, point, False) >= 0 for point in samples)


def _combined_tactical_size(
    source_size: Tuple[int, int], tactical_size: Tuple[int, int],
) -> Tuple[int, int]:
    """Fit a tactical panel beside the source without distorting or dominating it."""
    source_width, source_height = source_size
    tactical_width, tactical_height = tactical_size
    scale = min(
        source_height / tactical_height,
        (source_width * 0.6) / tactical_width,
    )
    return (
        max(1, int(round(tactical_width * scale))),
        max(1, int(round(tactical_height * scale))),
    )


def _combine_frames(source: np.ndarray, tactical: np.ndarray) -> np.ndarray:
    source_height = source.shape[0]
    tactical_width, tactical_height = _combined_tactical_size(
        (source.shape[1], source_height), (tactical.shape[1], tactical.shape[0]))
    resized_tactical = cv2.resize(
        tactical, (tactical_width, tactical_height), interpolation=cv2.INTER_AREA)
    panel = np.zeros((source_height, tactical_width, 3), dtype=np.uint8)
    top = (source_height - tactical_height) // 2
    panel[top:top + tactical_height] = resized_tactical
    return np.hstack((source, panel))


def run_pipeline(
    source_video: Path,
    output_dir: Path,
    detector: PlayerDetector,
    calibration: CourtCalibration,
    config: VolleyballCourtConfiguration,
    options: PipelineOptions,
) -> int:
    """Process a video stream and return the number of written frames."""
    if CameraView(options.camera_view) is not calibration.camera_view:
        raise ValueError(
            "Pipeline camera view does not match the saved calibration: "
            f"{CameraView(options.camera_view).value} != {calibration.camera_view.value}."
        )
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
    combined_tactical_width, _ = _combined_tactical_size(
        (frame_width, frame_height), tactical_size)
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
    selector = ActivePlayerSelector(
        court=config,
        config=ActivePlayerSelectorConfig(
            frame_rate=fps,
            net_hysteresis_m=options.net_hysteresis_m,
            side_switch_frames=options.side_switch_frames,
            active_occlusion_grace_frames=(
                options.active_occlusion_grace_frames
                if options.active_occlusion_grace_frames is not None
                else max(1, int(round(fps)))
            ),
            reconnection=ReconnectionConfig(
                max_gap_frames=(
                    options.reconnect_max_gap_frames
                    if options.reconnect_max_gap_frames is not None
                    else max(1, int(round(1.5 * fps)))
                ),
                max_distance_m=options.reconnect_max_distance_m,
                max_speed_mps=options.reconnect_max_speed_mps,
                min_score=options.reconnect_min_score,
                ambiguity_margin=options.reconnect_ambiguity_margin,
            ),
        ),
    )
    LOGGER.info(
        "Active-player occlusion grace: %d frames (%.2f s)",
        selector.config.active_occlusion_grace_frames,
        selector.config.active_occlusion_grace_frames / fps,
    )
    LOGGER.info(
        "Logical reconnection: max gap=%d frames (%.2f s), distance=%.2f m, "
        "speed=%.2f m/s, minimum score=%.2f, ambiguity margin=%.2f",
        selector.config.reconnection.max_gap_frames,
        selector.config.reconnection.max_gap_frames / fps,
        selector.config.reconnection.max_distance_m,
        selector.config.reconnection.max_speed_mps,
        selector.config.reconnection.min_score,
        selector.config.reconnection.ambiguity_margin,
    )
    diagnostics = _PipelineDiagnostics([], [], [])
    written_frames = 0

    try:
        with (output_dir / "player_tracks.csv").open(
            "w", newline="", encoding="utf-8") as csv_file, (
                output_dir / "reconnection_events.jsonl"
            ).open("w", encoding="utf-8") as event_file:
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

                detections = tracker.update_with_detections(detector.detect(frame))

                foot_points = get_bottom_center_points(detections.xyxy)
                court_points = transformer.transform_points(foot_points)
                tracker_ids = detections.tracker_id
                confidences = detections.confidence
                if tracker_ids is None:
                    tracker_ids = np.empty((0,), dtype=int)

                raw_records = []
                candidates = []
                for index, tracker_id_value in enumerate(tracker_ids):
                    track_id = int(tracker_id_value)
                    court_point = (float(court_points[index, 0]), float(court_points[index, 1]))
                    foot_point = foot_points[index]
                    inside_court = config.is_inside_court(court_point)
                    inside_analysis = config.is_inside_analysis_area(court_point)
                    confidence = (
                        float(confidences[index]) if confidences is not None else float("nan"))
                    box = detections.xyxy[index]
                    raw_records.append((
                        track_id, court_point, foot_point, confidence, box,
                        inside_court, inside_analysis,
                    ))
                    if inside_analysis and np.isfinite(court_point).all():
                        candidates.append(TrackCandidate(
                            track_id=track_id,
                            court_point=court_point,
                            confidence=confidence if np.isfinite(confidence) else 0.0,
                            inside_court=inside_court,
                            bbox=tuple(float(value) for value in box),
                            in_occlusion_zone=any(
                                _bbox_touches_zone(box, zone)
                                for zone in calibration.image_occlusion_zones),
                        ))

                result = selector.select(
                    candidates,
                    frame_index,
                    occluder_bboxes=tuple(
                        tuple(float(value) for value in box)
                        for box in detections.xyxy),
                )
                for event in result.reconnection_events:
                    event_file.write(json.dumps(event.to_dict(), sort_keys=True) + "\n")
                current_players: Dict[int, Point] = {}
                inactive_players: Dict[int, Point] = {}
                occluded_players: Dict[int, Point] = {}
                for record in raw_records:
                    (track_id, court_point, foot_point, confidence, box,
                     inside_court, inside_analysis) = record
                    decision = result.visible.get(track_id)
                    if decision is not None and decision.active:
                        logical_id = int(decision.logical_player_track_id)
                        current_players[logical_id] = court_point
                        trajectories[logical_id].append(court_point)
                    elif decision is not None:
                        inactive_players[track_id] = court_point
                    csv_writer.writerow({
                        "frame_index": frame_index,
                        "timestamp_s": f"{timestamp_s:.6f}",
                        "track_id": track_id,
                        "raw_track_id": track_id,
                        "logical_player_track_id": (
                            decision.logical_player_track_id
                            if decision is not None
                            and decision.logical_player_track_id is not None else ""),
                        "confidence": f"{confidence:.6f}",
                        "bbox_x1": f"{box[0]:.3f}",
                        "bbox_y1": f"{box[1]:.3f}",
                        "bbox_x2": f"{box[2]:.3f}",
                        "bbox_y2": f"{box[3]:.3f}",
                        "foot_x_px": f"{foot_point[0]:.3f}",
                        "foot_y_px": f"{foot_point[1]:.3f}",
                        "court_x_m": f"{court_point[0]:.6f}",
                        "court_y_m": f"{court_point[1]:.6f}",
                        "court_x_m_predicted": "",
                        "court_y_m_predicted": "",
                        "last_known_court_x_m": f"{court_point[0]:.6f}",
                        "last_known_court_y_m": f"{court_point[1]:.6f}",
                        "inside_court": inside_court,
                        "inside_analysis_area": inside_analysis,
                        "side": (
                            decision.side.value if decision is not None
                            else TrackSide.UNKNOWN.value),
                        "active_player": bool(decision and decision.active),
                        "active_rank": (
                            decision.rank if decision is not None and decision.rank is not None
                            else ""),
                        "active_score": (
                            f"{decision.score:.6f}" if decision is not None else ""),
                        "visible": True,
                        "occluded": False,
                        "position_observed": True,
                        "visibility_state": VisibilityState.VISIBLE.value,
                        "occlusion_age_frames": 0,
                        "occlusion_evidence": "",
                        "reconnected": bool(decision and decision.reconnected),
                        "reconnection_from_raw_track_id": (
                            decision.reconnection_from_raw_track_id
                            if decision is not None
                            and decision.reconnection_from_raw_track_id is not None else ""),
                        "reconnection_score": (
                            f"{decision.reconnection_score:.6f}"
                            if decision is not None
                            and decision.reconnection_score is not None else ""),
                        "reconnection_gap_frames": (
                            decision.reconnection_gap_frames
                            if decision is not None
                            and decision.reconnection_gap_frames is not None else ""),
                        "logical_reconnection_count": (
                            decision.logical_reconnection_count
                            if decision is not None
                            and decision.logical_player_track_id is not None else ""),
                    })

                for logical_id, decision in result.members.items():
                    if not decision.active or decision.position_observed:
                        continue
                    last_point = decision.last_known_court_point
                    if last_point is not None:
                        occluded_players[logical_id] = last_point
                    csv_writer.writerow({
                        "frame_index": frame_index,
                        "timestamp_s": f"{timestamp_s:.6f}",
                        "track_id": "",
                        "raw_track_id": "",
                        "logical_player_track_id": logical_id,
                        "confidence": "",
                        "bbox_x1": "",
                        "bbox_y1": "",
                        "bbox_x2": "",
                        "bbox_y2": "",
                        "foot_x_px": "",
                        "foot_y_px": "",
                        "court_x_m": "",
                        "court_y_m": "",
                        "court_x_m_predicted": "",
                        "court_y_m_predicted": "",
                        "last_known_court_x_m": (
                            f"{last_point[0]:.6f}" if last_point is not None else ""),
                        "last_known_court_y_m": (
                            f"{last_point[1]:.6f}" if last_point is not None else ""),
                        "inside_court": "",
                        "inside_analysis_area": "",
                        "side": decision.side.value,
                        "active_player": True,
                        "active_rank": decision.rank or "",
                        "active_score": f"{decision.score:.6f}",
                        "visible": False,
                        "occluded": decision.occluded,
                        "position_observed": False,
                        "visibility_state": decision.visibility.value,
                        "occlusion_age_frames": decision.occlusion_age,
                        "occlusion_evidence": decision.occlusion_evidence or "",
                        "reconnected": False,
                        "reconnection_from_raw_track_id": "",
                        "reconnection_score": "",
                        "reconnection_gap_frames": "",
                        "logical_reconnection_count": decision.logical_reconnection_count,
                    })

                diagnostics.observe(set(int(value) for value in tracker_ids), result)

                annotated = _annotate_source(
                    frame, detections, calibration, config,
                    options.show_detections, options.show_track_ids, result,
                    options.show_occluded_players, options.show_occlusion_zones)
                tactical = draw_player_tracks_on_volleyball_court(
                    config=config,
                    player_points=current_players,
                    trajectories=trajectories,
                    trajectory_length=options.trajectory_length,
                    resolution_wh=tactical_size,
                    padding=options.tactical_padding,
                    include_free_zone=True,
                    camera_view=options.camera_view,
                    inactive_player_points=(
                        inactive_players if options.show_all_tracks else None),
                    occluded_player_points=(
                        occluded_players if options.show_occluded_players else None),
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

    diagnostics.log(written_frames, selector)
    return written_frames
