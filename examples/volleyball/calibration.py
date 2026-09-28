"""Interactive landmark-based volleyball-court calibration."""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

from sports.common.calibration import CourtCalibration
from sports.configs.volleyball import CameraView, Point, VolleyballCourtConfiguration


LOGGER = logging.getLogger(__name__)
LANDMARK_DESCRIPTIONS = {
    "far_left_corner": "far baseline / left sideline",
    "far_right_corner": "far baseline / right sideline",
    "far_attack_left": "far attack line / left sideline",
    "far_attack_right": "far attack line / right sideline",
    "net_left": "net or center line / left sideline",
    "net_right": "net or center line / right sideline",
    "near_attack_left": "near attack line / left sideline",
    "near_attack_right": "near attack line / right sideline",
    "near_left_corner": "near baseline / left sideline",
    "near_right_corner": "near baseline / right sideline",
}


def read_video_frame(video_path: Union[str, Path], timestamp_s: float = 0.0) -> np.ndarray:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open source video: {video_path}")
    try:
        if timestamp_s > 0:
            capture.set(cv2.CAP_PROP_POS_MSEC, timestamp_s * 1000.0)
        ok, frame = capture.read()
    finally:
        capture.release()
    if not ok or frame is None:
        raise RuntimeError(
            f"Could not read calibration frame at {timestamp_s:.3f}s from {video_path}")
    return frame


def _safe_pixels(points: np.ndarray) -> Optional[np.ndarray]:
    if not np.isfinite(points).all():
        return None
    limit = np.iinfo(np.int32).max // 4
    return np.rint(np.clip(points, -limit, limit)).astype(np.int32)


def draw_projected_court(
    frame: np.ndarray,
    calibration: CourtCalibration,
    config: VolleyballCourtConfiguration,
) -> np.ndarray:
    """Project the complete court, including off-screen geometry, into the frame."""
    output = frame.copy()
    transformer = calibration.create_transformer()
    corners = _safe_pixels(transformer.inverse_transform_points(
        np.asarray(config.corner_points, dtype=np.float32)))
    if corners is not None:
        cv2.polylines(output, [corners], True, (0, 255, 255), 2, cv2.LINE_AA)
    far_attack_y, near_attack_y = config.attack_line_ys
    for y, color in (
        (far_attack_y, (255, 255, 255)),
        (config.center_line_y, (0, 100, 255)),
        (near_attack_y, (255, 255, 255)),
    ):
        line = _safe_pixels(transformer.inverse_transform_points(
            np.asarray([(0.0, y), (config.width, y)], dtype=np.float32)))
        if line is not None:
            cv2.line(output, tuple(line[0]), tuple(line[1]), color, 2, cv2.LINE_AA)

    projected = transformer.inverse_transform_points(
        np.asarray(list(config.landmarks.values()), dtype=np.float32))
    rejected = set(calibration.fit.rejected)
    height, width = output.shape[:2]
    for name, point in zip(config.landmarks, projected):
        if not np.isfinite(point).all():
            continue
        x, y = np.rint(point).astype(np.int64)
        if 0 <= x < width and 0 <= y < height:
            color = (0, 0, 255) if name in rejected else (0, 255, 0)
            cv2.circle(output, (int(x), int(y)), 5, color, -1, cv2.LINE_AA)
            cv2.putText(
                output, name, (int(x) + 7, int(y) - 7),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1, cv2.LINE_AA)
    return output


def _draw_reference(
    image: np.ndarray,
    config: VolleyballCourtConfiguration,
    current_name: Optional[str],
    camera_view: CameraView,
) -> None:
    height, width = image.shape[:2]
    if camera_view is CameraView.ENDLINE:
        desired_width, desired_height = 190, 300
    else:
        desired_width, desired_height = 300, 190
    panel_width = min(desired_width, width // 3)
    panel_height = min(desired_height, height - 115)
    if panel_width < 90 or panel_height < 150:
        return
    x0, y0 = width - panel_width - 20, 100
    x1, y1 = x0 + panel_width, y0 + panel_height
    overlay = image.copy()
    cv2.rectangle(overlay, (x0 - 8, y0 - 8), (x1 + 8, y1 + 8), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.72, image, 0.28, 0, image)

    def reference_point(point: Point) -> Tuple[int, int]:
        court_x, court_y = point
        if camera_view is CameraView.ENDLINE:
            u, v = court_x / config.width, court_y / config.length
        else:
            u, v = court_y / config.length, 1.0 - court_x / config.width
        return (int(round(x0 + u * panel_width)), int(round(y0 + v * panel_height)))

    corners = np.asarray(
        [reference_point(point) for point in config.corner_points], dtype=np.int32)
    cv2.polylines(image, [corners], True, (220, 220, 220), 1, cv2.LINE_AA)
    for line_y in (*config.attack_line_ys, config.center_line_y):
        start = reference_point((0.0, line_y))
        end = reference_point((config.width, line_y))
        color = (0, 120, 255) if line_y == config.center_line_y else (220, 220, 220)
        cv2.line(image, start, end, color, 1, cv2.LINE_AA)
    if current_name is not None:
        cv2.circle(
            image, reference_point(config.landmarks[current_name]), 7,
            (0, 255, 255), -1, cv2.LINE_AA)

    camera_label = "CAMERA (near baseline)" if camera_view is CameraView.ENDLINE else "CAMERA SIDE"
    far_label = "FAR baseline" if camera_view is CameraView.ENDLINE else "FAR SIDE"
    cv2.putText(
        image, far_label, (x0, y0 - 10), cv2.FONT_HERSHEY_SIMPLEX,
        0.38, (0, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(
        image, camera_label, (x0, min(height - 5, y1 + 18)),
        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 255, 255), 1, cv2.LINE_AA)


def _quality_lines(calibration: CourtCalibration) -> List[str]:
    fit = calibration.fit
    lines = [
        f"Selected {len(calibration.landmarks)} | inliers {len(fit.inliers)} | rejected {len(fit.rejected)}",
        f"Court error m: mean {fit.mean_error_m:.3f} | median {fit.median_error_m:.3f} | max {fit.max_error_m:.3f}",
        f"Image error px: mean {fit.mean_error_px:.1f} | median {fit.median_error_px:.1f} | max {fit.max_error_px:.1f}",
        f"Heuristic quality: {fit.quality.upper()} (not an accuracy guarantee)",
    ]
    if fit.rejected:
        lines.append("Rejected: " + ", ".join(fit.rejected))
    if fit.coverage_warning:
        lines.append("WARNING: " + fit.coverage_warning)
    return lines


def _draw_selection(
    frame: np.ndarray,
    config: VolleyballCourtConfiguration,
    actions: List[Tuple[str, Optional[Point]]],
    current_index: int,
    candidate: Optional[CourtCalibration],
    error: Optional[str],
    camera_view: CameraView,
) -> np.ndarray:
    preview = frame.copy()
    selected = {name: point for name, point in actions if point is not None}
    for name, point in selected.items():
        pixel = tuple(np.rint(point).astype(int))
        color = (0, 0, 255) if candidate and name in candidate.fit.rejected else (0, 255, 0)
        cv2.circle(preview, pixel, 7, color, thickness=-1)
        cv2.putText(
            preview, name, (pixel[0] + 9, pixel[1] - 9),
            cv2.FONT_HERSHEY_SIMPLEX, 0.43, color, 1, cv2.LINE_AA)

    names = tuple(config.landmarks)
    current_name = names[current_index] if current_index < len(names) else None
    _draw_reference(preview, config, current_name, camera_view)
    if candidate is not None:
        preview = draw_projected_court(preview, candidate, config)
        lines = [
            f"Camera orientation: {camera_view.value.upper()}",
            *_quality_lines(candidate),
        ]
        heading = "ENTER accept | R recalibrate | ESC cancel"
        lines.insert(1, heading)
    else:
        point_count = len(selected)
        if current_name is None:
            heading = "All landmarks reviewed. ENTER solve | U undo | R restart | ESC cancel"
        else:
            coord = config.landmarks[current_name]
            heading = (
                f"[{current_index + 1}/10] {current_name} {coord} - "
                f"{LANDMARK_DESCRIPTIONS[current_name]}"
            )
        lines = [
            f"Camera orientation: {camera_view.value.upper()}",
            heading,
            f"Click assign | S skip | U undo | R restart | ENTER solve ({point_count}/4 minimum; 6+ recommended)",
        ]
        if error:
            lines.append("Calibration failed: " + error)
    for index, text in enumerate(lines):
        color = (0, 0, 255) if text.startswith(("Calibration failed", "WARNING")) else (0, 255, 255)
        cv2.putText(
            preview, text, (15, 28 + index * 24), cv2.FONT_HERSHEY_SIMPLEX,
            0.48, color, 1, cv2.LINE_AA)
    return preview


def collect_manual_calibration(
    frame: np.ndarray,
    source_video: Union[str, Path],
    config: VolleyballCourtConfiguration,
    ransac_threshold_m: float = 0.15,
    camera_view: CameraView = CameraView.ENDLINE,
) -> CourtCalibration:
    """Collect any visible subset of four or more named landmarks."""
    window_name = "Volleyball landmark calibration"
    camera_view = CameraView(camera_view)
    names = tuple(config.landmarks)
    actions: List[Tuple[str, Optional[Point]]] = []
    candidate: Optional[CourtCalibration] = None
    error: Optional[str] = None
    window_created = False

    def on_mouse(event: int, x: int, y: int, _flags: int, _param: object) -> None:
        nonlocal candidate, error
        if event == cv2.EVENT_LBUTTONDOWN and len(actions) < len(names) and candidate is None:
            actions.append((names[len(actions)], (float(x), float(y))))
            error = None

    try:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        window_created = True
        cv2.setMouseCallback(window_name, on_mouse)
        while True:
            preview = _draw_selection(
                frame, config, actions, len(actions), candidate, error, camera_view)
            cv2.imshow(window_name, preview)
            key = cv2.waitKey(20) & 0xFF
            if key in (ord("r"), ord("R")):
                actions.clear()
                candidate = None
                error = None
            elif key in (ord("u"), ord("U")):
                candidate = None
                error = None
                if actions:
                    actions.pop()
            elif key in (ord("s"), ord("S")) and len(actions) < len(names) and candidate is None:
                actions.append((names[len(actions)], None))
                error = None
            elif key in (10, 13):
                if candidate is not None:
                    for line in _quality_lines(candidate):
                        LOGGER.info("Calibration: %s", line)
                    return candidate
                selected: Dict[str, Point] = {
                    name: point for name, point in actions if point is not None
                }
                try:
                    candidate = CourtCalibration(
                        landmarks=selected,
                        source_video=str(Path(source_video)),
                        court_width_m=config.width,
                        court_length_m=config.length,
                        camera_view=camera_view,
                        ransac_threshold_m=ransac_threshold_m,
                    )
                except ValueError as exc:
                    error = str(exc)
            elif key == 27:
                raise RuntimeError("Manual calibration was cancelled.")
    except cv2.error as exc:
        raise RuntimeError(
            "OpenCV could not create the calibration window. Run in a graphical "
            "desktop session, or provide --calibration-file with saved points."
        ) from exc
    finally:
        if window_created:
            cv2.destroyWindow(window_name)


def show_calibration_preview(
    frame: np.ndarray,
    calibration: CourtCalibration,
    config: VolleyballCourtConfiguration,
) -> None:
    window_name = "Volleyball calibration preview"
    preview = draw_projected_court(frame, calibration, config)
    lines = [
        f"Camera orientation: {calibration.camera_view.value.upper()}",
        *_quality_lines(calibration),
    ]
    for index, line in enumerate(lines):
        cv2.putText(
            preview, line, (15, 28 + 24 * index), cv2.FONT_HERSHEY_SIMPLEX,
            0.48, (0, 255, 255), 1, cv2.LINE_AA)
    cv2.imshow(window_name, preview)
    cv2.waitKey(0)
    cv2.destroyWindow(window_name)
