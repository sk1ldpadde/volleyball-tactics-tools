"""Interactive manual volleyball-court calibration."""

from pathlib import Path
from typing import List, Tuple, Union

import cv2
import numpy as np

from sports.common.calibration import CALIBRATION_POINT_NAMES, CourtCalibration
from sports.configs.volleyball import VolleyballCourtConfiguration


DISPLAY_LABELS = (
    "1 far-left",
    "2 far-right",
    "3 near-right",
    "4 near-left",
)


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


def draw_projected_court(
    frame: np.ndarray,
    calibration: CourtCalibration,
    config: VolleyballCourtConfiguration,
) -> np.ndarray:
    """Overlay the calibrated boundary, net, and attack lines on a camera frame."""
    output = frame.copy()
    transformer = calibration.create_transformer()
    corners = transformer.inverse_transform_points(
        np.asarray(config.corner_points, dtype=np.float32))
    cv2.polylines(
        output, [np.rint(corners).astype(np.int32)], True, (0, 255, 255), 2,
        cv2.LINE_AA)
    far_attack_y, near_attack_y = config.attack_line_ys
    for y, color in (
        (far_attack_y, (255, 255, 255)),
        (config.center_line_y, (0, 100, 255)),
        (near_attack_y, (255, 255, 255)),
    ):
        image_line = transformer.inverse_transform_points(
            np.asarray([(0.0, y), (config.width, y)], dtype=np.float32))
        p1, p2 = np.rint(image_line).astype(int)
        cv2.line(output, tuple(p1), tuple(p2), color, 2, cv2.LINE_AA)
    return output


def _draw_selection(frame: np.ndarray, points: List[Tuple[float, float]]) -> np.ndarray:
    preview = frame.copy()
    for index, point in enumerate(points):
        pixel = tuple(np.rint(point).astype(int))
        cv2.circle(preview, pixel, 7, (0, 255, 0), thickness=-1)
        cv2.putText(
            preview, DISPLAY_LABELS[index], (pixel[0] + 10, pixel[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2, cv2.LINE_AA)
    if len(points) >= 2:
        cv2.polylines(
            preview,
            [np.rint(np.asarray(points)).astype(np.int32)],
            len(points) == 4,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
    cv2.putText(
        preview,
        (
            "Click: far-left, far-right, near-right, near-left | "
            "Enter confirm | R restart | Esc cancel"
        ),
        (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2, cv2.LINE_AA)
    return preview


def collect_manual_calibration(
    frame: np.ndarray,
    source_video: Union[str, Path],
    config: VolleyballCourtConfiguration,
) -> CourtCalibration:
    """Collect four ordered court corners in an OpenCV window."""
    window_name = "Volleyball court calibration"
    points: List[Tuple[float, float]] = []
    window_created = False

    def on_mouse(event: int, x: int, y: int, _flags: int, _param: object) -> None:
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
            points.append((float(x), float(y)))

    try:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        window_created = True
        cv2.setMouseCallback(window_name, on_mouse)
        while True:
            preview = _draw_selection(frame, points)
            candidate = None
            if len(points) == 4:
                try:
                    candidate = CourtCalibration(
                        image_points=dict(zip(CALIBRATION_POINT_NAMES, points)),
                        source_video=str(Path(source_video)),
                        court_width_m=config.width,
                        court_length_m=config.length,
                    )
                    preview = draw_projected_court(preview, candidate, config)
                except ValueError:
                    candidate = None
                    cv2.putText(
                        preview, "Invalid quadrilateral - press R and retry", (15, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            cv2.imshow(window_name, preview)
            key = cv2.waitKey(20) & 0xFF
            if key in (ord("r"), ord("R")):
                points.clear()
            elif key in (10, 13) and candidate is not None:
                return candidate
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
    cv2.imshow(window_name, preview)
    cv2.waitKey(0)
    cv2.destroyWindow(window_name)
