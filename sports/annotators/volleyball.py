"""Top-down volleyball court rendering utilities."""

from typing import Mapping, Optional, Sequence, Tuple

import cv2
import numpy as np

from sports.configs.volleyball import (
    CameraEdge,
    CameraView,
    Point,
    VolleyballCourtConfiguration,
    default_tactical_resolution,
    default_camera_edge,
    validate_camera_edge,
)


Color = Tuple[int, int, int]
Resolution = Tuple[int, int]


def court_to_canvas(
    config: VolleyballCourtConfiguration,
    points: np.ndarray,
    resolution_wh: Optional[Resolution] = None,
    padding: int = 40,
    include_free_zone: bool = True,
    camera_view: CameraView = CameraView.ENDLINE,
    camera_edge: Optional[CameraEdge] = None,
) -> np.ndarray:
    """Convert canonical metric coordinates to orientation-specific canvas pixels."""
    points = np.asarray(points, dtype=np.float32)
    if points.size == 0:
        return points.reshape(-1, 2)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("points must be an Nx2 array")

    camera_view = CameraView(camera_view)
    camera_edge = validate_camera_edge(
        camera_view, camera_edge or default_camera_edge(camera_view))
    if resolution_wh is None:
        resolution_wh = default_tactical_resolution(camera_view)
    canvas_width, canvas_height = resolution_wh
    side_margin = config.side_margin if include_free_zone else 0.0
    baseline_margin = config.baseline_margin if include_free_zone else 0.0
    available_width = canvas_width - 2 * padding
    available_height = canvas_height - 2 * padding
    if available_width <= 0 or available_height <= 0:
        raise ValueError("resolution must be larger than twice the padding")

    if camera_view is CameraView.ENDLINE:
        if camera_edge is CameraEdge.Y18:
            oriented_points = points
        else:
            oriented_points = np.column_stack((points[:, 0], config.length - points[:, 1]))
        world_width = config.width + 2 * side_margin
        world_height = config.length + 2 * baseline_margin
        minimum_x, minimum_y = -side_margin, -baseline_margin
    else:
        # Clockwise display rotation only: canonical metres remain unchanged.
        oriented_y = (
            config.width - points[:, 0]
            if camera_edge is CameraEdge.X0
            else points[:, 0]
        )
        oriented_points = np.column_stack((points[:, 1], oriented_y))
        world_width = config.length + 2 * baseline_margin
        world_height = config.width + 2 * side_margin
        minimum_x, minimum_y = -baseline_margin, -side_margin
    scale = min(available_width / world_width, available_height / world_height)
    rendered_width = world_width * scale
    rendered_height = world_height * scale
    origin_x = (canvas_width - rendered_width) / 2.0 - minimum_x * scale
    origin_y = (canvas_height - rendered_height) / 2.0 - minimum_y * scale

    canvas_points = np.empty_like(points, dtype=np.float32)
    canvas_points[:, 0] = origin_x + oriented_points[:, 0] * scale
    canvas_points[:, 1] = origin_y + oriented_points[:, 1] * scale
    return canvas_points


def draw_volleyball_court(
    config: VolleyballCourtConfiguration,
    resolution_wh: Optional[Resolution] = None,
    padding: int = 40,
    include_free_zone: bool = True,
    background_color: Color = (36, 94, 58),
    free_zone_color: Color = (50, 120, 75),
    court_color: Color = (64, 142, 200),
    line_color: Color = (255, 255, 255),
    net_color: Color = (35, 35, 35),
    line_thickness: int = 3,
    camera_view: CameraView = CameraView.ENDLINE,
    camera_edge: Optional[CameraEdge] = None,
    debug_orientation_labels: bool = False,
) -> np.ndarray:
    """Render the full playing court, center line, and both attack lines."""
    camera_view = CameraView(camera_view)
    camera_edge = validate_camera_edge(
        camera_view, camera_edge or default_camera_edge(camera_view))
    if resolution_wh is None:
        resolution_wh = default_tactical_resolution(camera_view)
    width, height = resolution_wh
    if width <= 0 or height <= 0:
        raise ValueError("resolution dimensions must be positive")
    image = np.full((height, width, 3), background_color, dtype=np.uint8)

    if include_free_zone:
        extended = np.asarray(
            [
                (-config.side_margin, -config.baseline_margin),
                (config.width + config.side_margin, config.length + config.baseline_margin),
            ],
            dtype=np.float32,
        )
        free_zone_pixels = np.rint(court_to_canvas(
            config, extended, resolution_wh, padding, include_free_zone=True,
            camera_view=camera_view,
            camera_edge=camera_edge,
        )).astype(int)
        p1 = np.min(free_zone_pixels, axis=0)
        p2 = np.max(free_zone_pixels, axis=0)
        cv2.rectangle(image, tuple(p1), tuple(p2), free_zone_color, thickness=-1)

    corners = np.asarray(config.corner_points, dtype=np.float32)
    canvas_corners = np.rint(court_to_canvas(
        config, corners, resolution_wh, padding, include_free_zone,
        camera_view=camera_view,
        camera_edge=camera_edge,
    )).astype(int)
    cv2.fillConvexPoly(image, canvas_corners, court_color)
    cv2.polylines(image, [canvas_corners], True, line_color, line_thickness)

    far_attack_y, near_attack_y = config.attack_line_ys
    for y, color, thickness in (
        (far_attack_y, line_color, line_thickness),
        (config.center_line_y, net_color, line_thickness + 2),
        (near_attack_y, line_color, line_thickness),
    ):
        endpoints = court_to_canvas(
            config,
            np.asarray([(0.0, y), (config.width, y)], dtype=np.float32),
            resolution_wh,
            padding,
            include_free_zone,
            camera_view=camera_view,
            camera_edge=camera_edge,
        )
        start, end = np.rint(endpoints).astype(int)
        cv2.line(image, tuple(start), tuple(end), color, thickness)
    if debug_orientation_labels:
        cv2.putText(
            image, f"CAMERA SIDE ({camera_edge.value.upper()})", (12, height - 14),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(
            image, "FAR SIDE", (12, 26), cv2.FONT_HERSHEY_SIMPLEX,
            0.55, (0, 255, 255), 2, cv2.LINE_AA)
        edge_points = {
            "x=0": (0.0, config.center_line_y),
            "x=9": (config.width, config.center_line_y),
            "Y0 BASELINE": (config.width / 2.0, 0.0),
            "Y18 BASELINE": (config.width / 2.0, config.length),
        }
        pixels = court_to_canvas(
            config, np.asarray(list(edge_points.values()), dtype=np.float32),
            resolution_wh, padding, include_free_zone, camera_view, camera_edge)
        for label, point in zip(edge_points, np.rint(pixels).astype(int)):
            cv2.putText(
                image, label, tuple(point + np.asarray((5, -5))),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 255, 255), 1, cv2.LINE_AA)
    return image


def draw_points_on_volleyball_court(
    config: VolleyballCourtConfiguration,
    points: np.ndarray,
    labels: Optional[Sequence[str]] = None,
    court: Optional[np.ndarray] = None,
    resolution_wh: Optional[Resolution] = None,
    padding: int = 40,
    include_free_zone: bool = True,
    face_color: Color = (30, 30, 230),
    edge_color: Color = (255, 255, 255),
    radius: int = 9,
    camera_view: CameraView = CameraView.ENDLINE,
    camera_edge: Optional[CameraEdge] = None,
) -> np.ndarray:
    """Draw metric points and optional labels on a tactical court."""
    if court is None:
        court = draw_volleyball_court(
            config, resolution_wh, padding, include_free_zone,
            camera_view=camera_view, camera_edge=camera_edge)
    canvas_points = court_to_canvas(
        config, points, resolution_wh, padding, include_free_zone,
        camera_view=camera_view, camera_edge=camera_edge)
    if labels is not None and len(labels) != len(canvas_points):
        raise ValueError("labels and points must have equal lengths")
    for index, point in enumerate(np.rint(canvas_points).astype(int)):
        center = tuple(point)
        cv2.circle(court, center, radius, face_color, thickness=-1)
        cv2.circle(court, center, radius, edge_color, thickness=2)
        if labels is not None:
            cv2.putText(
                court,
                str(labels[index]),
                (center[0] + radius + 3, center[1] - radius),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                edge_color,
                2,
                cv2.LINE_AA,
            )
    return court


def draw_player_tracks_on_volleyball_court(
    config: VolleyballCourtConfiguration,
    player_points: Mapping[int, Point],
    trajectories: Optional[Mapping[int, Sequence[Point]]] = None,
    trajectory_length: int = 30,
    resolution_wh: Optional[Resolution] = None,
    padding: int = 40,
    include_free_zone: bool = True,
    camera_view: CameraView = CameraView.ENDLINE,
    camera_edge: Optional[CameraEdge] = None,
    inactive_player_points: Optional[Mapping[int, Point]] = None,
    occluded_player_points: Optional[Mapping[int, Point]] = None,
    raw_player_points: Optional[Mapping[int, Point]] = None,
    debug_orientation_labels: bool = False,
) -> np.ndarray:
    """Render current tracker positions and optional recent trajectory tails."""
    court = draw_volleyball_court(
        config, resolution_wh, padding, include_free_zone,
        camera_view=camera_view, camera_edge=camera_edge,
        debug_orientation_labels=debug_orientation_labels)
    if trajectories:
        for track_id, path in trajectories.items():
            recent_path = np.asarray(list(path)[-trajectory_length:], dtype=np.float32)
            if len(recent_path) < 2:
                continue
            pixels = np.rint(court_to_canvas(
                config, recent_path, resolution_wh, padding, include_free_zone,
                camera_view=camera_view,
                camera_edge=camera_edge,
            )).astype(np.int32)
            cv2.polylines(court, [pixels], False, (180, 180, 180), 2, cv2.LINE_AA)

    if inactive_player_points:
        inactive_ids = list(inactive_player_points)
        inactive_points = np.asarray(
            [inactive_player_points[track_id] for track_id in inactive_ids],
            dtype=np.float32,
        )
        court = draw_points_on_volleyball_court(
            config=config,
            points=inactive_points,
            labels=[str(track_id) for track_id in inactive_ids],
            court=court,
            resolution_wh=resolution_wh,
            padding=padding,
            include_free_zone=include_free_zone,
            face_color=(100, 100, 100),
            edge_color=(190, 190, 190),
            radius=6,
            camera_view=camera_view,
            camera_edge=camera_edge,
        )

    if occluded_player_points:
        occluded_ids = list(occluded_player_points)
        occluded_points = np.asarray(
            [occluded_player_points[track_id] for track_id in occluded_ids],
            dtype=np.float32,
        )
        canvas_points = court_to_canvas(
            config, occluded_points, resolution_wh, padding, include_free_zone,
            camera_view=camera_view, camera_edge=camera_edge)
        for track_id, point in zip(occluded_ids, np.rint(canvas_points).astype(int)):
            center = tuple(point)
            cv2.circle(court, center, 10, (0, 255, 255), thickness=2, lineType=cv2.LINE_AA)
            cv2.putText(
                court, f"{track_id}?", (center[0] + 13, center[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2, cv2.LINE_AA)

    if raw_player_points:
        raw_points = np.asarray(list(raw_player_points.values()), dtype=np.float32)
        raw_pixels = court_to_canvas(
            config, raw_points, resolution_wh, padding, include_free_zone,
            camera_view=camera_view, camera_edge=camera_edge)
        for point in np.rint(raw_pixels).astype(int):
            cv2.circle(court, tuple(point), 4, (255, 0, 255), thickness=-1)

    track_ids = list(player_points)
    points = np.asarray([player_points[track_id] for track_id in track_ids], dtype=np.float32)
    return draw_points_on_volleyball_court(
        config=config,
        points=points,
        labels=[str(track_id) for track_id in track_ids],
        court=court,
        resolution_wh=resolution_wh,
        padding=padding,
        include_free_zone=include_free_zone,
        camera_view=camera_view,
        camera_edge=camera_edge,
    )
