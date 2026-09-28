"""Top-down volleyball court rendering utilities."""

from typing import Mapping, Optional, Sequence, Tuple

import cv2
import numpy as np

from sports.configs.volleyball import Point, VolleyballCourtConfiguration


Color = Tuple[int, int, int]
Resolution = Tuple[int, int]


def court_to_canvas(
    config: VolleyballCourtConfiguration,
    points: np.ndarray,
    resolution_wh: Resolution = (600, 900),
    padding: int = 40,
    include_free_zone: bool = True,
) -> np.ndarray:
    """Convert metric court coordinates to tactical-canvas pixels."""
    points = np.asarray(points, dtype=np.float32)
    if points.size == 0:
        return points.reshape(-1, 2)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("points must be an Nx2 array")

    canvas_width, canvas_height = resolution_wh
    side_margin = config.side_margin if include_free_zone else 0.0
    baseline_margin = config.baseline_margin if include_free_zone else 0.0
    available_width = canvas_width - 2 * padding
    available_height = canvas_height - 2 * padding
    if available_width <= 0 or available_height <= 0:
        raise ValueError("resolution must be larger than twice the padding")

    world_width = config.width + 2 * side_margin
    world_height = config.length + 2 * baseline_margin
    scale = min(available_width / world_width, available_height / world_height)
    rendered_width = world_width * scale
    rendered_height = world_height * scale
    origin_x = (canvas_width - rendered_width) / 2.0 + side_margin * scale
    origin_y = (canvas_height - rendered_height) / 2.0 + baseline_margin * scale

    canvas_points = np.empty_like(points, dtype=np.float32)
    canvas_points[:, 0] = origin_x + points[:, 0] * scale
    canvas_points[:, 1] = origin_y + points[:, 1] * scale
    return canvas_points


def draw_volleyball_court(
    config: VolleyballCourtConfiguration,
    resolution_wh: Resolution = (600, 900),
    padding: int = 40,
    include_free_zone: bool = True,
    background_color: Color = (36, 94, 58),
    free_zone_color: Color = (50, 120, 75),
    court_color: Color = (64, 142, 200),
    line_color: Color = (255, 255, 255),
    net_color: Color = (35, 35, 35),
    line_thickness: int = 3,
) -> np.ndarray:
    """Render the full playing court, center line, and both attack lines."""
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
        p1, p2 = np.rint(court_to_canvas(
            config, extended, resolution_wh, padding, include_free_zone=True
        )).astype(int)
        cv2.rectangle(image, tuple(p1), tuple(p2), free_zone_color, thickness=-1)

    corners = np.asarray(config.corner_points, dtype=np.float32)
    canvas_corners = np.rint(court_to_canvas(
        config, corners, resolution_wh, padding, include_free_zone
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
        )
        start, end = np.rint(endpoints).astype(int)
        cv2.line(image, tuple(start), tuple(end), color, thickness)
    return image


def draw_points_on_volleyball_court(
    config: VolleyballCourtConfiguration,
    points: np.ndarray,
    labels: Optional[Sequence[str]] = None,
    court: Optional[np.ndarray] = None,
    resolution_wh: Resolution = (600, 900),
    padding: int = 40,
    include_free_zone: bool = True,
    face_color: Color = (30, 30, 230),
    edge_color: Color = (255, 255, 255),
    radius: int = 9,
) -> np.ndarray:
    """Draw metric points and optional labels on a tactical court."""
    if court is None:
        court = draw_volleyball_court(
            config, resolution_wh, padding, include_free_zone)
    canvas_points = court_to_canvas(
        config, points, resolution_wh, padding, include_free_zone)
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
    resolution_wh: Resolution = (600, 900),
    padding: int = 40,
    include_free_zone: bool = True,
) -> np.ndarray:
    """Render current tracker positions and optional recent trajectory tails."""
    court = draw_volleyball_court(
        config, resolution_wh, padding, include_free_zone)
    if trajectories:
        for track_id, path in trajectories.items():
            recent_path = np.asarray(list(path)[-trajectory_length:], dtype=np.float32)
            if len(recent_path) < 2:
                continue
            pixels = np.rint(court_to_canvas(
                config, recent_path, resolution_wh, padding, include_free_zone
            )).astype(np.int32)
            cv2.polylines(court, [pixels], False, (180, 180, 180), 2, cv2.LINE_AA)

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
    )
