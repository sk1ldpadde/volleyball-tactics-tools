from collections import deque

import numpy as np
import pytest

from sports.annotators.volleyball import (
    court_to_canvas,
    draw_player_tracks_on_volleyball_court,
    draw_volleyball_court,
)
from sports.configs.volleyball import (
    CameraEdge,
    CameraView,
    VolleyballCourtConfiguration,
)


def test_renderer_shape_and_court_points_are_on_canvas() -> None:
    config = VolleyballCourtConfiguration()
    resolution = (600, 900)

    rendered = draw_volleyball_court(config, resolution_wh=resolution)
    pixels = court_to_canvas(
        config, np.asarray(config.corner_points), resolution_wh=resolution)

    assert rendered.shape == (900, 600, 3)
    assert rendered.dtype == np.uint8
    assert np.all(pixels[:, 0] >= 0)
    assert np.all(pixels[:, 0] < resolution[0])
    assert np.all(pixels[:, 1] >= 0)
    assert np.all(pixels[:, 1] < resolution[1])


def test_player_track_renderer_accepts_empty_and_populated_tracks() -> None:
    config = VolleyballCourtConfiguration()

    empty = draw_player_tracks_on_volleyball_court(config, {})
    populated = draw_player_tracks_on_volleyball_court(
        config,
        {3: (4.5, 9.0)},
        trajectories={3: deque([(4.0, 8.0), (4.5, 9.0)])},
    )

    assert empty.shape == populated.shape == (900, 450, 3)
    assert not np.array_equal(empty, populated)


def test_default_renderer_orientation_shapes() -> None:
    config = VolleyballCourtConfiguration()

    endline = draw_volleyball_court(config, camera_view=CameraView.ENDLINE)
    sideline = draw_volleyball_court(config, camera_view=CameraView.SIDELINE)

    assert endline.shape == (900, 450, 3)
    assert sideline.shape == (450, 900, 3)


def test_sideline_rotation_preserves_canonical_landmark_spacing() -> None:
    config = VolleyballCourtConfiguration(side_margin=0, baseline_margin=0)
    landmarks = np.asarray([
        config.landmarks["far_left_corner"],
        config.landmarks["far_right_corner"],
        config.landmarks["far_attack_left"],
        config.landmarks["net_left"],
        config.landmarks["near_attack_left"],
        config.landmarks["near_left_corner"],
    ], dtype=np.float32)

    endline = court_to_canvas(
        config, landmarks, resolution_wh=(450, 900), padding=0,
        include_free_zone=False, camera_view=CameraView.ENDLINE)
    sideline = court_to_canvas(
        config, landmarks, resolution_wh=(900, 450), padding=0,
        include_free_zone=False, camera_view=CameraView.SIDELINE)

    # Endline length is top-to-bottom; sideline is the same geometry rotated clockwise.
    np.testing.assert_allclose(endline[0], (0, 0))
    np.testing.assert_allclose(endline[-1], (0, 900))
    np.testing.assert_allclose(sideline[0], (0, 450))
    np.testing.assert_allclose(sideline[1], (0, 0))
    np.testing.assert_allclose(sideline[-1], (900, 450))
    assert sideline[3, 0] == 450  # net remains centered
    assert sideline[3, 0] - sideline[2, 0] == 150  # 3 metres
    assert sideline[4, 0] - sideline[3, 0] == 150  # 3 metres


@pytest.mark.parametrize(
    ("camera_view", "camera_edge", "near", "far", "resolution"),
    [
        (CameraView.SIDELINE, CameraEdge.X0, (0.0, 9.0), (9.0, 9.0), (900, 450)),
        (CameraView.SIDELINE, CameraEdge.X9, (9.0, 9.0), (0.0, 9.0), (900, 450)),
        (CameraView.ENDLINE, CameraEdge.Y0, (4.5, 0.0), (4.5, 18.0), (450, 900)),
        (CameraView.ENDLINE, CameraEdge.Y18, (4.5, 18.0), (4.5, 0.0), (450, 900)),
    ],
)
def test_camera_near_edge_always_renders_below_far_edge(
    camera_view, camera_edge, near, far, resolution,
) -> None:
    config = VolleyballCourtConfiguration(side_margin=0, baseline_margin=0)
    world = np.asarray([near, far, (2.0, 14.0)], dtype=np.float32)

    pixels = court_to_canvas(
        config, world, resolution_wh=resolution, padding=0,
        include_free_zone=False, camera_view=camera_view,
        camera_edge=camera_edge)

    assert pixels[0, 1] > pixels[1, 1]
    np.testing.assert_array_equal(world[2], (2.0, 14.0))
