from collections import deque

import numpy as np

from sports.annotators.volleyball import (
    court_to_canvas,
    draw_player_tracks_on_volleyball_court,
    draw_volleyball_court,
)
from sports.configs.volleyball import VolleyballCourtConfiguration


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

    assert empty.shape == populated.shape == (900, 600, 3)
    assert not np.array_equal(empty, populated)
