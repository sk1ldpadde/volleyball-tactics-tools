import pytest

from sports.configs.volleyball import VolleyballCourtConfiguration


def test_official_indoor_dimensions_and_lines() -> None:
    config = VolleyballCourtConfiguration()

    assert config.width == pytest.approx(9.0)
    assert config.length == pytest.approx(18.0)
    assert config.center_line_y == pytest.approx(9.0)
    assert config.attack_line_ys == pytest.approx((6.0, 12.0))


def test_named_keypoints_use_documented_coordinate_system() -> None:
    points = VolleyballCourtConfiguration().keypoints

    assert points["far_left_corner"] == (0.0, 0.0)
    assert points["far_right_corner"] == (9.0, 0.0)
    assert points["near_right_corner"] == (9.0, 18.0)
    assert points["near_left_corner"] == (0.0, 18.0)
    assert points["left_far_attack_line"] == (0.0, 6.0)
    assert points["right_near_attack_line"] == (9.0, 12.0)


def test_analysis_area_extends_beyond_playing_court() -> None:
    config = VolleyballCourtConfiguration(side_margin=3.0, baseline_margin=5.0)

    assert not config.is_inside_court((-2.0, 20.0))
    assert config.is_inside_analysis_area((-2.0, 20.0))
    assert not config.is_inside_analysis_area((-3.1, 20.0))
