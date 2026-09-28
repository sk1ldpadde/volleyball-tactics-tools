import json

import cv2
import numpy as np
import pytest

from sports.common.calibration import CourtCalibration
from sports.configs.volleyball import CameraView, VolleyballCourtConfiguration


def _synthetic_landmarks():
    config = VolleyballCourtConfiguration()
    court_corners = np.asarray(config.corner_points, dtype=np.float32)
    image_corners = np.asarray(
        [(230.0, 105.0), (770.0, 125.0), (970.0, 690.0), (65.0, 660.0)],
        dtype=np.float32,
    )
    court_to_image = cv2.getPerspectiveTransform(court_corners, image_corners)
    court = np.asarray(list(config.landmarks.values()), dtype=np.float32)
    image = cv2.perspectiveTransform(court.reshape(-1, 1, 2), court_to_image).reshape(-1, 2)
    return config, dict(zip(config.landmarks, map(tuple, image))), court_to_image


def _assert_complete_court(calibration, court_to_image, atol_m=2e-3):
    config = calibration.configuration()
    expected_image = cv2.perspectiveTransform(
        np.asarray(config.corner_points, dtype=np.float32).reshape(-1, 1, 2),
        court_to_image,
    ).reshape(-1, 2)
    recovered = calibration.create_transformer().transform_points(expected_image)
    np.testing.assert_allclose(recovered, config.corner_points, atol=atol_m)


@pytest.mark.parametrize(
    "selected_names",
    [
        None,  # A: all 10
        (  # B: eight, both near corners missing
            "far_left_corner", "far_right_corner", "far_attack_left",
            "far_attack_right", "net_left", "net_right",
            "near_attack_left", "near_attack_right",
        ),
        (  # C: six distributed landmarks
            "far_left_corner", "far_right_corner", "net_left", "net_right",
            "near_left_corner", "near_right_corner",
        ),
        (  # D: mathematical minimum, still non-degenerate
            "far_left_corner", "far_right_corner",
            "near_attack_left", "near_attack_right",
        ),
    ],
)
def test_partial_landmark_subsets_recover_complete_court(selected_names) -> None:
    _config, landmarks, court_to_image = _synthetic_landmarks()
    if selected_names is not None:
        landmarks = {name: landmarks[name] for name in selected_names}

    calibration = CourtCalibration(landmarks=landmarks)

    _assert_complete_court(calibration, court_to_image)
    assert len(calibration.fit.inliers) == len(landmarks)


def test_three_landmarks_are_rejected() -> None:
    _config, landmarks, _matrix = _synthetic_landmarks()
    selected = dict(list(landmarks.items())[:3])

    with pytest.raises(ValueError, match="At least 4"):
        CourtCalibration(landmarks=selected)


def test_collinear_landmarks_are_rejected() -> None:
    _config, landmarks, _matrix = _synthetic_landmarks()
    selected = {
        name: landmarks[name]
        for name in (
            "far_left_corner", "far_attack_left", "net_left", "near_left_corner")
    }

    with pytest.raises(ValueError, match="degenerate|collinear"):
        CourtCalibration(landmarks=selected)


def test_near_collinear_image_points_are_rejected_as_unstable() -> None:
    landmarks = {
        "far_left_corner": (0.0, 0.0),
        "far_right_corner": (100.0, 0.000001),
        "near_right_corner": (300.0, 0.000004),
        "near_left_corner": (200.0, 0.000002),
    }

    with pytest.raises(ValueError, match="unstable|calculated|degenerate"):
        CourtCalibration(landmarks=landmarks)


def test_ransac_rejects_one_deliberately_bad_click() -> None:
    _config, landmarks, court_to_image = _synthetic_landmarks()
    landmarks.pop("near_right_corner")
    bad_name = "near_attack_right"
    bad = np.asarray(landmarks[bad_name]) + np.asarray((80.0, 55.0))
    landmarks[bad_name] = tuple(bad)

    calibration = CourtCalibration(landmarks=landmarks)

    assert bad_name in calibration.fit.rejected
    assert len(calibration.fit.inliers) == 8
    _assert_complete_court(calibration, court_to_image, atol_m=0.01)


def test_fixed_perturbations_report_error_and_reasonable_recovery() -> None:
    _config, landmarks, court_to_image = _synthetic_landmarks()
    perturbations = np.asarray([
        (0.8, -0.4), (-0.6, 0.5), (0.4, 0.7), (-0.9, -0.2),
        (0.3, -0.5), (-0.2, 0.9), (0.7, 0.3), (-0.4, -0.8),
        (0.5, 0.6), (-0.7, 0.2),
    ])
    noisy = {
        name: tuple(np.asarray(point) + perturbations[index])
        for index, (name, point) in enumerate(landmarks.items())
    }

    calibration = CourtCalibration(landmarks=noisy)

    assert 0 < calibration.fit.mean_error_m < 0.05
    _assert_complete_court(calibration, court_to_image, atol_m=0.05)


def test_calibration_v2_json_roundtrip(tmp_path) -> None:
    _config, landmarks, _matrix = _synthetic_landmarks()
    calibration = CourtCalibration(
        landmarks=landmarks,
        source_video="match.mp4",
        camera_view=CameraView.SIDELINE,
    )
    path = tmp_path / "calibration.json"

    calibration.save(path)
    restored = CourtCalibration.load(path)

    assert restored == calibration
    assert restored.to_dict()["version"] == 2
    assert "landmarks" in restored.to_dict()
    assert "fit" in restored.to_dict()
    assert restored.camera_view is CameraView.SIDELINE
    assert restored.to_dict()["camera_view"] == "sideline"


def test_version_one_four_corner_file_migrates(tmp_path) -> None:
    legacy = {
        "version": 1,
        "coordinate_system": "meters",
        "court_width_m": 9.0,
        "court_length_m": 18.0,
        "image_points": {
            "far_left": [100.0, 50.0],
            "far_right": [500.0, 60.0],
            "near_right": [620.0, 450.0],
            "near_left": [40.0, 440.0],
        },
    }
    path = tmp_path / "v1.json"
    path.write_text(json.dumps(legacy), encoding="utf-8")

    calibration = CourtCalibration.load(path)

    assert calibration.version == 2
    assert calibration.camera_view is CameraView.ENDLINE
    assert set(calibration.landmarks) == {
        "far_left_corner", "far_right_corner", "near_right_corner",
        "near_left_corner",
    }
    np.testing.assert_allclose(
        calibration.create_transformer().transform_points(
            calibration.image_points_array),
        calibration.court_points_array,
        atol=1e-4,
    )


def test_v2_without_camera_view_defaults_to_endline() -> None:
    _config, landmarks, _matrix = _synthetic_landmarks()
    data = CourtCalibration(landmarks=landmarks).to_dict()
    data.pop("camera_view")

    calibration = CourtCalibration.from_dict(data)

    assert calibration.camera_view is CameraView.ENDLINE


def test_camera_view_never_changes_metric_homography() -> None:
    _config, landmarks, _matrix = _synthetic_landmarks()
    point = np.asarray([(500.0, 400.0)], dtype=np.float32)
    endline = CourtCalibration(landmarks=landmarks, camera_view=CameraView.ENDLINE)
    sideline = CourtCalibration(landmarks=landmarks, camera_view=CameraView.SIDELINE)

    np.testing.assert_array_equal(
        endline.create_transformer().transform_points(point),
        sideline.create_transformer().transform_points(point),
    )
