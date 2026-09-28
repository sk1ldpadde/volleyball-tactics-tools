from sports.common.calibration import CourtCalibration


def test_calibration_json_roundtrip(tmp_path) -> None:
    calibration = CourtCalibration(
        image_points={
            "far_left": (100.0, 50.0),
            "far_right": (500.0, 60.0),
            "near_right": (620.0, 450.0),
            "near_left": (40.0, 440.0),
        },
        source_video="match.mp4",
    )
    path = tmp_path / "calibration.json"

    calibration.save(path)
    restored = CourtCalibration.load(path)

    assert restored == calibration
    assert restored.to_dict()["coordinate_system"] == "meters"
