from pathlib import Path

import pytest

from examples.volleyball.main import build_parser, resolve_camera_view
from sports.common.calibration import CourtCalibration
from sports.configs.volleyball import CameraView


BASE_ARGS = ["--output-dir", "output", "--player-model", "model.pt"]


def test_cli_accepts_local_video() -> None:
    args = build_parser().parse_args(["--source-video", "match.mp4", *BASE_ARGS])

    assert args.source_video == Path("match.mp4")
    assert args.youtube_url is None
    assert args.download_dir == Path("data/youtube")
    assert args.player_model == "model.pt"
    assert args.models_dir == Path("models/ultralytics")
    assert args.camera_view is None


def test_cli_defaults_to_auto_player_model() -> None:
    args = build_parser().parse_args(
        ["--source-video", "match.mp4", "--output-dir", "output"])

    assert args.player_model == "auto"
    assert args.device == "auto"


def test_cli_accepts_youtube_url() -> None:
    args = build_parser().parse_args(
        ["--youtube-url", "https://youtu.be/abc123", *BASE_ARGS]
    )

    assert args.source_video is None
    assert args.youtube_url == "https://youtu.be/abc123"


def test_cli_accepts_volleyball_camera_views() -> None:
    for value in ("endline", "sideline"):
        args = build_parser().parse_args([
            "--source-video", "match.mp4", *BASE_ARGS,
            "--camera-view", value,
        ])
        assert args.camera_view == value


def test_saved_calibration_camera_view_conflict_is_rejected() -> None:
    calibration = CourtCalibration(
        landmarks={
            "far_left_corner": (0, 0),
            "far_right_corner": (90, 0),
            "near_right_corner": (90, 180),
            "near_left_corner": (0, 180),
        },
        camera_view=CameraView.ENDLINE,
    )

    assert resolve_camera_view(None, calibration) is CameraView.ENDLINE
    with pytest.raises(ValueError, match="created with camera_view=endline"):
        resolve_camera_view("sideline", calibration)


@pytest.mark.parametrize(
    "source_args",
    [
        [],
        [
            "--source-video",
            "match.mp4",
            "--youtube-url",
            "https://youtu.be/abc123",
        ],
    ],
)
def test_cli_requires_exactly_one_source(source_args) -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args([*source_args, *BASE_ARGS])
