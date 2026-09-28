from pathlib import Path

import pytest

from examples.volleyball.main import build_parser


BASE_ARGS = ["--output-dir", "output", "--player-model", "model.pt"]


def test_cli_accepts_local_video() -> None:
    args = build_parser().parse_args(["--source-video", "match.mp4", *BASE_ARGS])

    assert args.source_video == Path("match.mp4")
    assert args.youtube_url is None
    assert args.download_dir == Path("data/youtube")
    assert args.player_model == "model.pt"
    assert args.models_dir == Path("models/ultralytics")


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
