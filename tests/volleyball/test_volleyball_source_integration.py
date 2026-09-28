from pathlib import Path
from types import SimpleNamespace

import numpy as np

from examples.volleyball import main as volleyball_main
from examples.volleyball.source import ResolvedVideoSource
from examples.volleyball.models import ResolvedPlayerModel


def test_pipeline_receives_only_resolved_local_path(tmp_path: Path, monkeypatch) -> None:
    model = tmp_path / "model.pt"
    model.write_bytes(b"trusted test placeholder")
    resolved_path = tmp_path / "youtube" / "abc123" / "video.mp4"
    captured = {}

    monkeypatch.setattr(
        volleyball_main,
        "resolve_video_source",
        lambda **_kwargs: ResolvedVideoSource(
            video_path=resolved_path,
            source_type="youtube",
            original_source="https://youtu.be/abc123",
        ),
    )
    monkeypatch.setattr(
        volleyball_main, "UltralyticsPlayerDetector", lambda **_kwargs: object()
    )
    monkeypatch.setattr(
        volleyball_main,
        "resolve_player_model",
        lambda *_args, **_kwargs: ResolvedPlayerModel(
            path=model,
            model_name=model.name,
            device="cpu",
            source_type="custom",
            ultralytics_version="test",
        ),
    )
    def fake_read_video_frame(video_path, _timestamp):
        captured["calibration"] = video_path
        return np.zeros((10, 10, 3), dtype=np.uint8)

    monkeypatch.setattr(volleyball_main, "read_video_frame", fake_read_video_frame)
    calibration = SimpleNamespace(
        save=lambda _path: None,
        configuration=lambda _side, _baseline: object(),
    )
    def fake_collect_calibration(_frame, source_video, _config, **_kwargs):
        captured["collection"] = source_video
        return calibration

    monkeypatch.setattr(
        volleyball_main, "collect_manual_calibration", fake_collect_calibration)

    def fake_run_pipeline(**kwargs):
        captured["pipeline"] = kwargs["source_video"]
        return 0

    monkeypatch.setattr(volleyball_main, "run_pipeline", fake_run_pipeline)

    result = volleyball_main.main(
        [
            "--youtube-url",
            "https://youtu.be/abc123",
            "--output-dir",
            str(tmp_path / "output"),
            "--player-model",
            str(model),
        ]
    )

    assert result == 0
    assert captured == {
        "calibration": resolved_path,
        "collection": resolved_path,
        "pipeline": resolved_path,
    }
