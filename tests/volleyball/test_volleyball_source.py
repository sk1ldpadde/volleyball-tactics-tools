import json
from pathlib import Path

import pytest

from examples.volleyball import source


VIDEO_ID = "abc123_SAFE"
VIDEO_URL = f"https://www.youtube.com/watch?v={VIDEO_ID}"


class FakeYoutubeDL:
    calls = []

    def __init__(self, options):
        self.options = options
        self.__class__.calls.append(options)

    def __enter__(self):
        return self

    def __exit__(self, _exc_type, _exc, _traceback):
        return False

    def extract_info(self, url, download):
        assert url == VIDEO_URL
        info = {
            "id": VIDEO_ID,
            "title": "../../untrusted title",
            "webpage_url": VIDEO_URL,
            "channel": "Example channel",
            "uploader": "Example uploader",
            "duration": 123,
            "upload_date": "20260102",
            "width": 1920,
            "height": 1080,
            "fps": 60,
            "ext": "mp4",
            "http_headers": {"Authorization": "must-not-be-written"},
        }
        if download:
            output = Path(self.options["outtmpl"].replace("%(ext)s", "mp4"))
            output.write_bytes(b"fake video")
            info["filepath"] = str(output)
        return info


@pytest.fixture(autouse=True)
def reset_fake_downloader():
    FakeYoutubeDL.calls = []


def test_valid_local_file_resolves_without_copy(tmp_path: Path) -> None:
    video = tmp_path / "match.mp4"
    video.write_bytes(b"video")

    resolved = source.resolve_video_source(
        source_video=video,
        youtube_url=None,
        download_dir=tmp_path / "downloads",
    )

    assert resolved.video_path == video.resolve()
    assert resolved.source_type == "local"
    assert not (tmp_path / "downloads").exists()


def test_missing_local_file_has_clear_error(tmp_path: Path) -> None:
    missing = tmp_path / "missing.mp4"

    with pytest.raises(source.VideoSourceError, match="does not exist"):
        source.resolve_video_source(
            source_video=missing,
            youtube_url=None,
            download_dir=tmp_path / "downloads",
        )


def test_local_file_expands_home(tmp_path: Path, monkeypatch) -> None:
    video = tmp_path / "Videos" / "match.mp4"
    video.parent.mkdir()
    video.write_bytes(b"video")
    monkeypatch.setenv("HOME", str(tmp_path))

    resolved = source.resolve_video_source(
        source_video=Path("~/Videos/match.mp4"),
        youtube_url=None,
        download_dir=tmp_path / "downloads",
    )

    assert resolved.video_path == video.resolve()


def test_cached_youtube_video_is_reused_without_downloader(
    tmp_path: Path, monkeypatch
) -> None:
    video_dir = tmp_path / VIDEO_ID
    video_dir.mkdir()
    cached = video_dir / "video.mp4"
    cached.write_bytes(b"cached")
    monkeypatch.setattr(source, "_probe_video", lambda _path: None)
    monkeypatch.setattr(
        source,
        "_load_youtube_dl",
        lambda: pytest.fail("yt-dlp should not be loaded for a valid cache entry"),
    )

    resolved = source.resolve_video_source(
        source_video=None,
        youtube_url=VIDEO_URL,
        download_dir=tmp_path,
    )

    assert resolved.video_path == cached.resolve()
    assert resolved.source_type == "youtube"


def test_youtube_download_uses_id_directory_and_writes_sanitized_metadata(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(source, "_load_youtube_dl", lambda: FakeYoutubeDL)
    monkeypatch.setattr(source, "_probe_video", lambda _path: None)
    monkeypatch.setattr(source.shutil, "which", lambda _name: "/usr/bin/ffmpeg")

    resolved = source.resolve_video_source(
        source_video=None,
        youtube_url=VIDEO_URL,
        download_dir=tmp_path,
        youtube_max_height=1080,
    )

    expected_dir = tmp_path / VIDEO_ID
    assert resolved.video_path == (expected_dir / "video.mp4").resolve()
    assert resolved.metadata_path == expected_dir / "metadata.json"
    metadata = json.loads(resolved.metadata_path.read_text(encoding="utf-8"))
    assert metadata["title"] == "../../untrusted title"
    assert metadata["fps"] == 60
    assert "http_headers" not in metadata
    assert not (tmp_path / "untrusted title").exists()
    assert all(options["noplaylist"] is True for options in FakeYoutubeDL.calls)
    assert "height<=1080" in FakeYoutubeDL.calls[-1]["format"]


def test_redownload_bypasses_cache_and_replaces_only_video_files(
    tmp_path: Path, monkeypatch
) -> None:
    video_dir = tmp_path / VIDEO_ID
    video_dir.mkdir()
    cached = video_dir / "video.mp4"
    cached.write_bytes(b"old")
    unrelated = video_dir / "notes.txt"
    unrelated.write_text("keep", encoding="utf-8")
    monkeypatch.setattr(source, "_load_youtube_dl", lambda: FakeYoutubeDL)
    monkeypatch.setattr(source, "_probe_video", lambda _path: None)
    monkeypatch.setattr(source.shutil, "which", lambda _name: "/usr/bin/ffmpeg")

    resolved = source.resolve_video_source(
        source_video=None,
        youtube_url=VIDEO_URL,
        download_dir=tmp_path,
        redownload=True,
    )

    assert resolved.video_path.read_bytes() == b"fake video"
    assert unrelated.read_text(encoding="utf-8") == "keep"
    assert len(FakeYoutubeDL.calls) == 2


def test_missing_ffmpeg_uses_single_file_fallback(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(source, "_load_youtube_dl", lambda: FakeYoutubeDL)
    monkeypatch.setattr(source, "_probe_video", lambda _path: None)
    monkeypatch.setattr(source.shutil, "which", lambda _name: None)

    source.resolve_video_source(
        source_video=None,
        youtube_url=VIDEO_URL,
        download_dir=tmp_path,
    )

    download_options = FakeYoutubeDL.calls[-1]
    assert "+" not in download_options["format"]
    assert "merge_output_format" not in download_options


def test_playlist_only_url_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(source.VideoSourceError, match="Playlist"):
        source.resolve_video_source(
            source_video=None,
            youtube_url="https://www.youtube.com/playlist?list=PL123",
            download_dir=tmp_path,
        )
