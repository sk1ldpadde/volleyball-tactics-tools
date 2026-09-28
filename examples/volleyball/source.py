"""Resolve local or YouTube inputs to an ordinary local video path."""

from dataclasses import dataclass
import json
import logging
import math
from pathlib import Path
import re
import shutil
from typing import Any, Callable, Dict, Iterable, Literal, Optional, Type
from urllib.parse import parse_qs, urlparse

import cv2


LOGGER = logging.getLogger(__name__)
VIDEO_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,128}$")
METADATA_FIELDS = (
    "id",
    "title",
    "webpage_url",
    "channel",
    "uploader",
    "duration",
    "upload_date",
    "width",
    "height",
    "fps",
    "ext",
)


class VideoSourceError(RuntimeError):
    """Raised when a requested video cannot be resolved to a usable local file."""


@dataclass(frozen=True)
class ResolvedVideoSource:
    video_path: Path
    source_type: Literal["local", "youtube"]
    original_source: str
    metadata_path: Optional[Path] = None


def _load_youtube_dl() -> Type[Any]:
    try:
        from yt_dlp import YoutubeDL
    except ImportError as exc:
        raise VideoSourceError(
            "YouTube input requires the optional 'yt-dlp' package. Install the "
            "volleyball dependencies with: pip install -e '.[volleyball]'"
        ) from exc
    return YoutubeDL


def _youtube_video_id(url: str) -> str:
    try:
        parsed = urlparse(url)
    except ValueError as exc:
        raise VideoSourceError(f"Malformed YouTube URL: {url}") from exc
    hostname = (parsed.hostname or "").lower().rstrip(".")
    if parsed.scheme not in ("http", "https"):
        raise VideoSourceError("YouTube URL must start with http:// or https://.")

    video_id: Optional[str] = None
    if hostname in ("youtu.be", "www.youtu.be"):
        video_id = parsed.path.strip("/").split("/", 1)[0]
    elif (
        hostname == "youtube.com"
        or hostname.endswith(".youtube.com")
        or hostname == "youtube-nocookie.com"
        or hostname.endswith(".youtube-nocookie.com")
    ):
        query = parse_qs(parsed.query)
        if parsed.path.rstrip("/") == "/watch":
            video_id = (query.get("v") or [None])[0]
        else:
            parts = [part for part in parsed.path.split("/") if part]
            if len(parts) >= 2 and parts[0] in ("embed", "live", "shorts"):
                video_id = parts[1]
        if video_id is None and "list" in query:
            raise VideoSourceError(
                "Playlist URLs are not supported. Provide a URL for one YouTube video."
            )
    else:
        raise VideoSourceError(
            "Unsupported URL. --youtube-url currently accepts YouTube video URLs only."
        )

    if not video_id or not VIDEO_ID_PATTERN.fullmatch(video_id):
        raise VideoSourceError(
            "Could not determine a safe video ID from the YouTube URL. "
            "Provide a single-video watch, youtu.be, shorts, live, or embed URL."
        )
    return video_id


def _video_candidates(video_dir: Path) -> Iterable[Path]:
    for path in sorted(video_dir.glob("video.*")):
        if path.is_file() and not path.name.endswith((".part", ".ytdl", ".temp")):
            yield path


def _probe_video(video_path: Path) -> None:
    if not video_path.is_file() or video_path.stat().st_size <= 0:
        raise VideoSourceError(f"Downloaded video is missing or empty: {video_path}")
    capture = cv2.VideoCapture(str(video_path))
    try:
        if not capture.isOpened():
            raise VideoSourceError(
                f"Downloaded video cannot be opened by OpenCV: {video_path}"
            )
        width = float(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = float(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        frame_count = float(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if (
            not all(math.isfinite(value) for value in (width, height, fps, frame_count))
            or width <= 0
            or height <= 0
            or fps <= 0
            or frame_count < 0
        ):
            raise VideoSourceError(
                f"Downloaded video has invalid dimensions, FPS, or frame count: {video_path}"
            )
    finally:
        capture.release()


def _find_cached_video(video_dir: Path) -> Optional[Path]:
    for candidate in _video_candidates(video_dir):
        try:
            _probe_video(candidate)
        except VideoSourceError:
            continue
        return candidate.resolve()
    return None


def _sanitized_metadata(info: Dict[str, Any], video_path: Path) -> Dict[str, Any]:
    metadata: Dict[str, Any] = {}
    for field in METADATA_FIELDS:
        value = info.get(field)
        if isinstance(value, (str, int, float, bool)) or value is None:
            metadata[field] = value
    metadata["id"] = str(info["id"])
    metadata["ext"] = video_path.suffix.lstrip(".")
    return metadata


def _write_metadata(path: Path, info: Dict[str, Any], video_path: Path) -> None:
    try:
        path.write_text(
            json.dumps(_sanitized_metadata(info, video_path), indent=2, ensure_ascii=False)
            + "\n",
            encoding="utf-8",
        )
    except OSError as exc:
        raise VideoSourceError(f"Could not write YouTube metadata to {path}: {exc}") from exc


def _progress_hook() -> Callable[[Dict[str, Any]], None]:
    last_reported = -10

    def report(status: Dict[str, Any]) -> None:
        nonlocal last_reported
        if status.get("status") == "finished":
            LOGGER.info("Download finished; finalizing media container...")
            return
        if status.get("status") != "downloading":
            return
        downloaded = status.get("downloaded_bytes")
        total = status.get("total_bytes") or status.get("total_bytes_estimate")
        if not isinstance(downloaded, (int, float)) or not isinstance(total, (int, float)):
            return
        if total <= 0:
            return
        percent = min(100, max(0, int(100 * downloaded / total)))
        bucket = percent // 10 * 10
        if bucket > last_reported:
            last_reported = bucket
            LOGGER.info("Downloading: %d%%", percent)

    return report


def _downloaded_video_path(video_dir: Path, info: Dict[str, Any]) -> Path:
    possible_paths = []
    for key in ("filepath", "_filename"):
        value = info.get(key)
        if isinstance(value, str):
            possible_paths.append(Path(value))
    for item in info.get("requested_downloads") or []:
        if isinstance(item, dict) and isinstance(item.get("filepath"), str):
            possible_paths.append(Path(item["filepath"]))
    possible_paths.extend(_video_candidates(video_dir))

    seen = set()
    for candidate in possible_paths:
        candidate = candidate.expanduser().resolve()
        if candidate in seen or candidate.parent != video_dir.resolve():
            continue
        seen.add(candidate)
        if candidate.name.startswith("video.") and candidate.is_file():
            try:
                _probe_video(candidate)
            except VideoSourceError:
                continue
            return candidate
    raise VideoSourceError(
        "yt-dlp finished, but no non-empty OpenCV-compatible video file was found in "
        f"{video_dir}. Check codec support in the local OpenCV/FFmpeg installation."
    )


def _remove_video_download_files(video_dir: Path) -> None:
    for path in video_dir.glob("video.*"):
        if path.is_file() or path.is_symlink():
            path.unlink()


def _resolve_youtube_source(
    youtube_url: str,
    download_dir: Path,
    youtube_max_height: int,
    redownload: bool,
) -> ResolvedVideoSource:
    if youtube_max_height <= 0:
        raise VideoSourceError("--youtube-max-height must be positive.")

    url_video_id = _youtube_video_id(youtube_url)
    root = download_dir.expanduser().resolve()
    video_dir = root / url_video_id
    metadata_path = video_dir / "metadata.json"
    if not redownload:
        cached = _find_cached_video(video_dir)
        if cached is not None:
            LOGGER.info("Using cached YouTube video: %s", cached)
            return ResolvedVideoSource(
                video_path=cached,
                source_type="youtube",
                original_source=youtube_url,
                metadata_path=metadata_path if metadata_path.is_file() else None,
            )

    YoutubeDL = _load_youtube_dl()
    LOGGER.info("Resolving YouTube source...")
    inspect_options = {
        "quiet": True,
        "no_warnings": True,
        "noplaylist": True,
        "skip_download": True,
    }
    try:
        with YoutubeDL(inspect_options) as downloader:
            info = downloader.extract_info(youtube_url, download=False)
    except Exception as exc:
        raise VideoSourceError(
            "Could not resolve the YouTube video. It may be unavailable, private, "
            "age/login restricted, region restricted, unsupported, or unreachable. "
            f"yt-dlp reported: {exc}"
        ) from exc
    if not isinstance(info, dict) or info.get("_type") in ("playlist", "multi_video"):
        raise VideoSourceError(
            "Only one YouTube video is supported; playlist downloads are disabled."
        )
    video_id = str(info.get("id") or "")
    if not VIDEO_ID_PATTERN.fullmatch(video_id):
        raise VideoSourceError("yt-dlp returned an invalid or unsafe YouTube video ID.")

    video_dir = root / video_id
    metadata_path = video_dir / "metadata.json"
    try:
        video_dir.mkdir(parents=True, exist_ok=True)
        if redownload:
            _remove_video_download_files(video_dir)
    except OSError as exc:
        raise VideoSourceError(f"Could not prepare download directory {video_dir}: {exc}") from exc

    ffmpeg_available = shutil.which("ffmpeg") is not None
    if ffmpeg_available:
        format_selector = (
            f"bestvideo[height<={youtube_max_height}]+bestaudio/"
            f"best[height<={youtube_max_height}]"
        )
    else:
        format_selector = (
            f"best[height<={youtube_max_height}][ext=mp4]/"
            f"best[height<={youtube_max_height}]"
        )
        LOGGER.warning(
            "FFmpeg was not found; using a single-file format when available. "
            "Install FFmpeg and make sure 'ffmpeg' is on PATH for best video plus audio."
        )

    LOGGER.info("Video: %s", info.get("title") or "(untitled)")
    LOGGER.info("ID: %s", video_id)
    LOGGER.info("Resolution target: <=%dp", youtube_max_height)
    LOGGER.info("Downloading to: %s", video_dir)
    download_options = {
        "format": format_selector,
        "outtmpl": str(video_dir / "video.%(ext)s"),
        "noplaylist": True,
        "overwrites": True,
        "continuedl": True,
        "quiet": True,
        "no_warnings": True,
        "progress_hooks": [_progress_hook()],
    }
    if ffmpeg_available:
        download_options["merge_output_format"] = "mp4"

    try:
        with YoutubeDL(download_options) as downloader:
            downloaded_info = downloader.extract_info(youtube_url, download=True)
    except Exception as exc:
        ffmpeg_hint = (
            " FFmpeg is not available; install it and make sure 'ffmpeg' is on PATH "
            "if this video has no suitable combined stream."
            if not ffmpeg_available
            else ""
        )
        raise VideoSourceError(
            "YouTube download failed. Check network access, video availability, access "
            f"restrictions, and free disk space.{ffmpeg_hint} yt-dlp reported: {exc}"
        ) from exc
    if not isinstance(downloaded_info, dict):
        raise VideoSourceError("yt-dlp did not return metadata for the downloaded video.")

    video_path = _downloaded_video_path(video_dir, downloaded_info)
    _write_metadata(metadata_path, downloaded_info, video_path)
    LOGGER.info("Download complete: %s", video_path)
    return ResolvedVideoSource(
        video_path=video_path,
        source_type="youtube",
        original_source=youtube_url,
        metadata_path=metadata_path,
    )


def resolve_video_source(
    *,
    source_video: Optional[Path],
    youtube_url: Optional[str],
    download_dir: Path,
    youtube_max_height: int = 1080,
    redownload: bool = False,
) -> ResolvedVideoSource:
    """Resolve exactly one input source into a validated local path."""
    if (source_video is None) == (youtube_url is None):
        raise VideoSourceError(
            "Exactly one of source_video or youtube_url must be supplied."
        )
    if source_video is not None:
        resolved_path = source_video.expanduser().resolve()
        if not resolved_path.exists():
            raise VideoSourceError(f"Source video does not exist: {resolved_path}")
        if not resolved_path.is_file():
            raise VideoSourceError(f"Source video is not a regular file: {resolved_path}")
        return ResolvedVideoSource(
            video_path=resolved_path,
            source_type="local",
            original_source=str(source_video),
        )
    return _resolve_youtube_source(
        youtube_url=youtube_url or "",
        download_dir=download_dir,
        youtube_max_height=youtube_max_height,
        redownload=redownload,
    )
