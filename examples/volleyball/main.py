"""Command-line entry point for the local volleyball MVP."""

import argparse
import logging
from pathlib import Path
from typing import Optional, Sequence

from sports.common.calibration import CourtCalibration
from sports.configs.volleyball import (
    CameraView,
    VolleyballCourtConfiguration,
    default_tactical_resolution,
)

try:
    from .calibration import (
        collect_manual_calibration,
        read_video_frame,
        show_calibration_preview,
    )
    from .detection import UltralyticsPlayerDetector
    from .pipeline import PipelineOptions, run_pipeline
    from .models import ModelSetupError, resolve_player_model
    from .source import VideoSourceError, resolve_video_source
except ImportError:  # Support ``python examples/volleyball/main.py``.
    from calibration import (
        collect_manual_calibration,
        read_video_frame,
        show_calibration_preview,
    )
    from detection import UltralyticsPlayerDetector
    from pipeline import PipelineOptions, run_pipeline
    from models import ModelSetupError, resolve_player_model
    from source import VideoSourceError, resolve_video_source


LOGGER = logging.getLogger("volleyball")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Detect and track volleyball players, project their foot points into "
            "metric court coordinates, and render synchronized tactical video."
        ))
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--source-video", type=Path, help="Existing local video")
    input_group.add_argument("--youtube-url", help="One public YouTube video URL")
    parser.add_argument(
        "--download-dir",
        type=Path,
        default=Path("data/youtube"),
        help="Persistent YouTube cache (default: data/youtube)",
    )
    parser.add_argument(
        "--youtube-max-height",
        type=int,
        default=1080,
        help="Maximum downloaded video height (default: 1080)",
    )
    parser.add_argument(
        "--redownload",
        action="store_true",
        help="Replace the selected YouTube video's cached media",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--camera-view",
        choices=tuple(view.value for view in CameraView),
        help=("Camera position: endline or sideline. New calibrations default to "
              "endline; saved calibrations retain their stored value."),
    )
    parser.add_argument(
        "--player-model", default="auto",
        help="auto, official yolo26n/s.pt, or a trusted local checkpoint path")
    parser.add_argument(
        "--models-dir", type=Path, default=Path("models/ultralytics"),
        help="Project-local official model cache (default: models/ultralytics)")
    parser.add_argument("--calibration", choices=("manual",), default="manual")
    parser.add_argument(
        "--calibration-file", type=Path,
        help="Reuse this JSON file; if absent, manual calibration is saved here.")
    parser.add_argument("--calibration-time", type=float, default=0.0)
    parser.add_argument(
        "--player-conf", "--confidence", dest="player_conf", type=float, default=0.25,
        help="Person detection confidence threshold (default: 0.25)")
    parser.add_argument("--person-class-id", type=int, default=0)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, or mps")
    parser.add_argument(
        "--calibration-ransac-threshold", type=float, default=0.15,
        help="Court-space RANSAC threshold in metres (default: 0.15)")
    parser.add_argument(
        "--player-imgsz", "--image-size", dest="player_imgsz", type=int,
        choices=(640, 960, 1280), default=640,
        help="Ultralytics inference image size (default: 640)")
    parser.add_argument("--side-margin", type=float, default=3.0)
    parser.add_argument("--baseline-margin", type=float, default=5.0)
    parser.add_argument("--start-time", type=float, default=0.0)
    parser.add_argument("--end-time", type=float)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument(
        "--tactical-width", type=int,
        help="Tactical canvas width (default: 450 endline, 900 sideline)")
    parser.add_argument(
        "--tactical-height", type=int,
        help="Tactical canvas height (default: 900 endline, 450 sideline)")
    parser.add_argument("--tactical-padding", type=int, default=40)
    parser.add_argument("--trajectory-length", type=int, default=30)
    parser.add_argument(
        "--show-all-tracks", action="store_true",
        help="Also draw geometrically eligible tracks rejected by max-six selection")
    parser.add_argument(
        "--net-hysteresis", type=float, default=0.5,
        help="No-switch band on each side of the 9 m net line (default: 0.5 m)")
    parser.add_argument(
        "--side-switch-frames", type=int, default=5,
        help="Consecutive beyond-band frames required to change sides (default: 5)")
    parser.add_argument("--show-calibration", action="store_true")
    parser.add_argument(
        "--debug", action="store_true", help="Show tracebacks for source errors")
    parser.add_argument(
        "--show-tactical-view", action="store_true",
        help="Open a live preview; press q to stop processing.")
    parser.add_argument("--show-detections", dest="show_detections", action="store_true")
    parser.add_argument("--hide-detections", dest="show_detections", action="store_false")
    parser.add_argument("--show-track-ids", dest="show_track_ids", action="store_true")
    parser.add_argument("--hide-track-ids", dest="show_track_ids", action="store_false")
    parser.set_defaults(show_detections=True, show_track_ids=True)
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if args.start_time < 0 or args.calibration_time < 0:
        raise ValueError("Time values cannot be negative.")
    if args.end_time is not None and args.end_time <= args.start_time:
        raise ValueError("--end-time must be later than --start-time.")
    if args.max_frames is not None and args.max_frames <= 0:
        raise ValueError("--max-frames must be positive.")
    if args.side_margin < 0 or args.baseline_margin < 0:
        raise ValueError("Analysis margins cannot be negative.")
    if not 0.0 <= args.player_conf <= 1.0:
        raise ValueError("--player-conf must be between 0 and 1.")
    if ((args.tactical_width is not None and args.tactical_width <= 0)
            or (args.tactical_height is not None and args.tactical_height <= 0)):
        raise ValueError("Tactical video dimensions must be positive.")
    if args.tactical_padding < 0:
        raise ValueError("--tactical-padding cannot be negative.")
    if args.trajectory_length <= 0:
        raise ValueError("--trajectory-length must be positive.")
    if args.youtube_max_height <= 0:
        raise ValueError("--youtube-max-height must be positive.")
    if args.calibration_ransac_threshold <= 0:
        raise ValueError("--calibration-ransac-threshold must be positive.")
    if args.net_hysteresis < 0:
        raise ValueError("--net-hysteresis cannot be negative.")
    if args.side_switch_frames <= 0:
        raise ValueError("--side-switch-frames must be positive.")


def resolve_camera_view(
    requested: Optional[str],
    calibration: Optional[CourtCalibration] = None,
) -> CameraView:
    """Resolve the effective view, rejecting saved/CLI orientation conflicts."""
    requested_view = CameraView(requested) if requested is not None else None
    if calibration is None:
        return requested_view or CameraView.ENDLINE
    if requested_view is not None and requested_view is not calibration.camera_view:
        raise ValueError(
            "Calibration was created with camera_view="
            f"{calibration.camera_view.value}, but CLI requested camera_view="
            f"{requested_view.value}. Use the matching orientation or recalibrate."
        )
    return calibration.camera_view


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(levelname)s %(message)s",
    )
    _validate_args(args)
    try:
        resolved_source = resolve_video_source(
            source_video=args.source_video,
            youtube_url=args.youtube_url,
            download_dir=args.download_dir,
            youtube_max_height=args.youtube_max_height,
            redownload=args.redownload,
        )
    except VideoSourceError as exc:
        if args.debug:
            raise
        parser.error(str(exc))
    source_video = resolved_source.video_path
    LOGGER.info("Starting volleyball analysis from %s", source_video)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    try:
        resolved_model = resolve_player_model(
            args.player_model,
            models_dir=args.models_dir,
            requested_device=args.device,
        )
    except ModelSetupError as exc:
        if args.debug:
            raise
        parser.error(str(exc))

    config = VolleyballCourtConfiguration(
        side_margin=args.side_margin,
        baseline_margin=args.baseline_margin,
    )
    try:
        detector = UltralyticsPlayerDetector(
            model_path=resolved_model.path,
            confidence=args.player_conf,
            device=resolved_model.device,
            person_class_id=args.person_class_id,
            image_size=args.player_imgsz,
        )
    except Exception as exc:
        if args.debug:
            raise
        parser.error(
            f"Could not load player model '{resolved_model.path}': {exc}")
    calibration_frame = read_video_frame(source_video, args.calibration_time)
    if args.calibration_file is not None and args.calibration_file.is_file():
        try:
            calibration = CourtCalibration.load(args.calibration_file)
            camera_view = resolve_camera_view(args.camera_view, calibration)
        except ValueError as exc:
            parser.error(str(exc))
        LOGGER.info("Loaded calibration from %s", args.calibration_file)
    else:
        camera_view = resolve_camera_view(args.camera_view)
        calibration = collect_manual_calibration(
            calibration_frame,
            source_video,
            config,
            ransac_threshold_m=args.calibration_ransac_threshold,
            camera_view=camera_view,
        )
        if args.calibration_file is not None:
            calibration.save(args.calibration_file)
            LOGGER.info("Saved reusable calibration to %s", args.calibration_file)

    # Always place a self-contained copy beside the generated outputs.
    calibration.save(args.output_dir / "calibration.json")
    config = calibration.configuration(args.side_margin, args.baseline_margin)
    if args.show_calibration:
        show_calibration_preview(calibration_frame, calibration, config)

    default_width, default_height = default_tactical_resolution(camera_view)
    options = PipelineOptions(
        start_time=args.start_time,
        end_time=args.end_time,
        max_frames=args.max_frames,
        tactical_resolution=(
            args.tactical_width or default_width,
            args.tactical_height or default_height,
        ),
        tactical_padding=args.tactical_padding,
        trajectory_length=args.trajectory_length,
        show_detections=args.show_detections,
        show_track_ids=args.show_track_ids,
        preview_tactical_view=args.show_tactical_view,
        camera_view=camera_view,
        show_all_tracks=args.show_all_tracks,
        net_hysteresis_m=args.net_hysteresis,
        side_switch_frames=args.side_switch_frames,
    )
    frame_count = run_pipeline(
        source_video=source_video,
        output_dir=args.output_dir,
        detector=detector,
        calibration=calibration,
        config=config,
        options=options,
    )
    LOGGER.info("Processed %d frames into %s", frame_count, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
