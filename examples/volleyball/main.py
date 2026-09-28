"""Command-line entry point for the local volleyball MVP."""

import argparse
import logging
from pathlib import Path
from typing import Optional, Sequence

from sports.common.calibration import CourtCalibration
from sports.configs.volleyball import VolleyballCourtConfiguration

try:
    from .calibration import (
        collect_manual_calibration,
        read_video_frame,
        show_calibration_preview,
    )
    from .detection import UltralyticsPlayerDetector
    from .pipeline import PipelineOptions, run_pipeline
except ImportError:  # Support ``python examples/volleyball/main.py``.
    from calibration import (
        collect_manual_calibration,
        read_video_frame,
        show_calibration_preview,
    )
    from detection import UltralyticsPlayerDetector
    from pipeline import PipelineOptions, run_pipeline


LOGGER = logging.getLogger("volleyball")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Detect and track volleyball players, project their foot points into "
            "metric court coordinates, and render synchronized tactical video."
        ))
    parser.add_argument("--source-video", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--player-model", type=Path, required=True)
    parser.add_argument("--calibration", choices=("manual",), default="manual")
    parser.add_argument(
        "--calibration-file", type=Path,
        help="Reuse this JSON file; if absent, manual calibration is saved here.")
    parser.add_argument("--calibration-time", type=float, default=0.0)
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--person-class-id", type=int, default=0)
    parser.add_argument("--device", default="cpu", help="cpu, cuda, cuda:0, or mps")
    parser.add_argument("--image-size", type=int, default=1280)
    parser.add_argument("--side-margin", type=float, default=3.0)
    parser.add_argument("--baseline-margin", type=float, default=5.0)
    parser.add_argument("--start-time", type=float, default=0.0)
    parser.add_argument("--end-time", type=float)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--tactical-width", type=int, default=600)
    parser.add_argument("--tactical-height", type=int, default=900)
    parser.add_argument("--tactical-padding", type=int, default=40)
    parser.add_argument("--trajectory-length", type=int, default=30)
    parser.add_argument("--show-calibration", action="store_true")
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
    if not args.source_video.is_file():
        raise FileNotFoundError(f"Source video does not exist: {args.source_video}")
    if args.start_time < 0 or args.calibration_time < 0:
        raise ValueError("Time values cannot be negative.")
    if args.end_time is not None and args.end_time <= args.start_time:
        raise ValueError("--end-time must be later than --start-time.")
    if args.max_frames is not None and args.max_frames <= 0:
        raise ValueError("--max-frames must be positive.")
    if args.side_margin < 0 or args.baseline_margin < 0:
        raise ValueError("Analysis margins cannot be negative.")
    if not args.player_model.expanduser().is_file():
        raise FileNotFoundError(
            f"Player model weights were not found at '{args.player_model}'. "
            "Provide a trusted local checkpoint with --player-model; bare model "
            "names are not downloaded automatically."
        )
    if not 0.0 <= args.confidence <= 1.0:
        raise ValueError("--confidence must be between 0 and 1.")
    if args.tactical_width <= 0 or args.tactical_height <= 0:
        raise ValueError("Tactical video dimensions must be positive.")
    if args.tactical_padding < 0:
        raise ValueError("--tactical-padding cannot be negative.")
    if args.trajectory_length <= 0:
        raise ValueError("--trajectory-length must be positive.")


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    _validate_args(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    config = VolleyballCourtConfiguration(
        side_margin=args.side_margin,
        baseline_margin=args.baseline_margin,
    )
    detector = UltralyticsPlayerDetector(
        model_path=args.player_model,
        confidence=args.confidence,
        device=args.device,
        person_class_id=args.person_class_id,
        image_size=args.image_size,
    )
    calibration_frame = read_video_frame(args.source_video, args.calibration_time)
    if args.calibration_file is not None and args.calibration_file.is_file():
        calibration = CourtCalibration.load(args.calibration_file)
        LOGGER.info("Loaded calibration from %s", args.calibration_file)
    else:
        calibration = collect_manual_calibration(
            calibration_frame, args.source_video, config)
        if args.calibration_file is not None:
            calibration.save(args.calibration_file)
            LOGGER.info("Saved reusable calibration to %s", args.calibration_file)

    # Always place a self-contained copy beside the generated outputs.
    calibration.save(args.output_dir / "calibration.json")
    config = calibration.configuration(args.side_margin, args.baseline_margin)
    if args.show_calibration:
        show_calibration_preview(calibration_frame, calibration, config)

    options = PipelineOptions(
        start_time=args.start_time,
        end_time=args.end_time,
        max_frames=args.max_frames,
        tactical_resolution=(args.tactical_width, args.tactical_height),
        tactical_padding=args.tactical_padding,
        trajectory_length=args.trajectory_length,
        show_detections=args.show_detections,
        show_track_ids=args.show_track_ids,
        preview_tactical_view=args.show_tactical_view,
    )
    frame_count = run_pipeline(
        source_video=args.source_video,
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
