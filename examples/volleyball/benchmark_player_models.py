"""Benchmark official person detectors on identical frames from one local video."""

import argparse
import csv
from dataclasses import dataclass
import logging
from pathlib import Path
from statistics import mean, median
from time import perf_counter
from typing import List, Optional, Sequence

import cv2
import numpy as np

from sports.common.calibration import CourtCalibration
from sports.common.view import get_bottom_center_points

try:
    from .detection import UltralyticsPlayerDetector
    from .models import ModelSetupError, resolve_player_model
    from .source import VideoSourceError, resolve_video_source
except ImportError:  # Support direct script execution.
    from detection import UltralyticsPlayerDetector
    from models import ModelSetupError, resolve_player_model
    from source import VideoSourceError, resolve_video_source


LOGGER = logging.getLogger("volleyball.player_benchmark")
DEFAULT_MODELS = ("yolo26n.pt", "yolo26s.pt", "yolo26m.pt")
DEFAULT_IMAGE_SIZES = (640, 960)


@dataclass(frozen=True)
class BenchmarkResult:
    model: str
    image_size: int
    frames: int
    mean_inference_ms: float
    median_inference_ms: float
    fps: float
    median_raw_persons: float
    mean_raw_persons: float
    mean_confidence: float
    median_plausible_persons: float
    mean_plausible_persons: float
    frames_under_12: int
    frames_over_12: int


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare official YOLO26 person detectors on identical video frames. "
            "Model initialization and one warm-up inference are excluded from timing."
        ))
    parser.add_argument("--source-video", type=Path, required=True)
    parser.add_argument("--calibration-file", type=Path)
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    parser.add_argument(
        "--player-imgsz", nargs="+", type=int, choices=(640, 960, 1280),
        default=list(DEFAULT_IMAGE_SIZES))
    parser.add_argument("--player-conf", type=float, default=0.25)
    parser.add_argument("--person-class-id", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--models-dir", type=Path, default=Path("models/ultralytics"))
    parser.add_argument("--start-time", type=float, default=0.0)
    parser.add_argument("--frames", type=int, default=300)
    parser.add_argument("--side-margin", type=float, default=3.0)
    parser.add_argument("--baseline-margin", type=float, default=5.0)
    parser.add_argument("--output-csv", type=Path)
    parser.add_argument("--debug", action="store_true")
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if args.frames <= 0:
        raise ValueError("--frames must be positive")
    if args.start_time < 0:
        raise ValueError("--start-time cannot be negative")
    if not 0 <= args.player_conf <= 1:
        raise ValueError("--player-conf must be between 0 and 1")
    if args.side_margin < 0 or args.baseline_margin < 0:
        raise ValueError("Analysis margins cannot be negative")


def _open_capture(source: Path, start_frame: int) -> cv2.VideoCapture:
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open source video: {source}")
    capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    return capture


def benchmark_combination(
    *,
    source_video: Path,
    detector: UltralyticsPlayerDetector,
    model_name: str,
    image_size: int,
    start_frame: int,
    frames: int,
    calibration: Optional[CourtCalibration],
    side_margin: float,
    baseline_margin: float,
) -> BenchmarkResult:
    """Measure one model/size combination on the requested sequential frame IDs."""
    capture = _open_capture(source_video, start_frame)
    try:
        ok, warmup_frame = capture.read()
        if not ok or warmup_frame is None:
            raise RuntimeError("Video ended before the requested benchmark start time")
        detector.detect(warmup_frame)
    finally:
        capture.release()

    transformer = calibration.create_transformer() if calibration is not None else None
    config = (
        calibration.configuration(side_margin, baseline_margin)
        if calibration is not None else None)
    capture = _open_capture(source_video, start_frame)
    durations: List[float] = []
    raw_counts: List[int] = []
    plausible_counts: List[int] = []
    confidences: List[float] = []
    try:
        for _ in range(frames):
            ok, frame = capture.read()
            if not ok or frame is None:
                break
            started = perf_counter()
            detections = detector.detect(frame)
            durations.append(perf_counter() - started)
            raw_counts.append(len(detections))
            if detections.confidence is not None:
                confidences.extend(float(value) for value in detections.confidence)

            plausible = len(detections)
            if transformer is not None and config is not None:
                feet = get_bottom_center_points(detections.xyxy)
                court_points = transformer.transform_points(feet)
                plausible = sum(
                    config.is_inside_analysis_area((float(point[0]), float(point[1])))
                    for point in court_points
                )
            plausible_counts.append(plausible)
    finally:
        capture.release()

    if not durations:
        raise RuntimeError("No benchmark frames could be read")
    mean_seconds = mean(durations)
    return BenchmarkResult(
        model=model_name,
        image_size=image_size,
        frames=len(durations),
        mean_inference_ms=mean_seconds * 1000.0,
        median_inference_ms=median(durations) * 1000.0,
        fps=1.0 / mean_seconds,
        median_raw_persons=median(raw_counts),
        mean_raw_persons=mean(raw_counts),
        mean_confidence=mean(confidences) if confidences else float("nan"),
        median_plausible_persons=median(plausible_counts),
        mean_plausible_persons=mean(plausible_counts),
        frames_under_12=sum(value < 12 for value in plausible_counts),
        frames_over_12=sum(value > 12 for value in plausible_counts),
    )


def _print_results(results: Sequence[BenchmarkResult], calibrated: bool) -> None:
    count_label = "court-area" if calibrated else "raw/plausible"
    print(f"\nPlausible count basis: {count_label}")
    header = (
        "Model", "imgsz", "frames", "FPS", "mean ms", "median ms",
        "raw med", "raw mean", "conf", "plaus med", "plaus mean", "<12", ">12",
    )
    widths = (12, 6, 7, 8, 9, 10, 8, 9, 7, 10, 11, 6, 6)
    print(" ".join(value.ljust(width) for value, width in zip(header, widths)))
    for result in results:
        values = (
            result.model, str(result.image_size), str(result.frames), f"{result.fps:.2f}",
            f"{result.mean_inference_ms:.1f}", f"{result.median_inference_ms:.1f}",
            f"{result.median_raw_persons:.1f}", f"{result.mean_raw_persons:.2f}",
            f"{result.mean_confidence:.3f}", f"{result.median_plausible_persons:.1f}",
            f"{result.mean_plausible_persons:.2f}", str(result.frames_under_12),
            str(result.frames_over_12),
        )
        print(" ".join(value.ljust(width) for value, width in zip(values, widths)))


def _write_results(path: Path, results: Sequence[BenchmarkResult]) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(BenchmarkResult.__dataclass_fields__))
        writer.writeheader()
        for result in results:
            writer.writerow(result.__dict__)
    LOGGER.info("Saved benchmark results to %s", path)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(levelname)s %(message)s")
    try:
        _validate_args(args)
        source = resolve_video_source(
            source_video=args.source_video,
            youtube_url=None,
            download_dir=Path("data/youtube"),
        ).video_path
        calibration = (
            CourtCalibration.load(args.calibration_file)
            if args.calibration_file is not None else None)
        capture = _open_capture(source, 0)
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        capture.release()
        if fps <= 0:
            raise RuntimeError("Source video has invalid FPS")
        start_frame = int(round(args.start_time * fps))

        results = []
        for model_name in args.models:
            resolved = resolve_player_model(
                model_name,
                models_dir=args.models_dir,
                requested_device=args.device,
            )
            for image_size in args.player_imgsz:
                LOGGER.info(
                    "Benchmarking %s @ %d on %s (%d frames from frame %d)",
                    model_name, image_size, resolved.device, args.frames, start_frame)
                detector = UltralyticsPlayerDetector(
                    resolved.path,
                    confidence=args.player_conf,
                    device=resolved.device,
                    person_class_id=args.person_class_id,
                    image_size=image_size,
                )
                results.append(benchmark_combination(
                    source_video=source,
                    detector=detector,
                    model_name=model_name,
                    image_size=image_size,
                    start_frame=start_frame,
                    frames=args.frames,
                    calibration=calibration,
                    side_margin=args.side_margin,
                    baseline_margin=args.baseline_margin,
                ))
        _print_results(results, calibrated=calibration is not None)
        if args.output_csv is not None:
            _write_results(args.output_csv, results)
        return 0
    except (ModelSetupError, VideoSourceError, ValueError, RuntimeError) as exc:
        if args.debug:
            raise
        parser.error(str(exc))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
