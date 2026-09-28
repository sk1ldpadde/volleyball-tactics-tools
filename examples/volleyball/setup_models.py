"""Download and smoke-test the official volleyball MVP player detector."""

import argparse
import logging
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

try:
    from .detection import UltralyticsPlayerDetector
    from .models import ModelSetupError, inspect_hardware, resolve_player_model
except ImportError:
    from detection import UltralyticsPlayerDetector
    from models import ModelSetupError, inspect_hardware, resolve_player_model


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download and smoke-test official person-detection weights.")
    parser.add_argument("--player-model", default="auto")
    parser.add_argument(
        "--models-dir", type=Path, default=Path("models/ultralytics"))
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda:0, or mps")
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--debug", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(levelname)s %(message)s")
    try:
        hardware = inspect_hardware()
        resolved = resolve_player_model(
            args.player_model,
            models_dir=args.models_dir,
            requested_device=args.device,
        )
        detector = UltralyticsPlayerDetector(
            resolved.path, device=resolved.device, image_size=args.image_size)
        detections = detector.detect(np.zeros((640, 640, 3), dtype=np.uint8))
        if detector.task != "detect":
            raise ModelSetupError(
                f"Expected a detection model, but model task is '{detector.task}'.")
        if detector.class_names.get(0) != "person":
            raise ModelSetupError(
                "The model does not expose COCO class 0 as 'person'.")
    except Exception as exc:
        if args.debug:
            raise
        parser.error(str(exc))

    print("Volleyball model setup")
    print(f"Ultralytics: {resolved.ultralytics_version}")
    print(f"Torch: {hardware.torch_version}")
    print(f"CUDA: {'available' if hardware.cuda_available else 'unavailable'}")
    print(f"MPS: {'available' if hardware.mps_available else 'unavailable'}")
    print(f"Selected player model: {resolved.model_name}")
    print(f"Device: {resolved.device}")
    print(f"Model path: {resolved.path}")
    print(f"Smoke test: PASS ({len(detections)} detections on synthetic image)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
