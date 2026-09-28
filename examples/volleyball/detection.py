"""Replaceable local player-detection adapters."""

from pathlib import Path
from typing import Dict, Protocol, Union

import numpy as np
import supervision as sv


class PlayerDetector(Protocol):
    """Interface required by the volleyball processing pipeline."""

    def detect(self, frame: np.ndarray) -> sv.Detections:
        """Return person detections for one BGR frame."""


class UltralyticsPlayerDetector:
    """Person-only adapter for already-resolved local Ultralytics weights."""

    def __init__(
        self,
        model_path: Union[str, Path],
        confidence: float = 0.25,
        device: str = "cpu",
        person_class_id: int = 0,
        image_size: int = 1280,
    ) -> None:
        path = Path(model_path).expanduser()
        if not path.is_file():
            raise FileNotFoundError(
                "Player model weights were not found at "
                f"'{path}'. Download or train compatible YOLO weights yourself, "
                "verify their license and provenance, and pass the local file with "
                "--player-model /path/to/model.pt. Bare model names are rejected "
                "to prevent an implicit download."
            )
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError(
                "Ultralytics is required for the built-in detector. Install the "
                "volleyball extra with: pip install -e '.[volleyball]'"
            ) from exc

        self._model = YOLO(str(path))
        self._confidence = confidence
        self._device = device
        self._person_class_id = person_class_id
        self._image_size = image_size

    @property
    def task(self) -> str:
        return str(getattr(self._model, "task", ""))

    @property
    def class_names(self) -> Dict[int, str]:
        names = getattr(self._model, "names", {})
        if isinstance(names, list):
            return dict(enumerate(str(name) for name in names))
        return {int(key): str(value) for key, value in names.items()}

    def detect(self, frame: np.ndarray) -> sv.Detections:
        result = self._model.predict(
            frame,
            conf=self._confidence,
            classes=[self._person_class_id],
            device=self._device,
            imgsz=self._image_size,
            verbose=False,
        )[0]
        detections = sv.Detections.from_ultralytics(result)
        if detections.class_id is not None:
            detections = detections[detections.class_id == self._person_class_id]
        return detections
