"""Serializable planar court calibration built on :class:`ViewTransformer`."""

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Tuple, Union

import numpy as np

from sports.common.view import ViewTransformer
from sports.configs.volleyball import VolleyballCourtConfiguration


CALIBRATION_POINT_NAMES = (
    "far_left",
    "far_right",
    "near_right",
    "near_left",
)


@dataclass(frozen=True)
class CourtCalibration:
    """Four-point image-to-volleyball-court calibration."""

    image_points: Mapping[str, Tuple[float, float]]
    source_video: str = ""
    court_width_m: float = 9.0
    court_length_m: float = 18.0
    version: int = 1

    def __post_init__(self) -> None:
        if self.version != 1:
            raise ValueError(f"Unsupported calibration version: {self.version}")
        if self.court_width_m <= 0 or self.court_length_m <= 0:
            raise ValueError("Calibration court dimensions must be positive.")
        missing = set(CALIBRATION_POINT_NAMES) - set(self.image_points)
        if missing:
            raise ValueError(
                "Calibration is missing image point(s): " + ", ".join(sorted(missing)))
        image_array = self.image_points_array
        if not np.isfinite(image_array).all():
            raise ValueError("Calibration points must contain finite coordinates.")
        # Constructing the transformer also rejects degenerate quadrilaterals.
        self.create_transformer()

    @property
    def image_points_array(self) -> np.ndarray:
        return np.asarray(
            [self.image_points[name] for name in CALIBRATION_POINT_NAMES],
            dtype=np.float32,
        )

    @property
    def court_points_array(self) -> np.ndarray:
        return np.asarray(
            [
                (0.0, 0.0),
                (self.court_width_m, 0.0),
                (self.court_width_m, self.court_length_m),
                (0.0, self.court_length_m),
            ],
            dtype=np.float32,
        )

    def create_transformer(self) -> ViewTransformer:
        return ViewTransformer(
            source=self.image_points_array,
            target=self.court_points_array,
        )

    def configuration(
        self,
        side_margin: float = 3.0,
        baseline_margin: float = 5.0,
    ) -> VolleyballCourtConfiguration:
        return VolleyballCourtConfiguration(
            width=self.court_width_m,
            length=self.court_length_m,
            side_margin=side_margin,
            baseline_margin=baseline_margin,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "source_video": self.source_video,
            "coordinate_system": "meters",
            "court_width_m": self.court_width_m,
            "court_length_m": self.court_length_m,
            "image_points": {
                name: [float(value) for value in self.image_points[name]]
                for name in CALIBRATION_POINT_NAMES
            },
        }

    def save(self, path: Union[str, Path]) -> None:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "CourtCalibration":
        if data.get("coordinate_system") != "meters":
            raise ValueError("Only metre-based calibration files are supported.")
        raw_points = data.get("image_points")
        if not isinstance(raw_points, Mapping):
            raise ValueError("Calibration image_points must be an object.")
        points = {
            name: (float(raw_points[name][0]), float(raw_points[name][1]))
            for name in CALIBRATION_POINT_NAMES
        }
        return cls(
            image_points=points,
            source_video=str(data.get("source_video", "")),
            court_width_m=float(data.get("court_width_m", 9.0)),
            court_length_m=float(data.get("court_length_m", 18.0)),
            version=int(data.get("version", 1)),
        )

    @classmethod
    def load(cls, path: Union[str, Path]) -> "CourtCalibration":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(data, Mapping):
            raise ValueError("Calibration file must contain a JSON object.")
        return cls.from_dict(data)
