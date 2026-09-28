"""Serializable robust planar court calibration built on ``ViewTransformer``."""

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple, Union

import cv2
import numpy as np

from sports.common.view import ViewTransformer
from sports.configs.volleyball import VolleyballCourtConfiguration


LEGACY_POINT_NAMES = ("far_left", "far_right", "near_right", "near_left")
CALIBRATION_POINT_NAMES = LEGACY_POINT_NAMES
LEGACY_TO_LANDMARK = {
    "far_left": "far_left_corner",
    "far_right": "far_right_corner",
    "near_right": "near_right_corner",
    "near_left": "near_left_corner",
}
DEFAULT_RANSAC_THRESHOLD_M = 0.15


@dataclass(frozen=True)
class CalibrationFit:
    """Quality diagnostics for an image-to-court homography."""

    method: str
    inliers: Tuple[str, ...]
    rejected: Tuple[str, ...]
    mean_error_m: float
    median_error_m: float
    max_error_m: float
    mean_error_px: float
    median_error_px: float
    max_error_px: float
    coverage_warning: Optional[str] = None

    @property
    def quality(self) -> str:
        """Heuristic label; it is not an accuracy guarantee."""
        if self.median_error_m < 0.10:
            return "good"
        if self.median_error_m <= 0.25:
            return "warning"
        return "poor"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "method": self.method,
            "inliers": list(self.inliers),
            "rejected": list(self.rejected),
            "mean_error_m": self.mean_error_m,
            "median_error_m": self.median_error_m,
            "max_error_m": self.max_error_m,
            "mean_error_px": self.mean_error_px,
            "median_error_px": self.median_error_px,
            "max_error_px": self.max_error_px,
            "quality": self.quality,
            "coverage_warning": self.coverage_warning,
        }


@dataclass(frozen=True, init=False, eq=False)
class CourtCalibration:
    """Four-or-more named image-to-volleyball-court correspondences.

    Residuals are measured in court metres because the fitted transform maps from
    image pixels to metric court coordinates. ``0.15`` m is therefore used as the
    default RANSAC inlier threshold.
    """

    landmarks: Mapping[str, Tuple[float, float]]
    source_video: str
    court_width_m: float
    court_length_m: float
    ransac_threshold_m: float
    version: int
    fit: CalibrationFit
    _transformer: ViewTransformer

    def __init__(
        self,
        landmarks: Optional[Mapping[str, Tuple[float, float]]] = None,
        *,
        image_points: Optional[Mapping[str, Tuple[float, float]]] = None,
        source_video: str = "",
        court_width_m: float = 9.0,
        court_length_m: float = 18.0,
        ransac_threshold_m: float = DEFAULT_RANSAC_THRESHOLD_M,
        version: int = 2,
    ) -> None:
        if landmarks is not None and image_points is not None:
            raise ValueError("Provide landmarks or legacy image_points, not both.")
        raw_points = landmarks if landmarks is not None else image_points
        if raw_points is None:
            raise ValueError("Calibration landmarks are required.")
        if version not in (1, 2):
            raise ValueError(f"Unsupported calibration version: {version}")
        if court_width_m <= 0 or court_length_m <= 0:
            raise ValueError("Calibration court dimensions must be positive.")
        if ransac_threshold_m <= 0:
            raise ValueError("RANSAC threshold must be positive.")

        normalized = self._normalize_landmarks(raw_points)
        if len(normalized) < 4:
            raise ValueError(
                f"Only {len(normalized)} landmarks were selected. At least 4 "
                "non-degenerate landmarks are required; 6 or more are recommended."
            )
        config = VolleyballCourtConfiguration(
            width=court_width_m, length=court_length_m)
        unknown = set(normalized) - set(config.landmarks)
        if unknown:
            raise ValueError(
                "Unknown volleyball landmark(s): " + ", ".join(sorted(unknown)))
        names = tuple(name for name in config.landmarks if name in normalized)
        source = np.asarray([normalized[name] for name in names], dtype=np.float32)
        target = np.asarray([config.landmarks[name] for name in names], dtype=np.float32)
        transformer = ViewTransformer(
            source=source,
            target=target,
            method=cv2.RANSAC,
            ransac_reproj_threshold=ransac_threshold_m,
            minimum_inliers=4,
        )
        mask = transformer.inlier_mask
        if mask is None:
            mask = np.ones(len(names), dtype=bool)
        fit = self._calculate_fit(names, source, target, transformer, mask, config)

        object.__setattr__(self, "landmarks", {
            name: (float(normalized[name][0]), float(normalized[name][1]))
            for name in names
        })
        object.__setattr__(self, "source_video", source_video)
        object.__setattr__(self, "court_width_m", float(court_width_m))
        object.__setattr__(self, "court_length_m", float(court_length_m))
        object.__setattr__(self, "ransac_threshold_m", float(ransac_threshold_m))
        object.__setattr__(self, "version", 2)
        object.__setattr__(self, "fit", fit)
        object.__setattr__(self, "_transformer", transformer)

    @staticmethod
    def _normalize_landmarks(
        points: Mapping[str, Tuple[float, float]],
    ) -> Dict[str, Tuple[float, float]]:
        normalized: Dict[str, Tuple[float, float]] = {}
        for raw_name, raw_point in points.items():
            name = LEGACY_TO_LANDMARK.get(raw_name, raw_name)
            if name in normalized:
                raise ValueError(f"Landmark '{name}' was provided more than once.")
            try:
                point = (float(raw_point[0]), float(raw_point[1]))
            except (IndexError, TypeError, ValueError) as exc:
                raise ValueError(f"Landmark '{name}' must be a two-value point.") from exc
            if not np.isfinite(point).all():
                raise ValueError(f"Landmark '{name}' must contain finite coordinates.")
            normalized[name] = point
        if len(set(normalized.values())) != len(normalized):
            raise ValueError("Each landmark must use a unique image point.")
        return normalized

    @staticmethod
    def _calculate_fit(
        names: Tuple[str, ...],
        source: np.ndarray,
        target: np.ndarray,
        transformer: ViewTransformer,
        mask: np.ndarray,
        config: VolleyballCourtConfiguration,
    ) -> CalibrationFit:
        projected_court = transformer.transform_points(source)
        projected_image = transformer.inverse_transform_points(target)
        if not np.isfinite(projected_court).all() or not np.isfinite(projected_image).all():
            raise ValueError("Calibrated landmark projection contains non-finite values.")
        court_errors = np.linalg.norm(projected_court - target, axis=1)
        image_errors = np.linalg.norm(projected_image - source, axis=1)
        inlier_errors_m = court_errors[mask]
        inlier_errors_px = image_errors[mask]
        selected_target = target[mask]
        x_span = float(np.ptp(selected_target[:, 0])) / config.width
        y_span = float(np.ptp(selected_target[:, 1])) / config.length
        warnings = []
        if int(mask.sum()) < 6:
            warnings.append(
                "Fewer than 6 inliers provide limited outlier redundancy; "
                "additional visible landmarks are recommended.")
        if x_span < 0.75 or y_span < 0.50:
            warnings.append(
                "Landmarks cover only a concentrated part of the court; "
                "extrapolation toward the opposite area may be inaccurate.")
        return CalibrationFit(
            method="RANSAC",
            inliers=tuple(name for name, keep in zip(names, mask) if keep),
            rejected=tuple(name for name, keep in zip(names, mask) if not keep),
            mean_error_m=float(np.mean(inlier_errors_m)),
            median_error_m=float(np.median(inlier_errors_m)),
            max_error_m=float(np.max(inlier_errors_m)),
            mean_error_px=float(np.mean(inlier_errors_px)),
            median_error_px=float(np.median(inlier_errors_px)),
            max_error_px=float(np.max(inlier_errors_px)),
            coverage_warning=" ".join(warnings) or None,
        )

    @property
    def image_points_array(self) -> np.ndarray:
        return np.asarray(list(self.landmarks.values()), dtype=np.float32)

    @property
    def court_points_array(self) -> np.ndarray:
        config = self.configuration()
        return np.asarray(
            [config.landmarks[name] for name in self.landmarks], dtype=np.float32)

    def create_transformer(self) -> ViewTransformer:
        return self._transformer

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, CourtCalibration):
            return NotImplemented
        return self.to_dict() == other.to_dict()

    def configuration(
        self, side_margin: float = 3.0, baseline_margin: float = 5.0,
    ) -> VolleyballCourtConfiguration:
        return VolleyballCourtConfiguration(
            width=self.court_width_m,
            length=self.court_length_m,
            side_margin=side_margin,
            baseline_margin=baseline_margin,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": 2,
            "source_video": self.source_video,
            "coordinate_system": "meters",
            "court_width_m": self.court_width_m,
            "court_length_m": self.court_length_m,
            "ransac_threshold_m": self.ransac_threshold_m,
            "landmarks": {
                name: [float(value) for value in point]
                for name, point in self.landmarks.items()
            },
            "fit": self.fit.to_dict(),
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
        version = int(data.get("version", 1))
        key = "image_points" if version == 1 else "landmarks"
        raw_points = data.get(key)
        if not isinstance(raw_points, Mapping):
            raise ValueError(f"Calibration {key} must be an object.")
        return cls(
            landmarks={
                str(name): (float(point[0]), float(point[1]))
                for name, point in raw_points.items()
            },
            source_video=str(data.get("source_video", "")),
            court_width_m=float(data.get("court_width_m", 9.0)),
            court_length_m=float(data.get("court_length_m", 18.0)),
            ransac_threshold_m=float(
                data.get("ransac_threshold_m", DEFAULT_RANSAC_THRESHOLD_M)),
            version=version,
        )

    @classmethod
    def load(cls, path: Union[str, Path]) -> "CourtCalibration":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(data, Mapping):
            raise ValueError("Calibration file must contain a JSON object.")
        return cls.from_dict(data)
