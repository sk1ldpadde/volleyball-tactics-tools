"""Serializable robust planar court calibration built on ``ViewTransformer``."""

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple, Union

import cv2
import numpy as np

from sports.common.view import ViewTransformer
from sports.configs.volleyball import (
    CameraEdge,
    CameraView,
    VolleyballCourtConfiguration,
    validate_camera_edge,
)


LEGACY_POINT_NAMES = ("far_left", "far_right", "near_right", "near_left")
CALIBRATION_POINT_NAMES = LEGACY_POINT_NAMES
LEGACY_TO_LANDMARK = {
    "far_left": "far_left_corner",
    "far_right": "far_right_corner",
    "near_right": "near_right_corner",
    "near_left": "near_left_corner",
}
DEFAULT_RANSAC_THRESHOLD_M = 0.15
DEFAULT_CAMERA_EDGE_MIN_SEPARATION_PX = 5.0


def camera_edge_midpoints(
    camera_view: CameraView,
    config: VolleyballCourtConfiguration,
) -> Mapping[CameraEdge, Tuple[float, float]]:
    """Return canonical midpoints of the two camera-edge candidates."""
    camera_view = CameraView(camera_view)
    if camera_view is CameraView.SIDELINE:
        return {
            CameraEdge.X0: (0.0, config.center_line_y),
            CameraEdge.X9: (config.width, config.center_line_y),
        }
    return {
        CameraEdge.Y0: (config.width / 2.0, 0.0),
        CameraEdge.Y18: (config.width / 2.0, config.length),
    }


def infer_camera_edge(
    camera_view: CameraView,
    transformer: ViewTransformer,
    config: VolleyballCourtConfiguration,
    minimum_separation_px: float = DEFAULT_CAMERA_EDGE_MIN_SEPARATION_PX,
) -> Tuple[CameraEdge, Mapping[CameraEdge, Tuple[float, float]]]:
    """Infer the camera-near canonical edge from projected image vertical position."""
    candidates = camera_edge_midpoints(camera_view, config)
    projected = transformer.inverse_transform_points(
        np.asarray(list(candidates.values()), dtype=np.float32))
    if projected.shape != (2, 2) or not np.isfinite(projected).all():
        raise ValueError("Could not infer camera edge from finite image projections.")
    projections = {
        edge: (float(point[0]), float(point[1]))
        for edge, point in zip(candidates, projected)
    }
    ordered = list(projections)
    separation = abs(projections[ordered[0]][1] - projections[ordered[1]][1])
    if separation < minimum_separation_px:
        raise ValueError(
            "Camera edge is ambiguous: projected candidate edges differ by only "
            f"{separation:.1f} px vertically. Supply --camera-edge explicitly.")
    edge = max(projections, key=lambda item: projections[item][1])
    return edge, projections


@dataclass(frozen=True)
class ImageOcclusionZone:
    """Named camera-space polygon describing a fixed foreground obstruction."""

    name: str
    points: Tuple[Tuple[float, float], ...]

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("Image occlusion zone name cannot be empty.")
        if len(self.points) < 3:
            raise ValueError(
                f"Image occlusion zone '{self.name}' needs at least 3 points.")
        normalized = tuple((float(point[0]), float(point[1])) for point in self.points)
        if not np.isfinite(normalized).all():
            raise ValueError(
                f"Image occlusion zone '{self.name}' must contain finite points.")
        object.__setattr__(self, "points", normalized)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "points": [[float(x), float(y)] for x, y in self.points],
        }


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
    camera_view: CameraView
    camera_edge: CameraEdge
    image_occlusion_zones: Tuple[ImageOcclusionZone, ...]
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
        camera_view: Union[CameraView, str] = CameraView.ENDLINE,
        camera_edge: Optional[Union[CameraEdge, str]] = None,
        image_occlusion_zones: Optional[Tuple[ImageOcclusionZone, ...]] = None,
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
        try:
            normalized_camera_view = CameraView(camera_view)
        except ValueError as exc:
            raise ValueError(
                f"Unsupported camera view: {camera_view!r}. Expected endline or sideline."
            ) from exc
        zones = tuple(image_occlusion_zones or ())
        if len({zone.name for zone in zones}) != len(zones):
            raise ValueError("Image occlusion zone names must be unique.")

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
        if camera_edge is None or str(camera_edge) == "auto":
            normalized_camera_edge, _ = infer_camera_edge(
                normalized_camera_view, transformer, config)
        else:
            try:
                normalized_camera_edge = validate_camera_edge(
                    normalized_camera_view, CameraEdge(camera_edge))
            except ValueError as exc:
                raise ValueError(f"Unsupported camera edge: {camera_edge!r}. {exc}") from exc

        object.__setattr__(self, "landmarks", {
            name: (float(normalized[name][0]), float(normalized[name][1]))
            for name in names
        })
        object.__setattr__(self, "source_video", source_video)
        object.__setattr__(self, "court_width_m", float(court_width_m))
        object.__setattr__(self, "court_length_m", float(court_length_m))
        object.__setattr__(self, "camera_view", normalized_camera_view)
        object.__setattr__(self, "camera_edge", normalized_camera_edge)
        object.__setattr__(self, "image_occlusion_zones", zones)
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

    @property
    def camera_edge_projections(self) -> Mapping[CameraEdge, Tuple[float, float]]:
        """Image positions used to infer/inspect the camera-near edge."""
        _edge, projections = infer_camera_edge(
            self.camera_view, self._transformer, self.configuration(),
            minimum_separation_px=0.0)
        return projections

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
            "camera_view": self.camera_view.value,
            "camera_edge": self.camera_edge.value,
            "image_occlusion_zones": [
                zone.to_dict() for zone in self.image_occlusion_zones
            ],
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
        raw_zones = data.get("image_occlusion_zones", [])
        if not isinstance(raw_zones, list):
            raise ValueError("Calibration image_occlusion_zones must be a list.")
        zones = []
        for raw_zone in raw_zones:
            if not isinstance(raw_zone, Mapping):
                raise ValueError("Each image occlusion zone must be an object.")
            raw_zone_points = raw_zone.get("points")
            if not isinstance(raw_zone_points, list):
                raise ValueError("Image occlusion zone points must be a list.")
            zones.append(ImageOcclusionZone(
                name=str(raw_zone.get("name", "")),
                points=tuple(
                    (float(point[0]), float(point[1])) for point in raw_zone_points),
            ))
        return cls(
            landmarks={
                str(name): (float(point[0]), float(point[1]))
                for name, point in raw_points.items()
            },
            source_video=str(data.get("source_video", "")),
            court_width_m=float(data.get("court_width_m", 9.0)),
            court_length_m=float(data.get("court_length_m", 18.0)),
            camera_view=str(data.get("camera_view", CameraView.ENDLINE.value)),
            camera_edge=data.get("camera_edge"),
            image_occlusion_zones=tuple(zones),
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
