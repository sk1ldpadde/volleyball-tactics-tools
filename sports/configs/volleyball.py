"""Metric geometry for an indoor volleyball court."""

from dataclasses import dataclass
from typing import Dict, List, Tuple


Point = Tuple[float, float]


@dataclass(frozen=True)
class VolleyballCourtConfiguration:
    """Indoor court dimensions in metres.

    The origin is the far-left baseline corner. ``x`` runs left-to-right from
    0 to ``width`` and ``y`` runs far-to-near from 0 to ``length``.
    """

    width: float = 9.0
    length: float = 18.0
    attack_line_distance: float = 3.0
    side_margin: float = 3.0
    baseline_margin: float = 5.0

    def __post_init__(self) -> None:
        if self.width <= 0 or self.length <= 0:
            raise ValueError("Court width and length must be positive.")
        if not 0 < self.attack_line_distance < self.length / 2.0:
            raise ValueError(
                "Attack-line distance must be between zero and half the court length.")
        if self.side_margin < 0 or self.baseline_margin < 0:
            raise ValueError("Analysis margins cannot be negative.")

    @property
    def center_line_y(self) -> float:
        return self.length / 2.0

    @property
    def attack_line_ys(self) -> Tuple[float, float]:
        return (
            self.center_line_y - self.attack_line_distance,
            self.center_line_y + self.attack_line_distance,
        )

    @property
    def keypoints(self) -> Dict[str, Point]:
        far_attack_y, near_attack_y = self.attack_line_ys
        return {
            "far_left_corner": (0.0, 0.0),
            "far_right_corner": (self.width, 0.0),
            "left_far_attack_line": (0.0, far_attack_y),
            "right_far_attack_line": (self.width, far_attack_y),
            "left_net": (0.0, self.center_line_y),
            "right_net": (self.width, self.center_line_y),
            "left_near_attack_line": (0.0, near_attack_y),
            "right_near_attack_line": (self.width, near_attack_y),
            "near_left_corner": (0.0, self.length),
            "near_right_corner": (self.width, self.length),
        }

    @property
    def corner_points(self) -> List[Point]:
        """Corners ordered far-left, far-right, near-right, near-left."""
        points = self.keypoints
        return [
            points["far_left_corner"],
            points["far_right_corner"],
            points["near_right_corner"],
            points["near_left_corner"],
        ]

    def is_inside_court(self, point: Point) -> bool:
        x, y = point
        return 0.0 <= x <= self.width and 0.0 <= y <= self.length

    def is_inside_analysis_area(self, point: Point) -> bool:
        x, y = point
        return (
            -self.side_margin <= x <= self.width + self.side_margin
            and -self.baseline_margin <= y <= self.length + self.baseline_margin
        )
