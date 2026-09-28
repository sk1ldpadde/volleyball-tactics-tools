"""Temporal selection of plausible active volleyball players from raw tracks."""

from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Deque, Dict, Iterable, Optional, Sequence, Tuple

import numpy as np

from sports.configs.volleyball import Point, VolleyballCourtConfiguration


BBox = Tuple[float, float, float, float]


class TrackSide(str, Enum):
    """Canonical court half occupied by a track."""

    FAR = "far"
    NEAR = "near"
    UNKNOWN = "unknown"


class VisibilityState(str, Enum):
    """Whether an active member has a current measured position."""

    VISIBLE = "visible"
    OCCLUDED = "occluded"
    LOST = "lost"


@dataclass(frozen=True)
class ActivePlayerSelectorConfig:
    """Explainable weights and temporal thresholds used by the selector."""

    max_players_per_side: int = 6
    net_hysteresis_m: float = 0.5
    side_switch_frames: int = 5
    track_age_normalization_frames: int = 120
    active_occlusion_grace_frames: int = 30
    active_confirmation_frames: int = 10
    confidence_weight: float = 1.0
    continuity_weight: float = 0.35
    active_history_bonus: float = 0.8
    selection_history_weight: float = 0.2
    court_proximity_weight: float = 0.25
    dynamic_overlap_threshold: float = 0.25
    movement_window_frames: int = 90
    stationary_footprint_m: float = 0.35
    stationary_referee_penalty: float = 1.25
    net_referee_band_m: float = 1.0

    def __post_init__(self) -> None:
        if not 1 <= self.max_players_per_side <= 6:
            raise ValueError(
                "max_players_per_side must be between 1 and volleyball's "
                "hard maximum of 6")
        if self.net_hysteresis_m < 0:
            raise ValueError("net_hysteresis_m cannot be negative")
        if self.side_switch_frames <= 0:
            raise ValueError("side_switch_frames must be positive")
        if self.track_age_normalization_frames <= 0:
            raise ValueError("track_age_normalization_frames must be positive")
        if self.active_occlusion_grace_frames < 0:
            raise ValueError("active_occlusion_grace_frames cannot be negative")
        if self.active_confirmation_frames <= 0:
            raise ValueError("active_confirmation_frames must be positive")
        if not 0.0 <= self.dynamic_overlap_threshold <= 1.0:
            raise ValueError("dynamic_overlap_threshold must be between 0 and 1")
        if self.movement_window_frames <= 1:
            raise ValueError("movement_window_frames must be greater than 1")
        if self.stationary_footprint_m < 0 or self.net_referee_band_m < 0:
            raise ValueError("movement and net thresholds cannot be negative")


@dataclass(frozen=True)
class TrackCandidate:
    """One currently visible, geometrically eligible raw track."""

    track_id: int
    court_point: Point
    confidence: float
    inside_court: bool
    bbox: Optional[BBox] = None
    in_occlusion_zone: bool = False


@dataclass(frozen=True)
class PlayerSelection:
    """Selector decision for a visible candidate or reserved active member."""

    side: TrackSide
    active: bool
    rank: Optional[int]
    score: float
    visibility: VisibilityState = VisibilityState.VISIBLE
    position_observed: bool = True
    occlusion_age: int = 0
    last_known_court_point: Optional[Point] = None
    last_known_bbox: Optional[BBox] = None
    occlusion_evidence: Optional[str] = None

    @property
    def visible(self) -> bool:
        return self.visibility is VisibilityState.VISIBLE

    @property
    def occluded(self) -> bool:
        return self.visibility is VisibilityState.OCCLUDED


@dataclass
class _TrackMemory:
    side: TrackSide = TrackSide.UNKNOWN
    pending_side: TrackSide = TrackSide.UNKNOWN
    pending_side_frames: int = 0
    observations: int = 0
    selected_frames: int = 0
    last_seen_frame: Optional[int] = None
    last_selected_frame: Optional[int] = None
    last_known_court_point: Optional[Point] = None
    last_known_bbox: Optional[BBox] = None
    last_seen_in_occlusion_zone: bool = False
    missing_evidence: Optional[str] = None
    was_active: bool = False
    positions: Deque[Point] = field(default_factory=deque)


@dataclass
class ActivePlayerSelector:
    """Reserve up to six logical active memberships per canonical court side.

    Current observations and logical membership are deliberately separate. A stable
    active member may reserve a slot while its position is unobserved, but no current
    metric point is fabricated. Dynamic person overlap and configured static image
    zones classify likely occlusion; the short reservation also protects unexplained
    detector dropouts.
    """

    court: VolleyballCourtConfiguration
    config: ActivePlayerSelectorConfig = field(default_factory=ActivePlayerSelectorConfig)
    _memory: Dict[int, _TrackMemory] = field(default_factory=dict, init=False)

    def select(
        self,
        candidates: Iterable[TrackCandidate],
        frame_index: int,
        occluder_bboxes: Sequence[BBox] = (),
    ) -> Dict[int, PlayerSelection]:
        candidate_list = list(candidates)
        visible_ids = {candidate.track_id for candidate in candidate_list}
        scored_by_side = {TrackSide.FAR: [], TrackSide.NEAR: []}
        reserved_by_side = {TrackSide.FAR: [], TrackSide.NEAR: []}
        decisions: Dict[int, PlayerSelection] = {}

        for candidate in candidate_list:
            memory = self._memory.setdefault(candidate.track_id, _TrackMemory())
            self._update_side(memory, candidate.court_point[1], frame_index)
            memory.observations += 1
            memory.last_seen_frame = frame_index
            memory.last_known_court_point = candidate.court_point
            memory.last_known_bbox = candidate.bbox
            memory.last_seen_in_occlusion_zone = candidate.in_occlusion_zone
            memory.missing_evidence = None
            memory.positions.append(candidate.court_point)
            while len(memory.positions) > self.config.movement_window_frames:
                memory.positions.popleft()
            score = self._score(candidate, memory, frame_index)
            if memory.side in scored_by_side:
                scored_by_side[memory.side].append((score, memory.observations, candidate))
            decisions[candidate.track_id] = PlayerSelection(
                side=memory.side,
                active=False,
                rank=None,
                score=score,
                last_known_court_point=candidate.court_point,
                last_known_bbox=candidate.bbox,
            )

        for track_id, memory in self._memory.items():
            if track_id in visible_ids or not self._can_reserve(memory, frame_index):
                continue
            evidence = self._occlusion_evidence(memory, occluder_bboxes)
            visibility = (
                VisibilityState.OCCLUDED if evidence is not None else VisibilityState.LOST)
            age = frame_index - int(memory.last_seen_frame)
            incumbent_score = self._incumbent_score(memory, age)
            if memory.side in reserved_by_side:
                reserved_by_side[memory.side].append(
                    (incumbent_score, memory.selected_frames, track_id, visibility, age, evidence)
                )

        for side in (TrackSide.FAR, TrackSide.NEAR):
            reserved = reserved_by_side[side]
            reserved.sort(key=lambda item: (-item[0], -item[1], item[2]))
            selected_ids = set()
            rank = 1
            for score, _history, track_id, visibility, age, evidence in reserved[
                :self.config.max_players_per_side
            ]:
                memory = self._memory[track_id]
                decisions[track_id] = PlayerSelection(
                    side=side,
                    active=True,
                    rank=rank,
                    score=score,
                    visibility=visibility,
                    position_observed=False,
                    occlusion_age=age,
                    last_known_court_point=memory.last_known_court_point,
                    last_known_bbox=memory.last_known_bbox,
                    occlusion_evidence=evidence,
                )
                self._mark_selected(memory, frame_index)
                selected_ids.add(track_id)
                rank += 1

            scored = scored_by_side[side]
            scored.sort(key=lambda item: (-item[0], -item[1], item[2].track_id))
            remaining = self.config.max_players_per_side - len(selected_ids)
            for score, _age, candidate in scored[:remaining]:
                decisions[candidate.track_id] = PlayerSelection(
                    side=side,
                    active=True,
                    rank=rank,
                    score=score,
                    last_known_court_point=candidate.court_point,
                    last_known_bbox=candidate.bbox,
                )
                self._mark_selected(self._memory[candidate.track_id], frame_index)
                selected_ids.add(candidate.track_id)
                rank += 1

            for track_id, memory in self._memory.items():
                if memory.side is side:
                    memory.was_active = track_id in selected_ids

        return decisions

    def _can_reserve(self, memory: _TrackMemory, frame_index: int) -> bool:
        if (
            not memory.was_active
            or memory.last_seen_frame is None
            or memory.last_selected_frame is None
            or memory.selected_frames < self.config.active_confirmation_frames
        ):
            return False
        missing_frames = frame_index - memory.last_seen_frame
        return 1 <= missing_frames <= self.config.active_occlusion_grace_frames

    def _occlusion_evidence(
        self, memory: _TrackMemory, occluder_bboxes: Sequence[BBox],
    ) -> Optional[str]:
        if memory.last_seen_in_occlusion_zone:
            memory.missing_evidence = "static_zone"
        elif memory.last_known_bbox is not None and any(
            _bbox_overlap_fraction(memory.last_known_bbox, box)
            >= self.config.dynamic_overlap_threshold
            for box in occluder_bboxes
        ):
            memory.missing_evidence = "dynamic_person_overlap"
        return memory.missing_evidence

    def _update_side(
        self, memory: _TrackMemory, court_y: float, frame_index: int,
    ) -> None:
        center = self.court.center_line_y
        if memory.side is TrackSide.UNKNOWN:
            memory.side = TrackSide.FAR if court_y < center else TrackSide.NEAR
            return

        if (memory.last_seen_frame is not None
                and frame_index != memory.last_seen_frame + 1):
            memory.pending_side = TrackSide.UNKNOWN
            memory.pending_side_frames = 0

        proposed = TrackSide.UNKNOWN
        if memory.side is TrackSide.FAR and court_y > center + self.config.net_hysteresis_m:
            proposed = TrackSide.NEAR
        elif (memory.side is TrackSide.NEAR
              and court_y < center - self.config.net_hysteresis_m):
            proposed = TrackSide.FAR

        if proposed is TrackSide.UNKNOWN:
            memory.pending_side = TrackSide.UNKNOWN
            memory.pending_side_frames = 0
        elif proposed is memory.pending_side:
            memory.pending_side_frames += 1
        else:
            memory.pending_side = proposed
            memory.pending_side_frames = 1

        if memory.pending_side_frames >= self.config.side_switch_frames:
            memory.side = memory.pending_side
            memory.pending_side = TrackSide.UNKNOWN
            memory.pending_side_frames = 0

    def _score(
        self, candidate: TrackCandidate, memory: _TrackMemory, frame_index: int,
    ) -> float:
        confidence = max(0.0, min(1.0, candidate.confidence))
        normalized_age = min(
            1.0, memory.observations / self.config.track_age_normalization_frames)
        normalized_history = min(
            1.0, memory.selected_frames / self.config.track_age_normalization_frames)
        recently_active = (
            memory.last_selected_frame is not None
            and frame_index - memory.last_selected_frame
            <= self.config.active_occlusion_grace_frames
        )
        active_stability = min(
            1.0, memory.selected_frames / self.config.active_confirmation_frames)
        proximity = (
            1.0 if candidate.inside_court
            else self._court_proximity(candidate.court_point))
        stationary_penalty = self._stationary_referee_penalty(candidate, memory)
        return (
            self.config.confidence_weight * confidence
            + self.config.continuity_weight * normalized_age
            + self.config.active_history_bonus * float(recently_active) * active_stability
            + self.config.selection_history_weight * normalized_history
            + self.config.court_proximity_weight * proximity
            - stationary_penalty
        )

    def _stationary_referee_penalty(
        self, candidate: TrackCandidate, memory: _TrackMemory,
    ) -> float:
        if len(memory.positions) < self.config.movement_window_frames:
            return 0.0
        positions = np.asarray(memory.positions, dtype=np.float32)
        center = np.median(positions, axis=0)
        # The 90th-percentile radius ignores occasional bottom-center jumps caused by
        # arm gestures or partial boxes while still requiring long-term confinement.
        footprint = float(np.percentile(
            np.linalg.norm(positions - center, axis=1), 90))
        suspicious_location = (
            candidate.in_occlusion_zone
            or abs(candidate.court_point[1] - self.court.center_line_y)
            <= self.config.net_referee_band_m
        )
        if suspicious_location and footprint <= self.config.stationary_footprint_m:
            return self.config.stationary_referee_penalty
        return 0.0

    def _incumbent_score(self, memory: _TrackMemory, missing_frames: int) -> float:
        history = min(
            1.0, memory.selected_frames / self.config.track_age_normalization_frames)
        grace = max(1, self.config.active_occlusion_grace_frames)
        recency = max(0.0, 1.0 - (missing_frames - 1) / grace)
        return (
            self.config.active_history_bonus
            + self.config.selection_history_weight * history
            + recency
        )

    @staticmethod
    def _mark_selected(memory: _TrackMemory, frame_index: int) -> None:
        memory.selected_frames += 1
        memory.last_selected_frame = frame_index

    def _court_proximity(self, point: Point) -> float:
        x, y = point
        if self.court.is_inside_court(point):
            return 1.0
        outside_x = max(-x, 0.0, x - self.court.width)
        outside_y = max(-y, 0.0, y - self.court.length)
        normalized_x = outside_x / max(self.court.side_margin, 1e-9)
        normalized_y = outside_y / max(self.court.baseline_margin, 1e-9)
        return max(0.0, 1.0 - max(normalized_x, normalized_y))


def _bbox_overlap_fraction(first: BBox, second: BBox) -> float:
    """Intersection divided by the smaller box area; useful for occlusion evidence."""
    left = max(first[0], second[0])
    top = max(first[1], second[1])
    right = min(first[2], second[2])
    bottom = min(first[3], second[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    first_area = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    second_area = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    smaller = min(first_area, second_area)
    return intersection / smaller if smaller > 0 else 0.0
