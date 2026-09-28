"""Temporal selection of plausible active volleyball players from raw tracks."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Iterable, Optional

from sports.configs.volleyball import Point, VolleyballCourtConfiguration


class TrackSide(str, Enum):
    """Canonical court half occupied by a track."""

    FAR = "far"
    NEAR = "near"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class ActivePlayerSelectorConfig:
    """Explainable weights and temporal thresholds used by the selector."""

    max_players_per_side: int = 6
    net_hysteresis_m: float = 0.5
    side_switch_frames: int = 5
    track_age_normalization_frames: int = 120
    active_grace_frames: int = 8
    active_confirmation_frames: int = 10
    confidence_weight: float = 1.0
    continuity_weight: float = 0.35
    active_history_bonus: float = 0.8
    selection_history_weight: float = 0.2
    court_proximity_weight: float = 0.25

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
        if self.active_grace_frames < 0:
            raise ValueError("active_grace_frames cannot be negative")
        if self.active_confirmation_frames <= 0:
            raise ValueError("active_confirmation_frames must be positive")


@dataclass(frozen=True)
class TrackCandidate:
    """One currently visible, geometrically eligible raw track."""

    track_id: int
    court_point: Point
    confidence: float
    inside_court: bool


@dataclass(frozen=True)
class PlayerSelection:
    """Selector decision for one currently visible candidate."""

    side: TrackSide
    active: bool
    rank: Optional[int]
    score: float


@dataclass
class _TrackMemory:
    side: TrackSide = TrackSide.UNKNOWN
    pending_side: TrackSide = TrackSide.UNKNOWN
    pending_side_frames: int = 0
    observations: int = 0
    selected_frames: int = 0
    last_seen_frame: Optional[int] = None
    last_selected_frame: Optional[int] = None


@dataclass
class ActivePlayerSelector:
    """Keep at most six stable current tracks on each canonical court side.

    Side state belongs to tracker IDs and changes only after consecutive observations
    clearly beyond the net hysteresis band. Active history influences ranking but never
    fabricates a current position for an occluded player.
    """

    court: VolleyballCourtConfiguration
    config: ActivePlayerSelectorConfig = field(default_factory=ActivePlayerSelectorConfig)
    _memory: Dict[int, _TrackMemory] = field(default_factory=dict, init=False)

    def select(
        self, candidates: Iterable[TrackCandidate], frame_index: int,
    ) -> Dict[int, PlayerSelection]:
        candidate_list = list(candidates)
        scored_by_side = {TrackSide.FAR: [], TrackSide.NEAR: []}
        decisions: Dict[int, PlayerSelection] = {}

        for candidate in candidate_list:
            memory = self._memory.setdefault(candidate.track_id, _TrackMemory())
            self._update_side(memory, candidate.court_point[1], frame_index)
            memory.observations += 1
            memory.last_seen_frame = frame_index
            score = self._score(candidate, memory, frame_index)
            if memory.side in scored_by_side:
                scored_by_side[memory.side].append((score, memory.observations, candidate))
            decisions[candidate.track_id] = PlayerSelection(
                side=memory.side, active=False, rank=None, score=score)

        for side, scored in scored_by_side.items():
            scored.sort(key=lambda item: (-item[0], -item[1], item[2].track_id))
            for zero_based_rank, (score, _age, candidate) in enumerate(
                scored[:self.config.max_players_per_side]
            ):
                rank = zero_based_rank + 1
                decisions[candidate.track_id] = PlayerSelection(
                    side=side, active=True, rank=rank, score=score)
                memory = self._memory[candidate.track_id]
                memory.selected_frames += 1
                memory.last_selected_frame = frame_index
        return decisions

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
            and frame_index - memory.last_selected_frame <= self.config.active_grace_frames
        )
        active_stability = min(
            1.0, memory.selected_frames / self.config.active_confirmation_frames)
        proximity = (
            1.0 if candidate.inside_court
            else self._court_proximity(candidate.court_point))
        return (
            self.config.confidence_weight * confidence
            + self.config.continuity_weight * normalized_age
            + self.config.active_history_bonus * float(recently_active) * active_stability
            + self.config.selection_history_weight * normalized_history
            + self.config.court_proximity_weight * proximity
        )

    def _court_proximity(self, point: Point) -> float:
        x, y = point
        if self.court.is_inside_court(point):
            return 1.0
        outside_x = max(-x, 0.0, x - self.court.width)
        outside_y = max(-y, 0.0, y - self.court.length)
        normalized_x = outside_x / max(self.court.side_margin, 1e-9)
        normalized_y = outside_y / max(self.court.baseline_margin, 1e-9)
        return max(0.0, 1.0 - max(normalized_x, normalized_y))
