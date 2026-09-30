"""Temporal active-player selection and conservative raw-track reconnection."""

from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Deque, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

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


class PositionSource(str, Enum):
    """Provenance of the tactical ground-plane position."""

    OBSERVED = "observed"
    SMOOTHED = "smoothed"
    PREDICTED = "predicted"
    HELD = "held"


@dataclass(frozen=True)
class ReconnectionConfig:
    """Thresholds and normalized score weights for fragment reconnection."""

    max_gap_frames: int = 45
    max_distance_m: float = 3.0
    max_speed_mps: float = 8.0
    min_score: float = 0.68
    ambiguity_margin: float = 0.12
    position_weight: float = 0.25
    prediction_weight: float = 0.25
    time_weight: float = 0.15
    motion_weight: float = 0.10
    occlusion_weight: float = 0.15
    incumbent_weight: float = 0.10
    motion_history_size: int = 5

    def __post_init__(self) -> None:
        if self.max_gap_frames <= 0:
            raise ValueError("reconnect max_gap_frames must be positive")
        if self.max_distance_m <= 0 or self.max_speed_mps <= 0:
            raise ValueError("reconnect distance and speed must be positive")
        if not 0.0 <= self.min_score <= 1.0:
            raise ValueError("reconnect min_score must be between 0 and 1")
        if not 0.0 <= self.ambiguity_margin <= 1.0:
            raise ValueError("reconnect ambiguity_margin must be between 0 and 1")
        if self.motion_history_size < 2:
            raise ValueError("reconnect motion_history_size must be at least 2")
        weights = (
            self.position_weight, self.prediction_weight, self.time_weight,
            self.motion_weight, self.occlusion_weight, self.incumbent_weight,
        )
        if any(weight < 0 for weight in weights) or not np.isclose(sum(weights), 1.0):
            raise ValueError("reconnect score weights must be non-negative and sum to 1")


@dataclass(frozen=True)
class ActivePlayerSelectorConfig:
    """Explainable weights and temporal thresholds used by the selector."""

    max_players_per_side: int = 6
    net_hysteresis_m: float = 0.5
    side_switch_frames: int = 5
    track_age_normalization_frames: int = 120
    active_occlusion_grace_frames: int = 30
    active_confirmation_frames: int = 10
    frame_rate: float = 30.0
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
    position_max_speed_mps: float = 10.0
    position_median_window: int = 3
    position_smoothing_alpha: float = 0.45
    position_reset_gap_frames: int = 45
    player_overlap_distance_m: float = 1.5
    reconnection: ReconnectionConfig = field(default_factory=ReconnectionConfig)

    def __post_init__(self) -> None:
        if not 1 <= self.max_players_per_side <= 6:
            raise ValueError(
                "max_players_per_side must be between 1 and volleyball's hard maximum of 6")
        if self.net_hysteresis_m < 0 or self.side_switch_frames <= 0:
            raise ValueError("side hysteresis must be non-negative and frame count positive")
        if self.track_age_normalization_frames <= 0 or self.frame_rate <= 0:
            raise ValueError("normalization frames and frame_rate must be positive")
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
        if self.position_max_speed_mps <= 0 or self.position_reset_gap_frames <= 0:
            raise ValueError("position speed and reset gap must be positive")
        if self.position_median_window <= 0:
            raise ValueError("position_median_window must be positive")
        if not 0.0 < self.position_smoothing_alpha <= 1.0:
            raise ValueError("position_smoothing_alpha must be in (0, 1]")
        if self.player_overlap_distance_m <= 0:
            raise ValueError("player_overlap_distance_m must be positive")


@dataclass(frozen=True)
class TrackCandidate:
    """One currently visible, geometrically eligible raw track."""

    track_id: int
    court_point: Point
    confidence: float
    inside_court: bool
    bbox: Optional[BBox] = None
    in_occlusion_zone: bool = False
    appearance_score: Optional[float] = None


@dataclass(frozen=True)
class ReconnectionEvent:
    """An accepted raw-fragment repair, suitable for JSONL serialization."""

    frame: int
    logical_player_track_id: int
    old_raw_track_id: int
    new_raw_track_id: int
    gap_frames: int
    side: TrackSide
    distance_m: float
    predicted_distance_m: float
    score: float
    occlusion_evidence: bool
    occlusion_evidence_type: Optional[str]

    def to_dict(self) -> Dict[str, object]:
        return {
            "frame": self.frame,
            "logical_player_track_id": self.logical_player_track_id,
            "old_raw_track_id": self.old_raw_track_id,
            "new_raw_track_id": self.new_raw_track_id,
            "gap_frames": self.gap_frames,
            "side": self.side.value,
            "distance_m": self.distance_m,
            "predicted_distance_m": self.predicted_distance_m,
            "score": self.score,
            "occlusion_evidence": self.occlusion_evidence,
            "occlusion_evidence_type": self.occlusion_evidence_type,
        }


@dataclass
class ReconnectionDiagnostics:
    """Counters describing conservative gate and ambiguity outcomes."""

    accepted: int = 0
    ambiguous_rejected: int = 0
    rejected_side: int = 0
    rejected_distance: int = 0
    rejected_speed: int = 0
    rejected_score: int = 0
    rejected_referee: int = 0


@dataclass(frozen=True)
class PlayerSelection:
    """Decision for a visible raw track or a reserved logical member."""

    side: TrackSide
    active: bool
    rank: Optional[int]
    score: float
    raw_track_id: Optional[int]
    logical_player_track_id: Optional[int]
    last_known_raw_track_id: Optional[int] = None
    visibility: VisibilityState = VisibilityState.VISIBLE
    position_observed: bool = True
    occlusion_age: int = 0
    last_known_court_point: Optional[Point] = None
    last_known_bbox: Optional[BBox] = None
    occlusion_evidence: Optional[str] = None
    reconnected: bool = False
    reconnection_from_raw_track_id: Optional[int] = None
    reconnection_score: Optional[float] = None
    reconnection_gap_frames: Optional[int] = None
    logical_reconnection_count: int = 0
    raw_court_point: Optional[Point] = None
    court_point: Optional[Point] = None
    position_used: bool = True
    position_outlier: bool = False
    position_source: PositionSource = PositionSource.OBSERVED

    @property
    def visible(self) -> bool:
        return self.visibility is VisibilityState.VISIBLE

    @property
    def occluded(self) -> bool:
        return self.visibility is VisibilityState.OCCLUDED


@dataclass(frozen=True)
class SelectionResult:
    """Frame decisions indexed separately by immutable raw and logical identity."""

    visible: Mapping[int, PlayerSelection]
    members: Mapping[int, PlayerSelection]
    reconnection_events: Tuple[ReconnectionEvent, ...] = ()


@dataclass
class LogicalPlayerState:
    """Persistent identity independent from any one ByteTrack fragment."""

    logical_player_track_id: int
    current_raw_track_id: Optional[int]
    previous_raw_track_ids: List[int]
    side: TrackSide
    visibility_state: VisibilityState
    active: bool
    first_seen_frame: int
    last_seen_frame: int
    last_observed_frame: int
    last_observed_court_position: Point
    last_bbox: Optional[BBox]
    last_confidence: float
    selected_frames: int = 0
    last_selected_frame: Optional[int] = None
    occlusion_evidence: Optional[str] = None
    last_seen_in_occlusion_zone: bool = False
    reconnection_count: int = 0
    was_ever_active: bool = False
    observed_positions: Deque[Tuple[int, Point]] = field(default_factory=deque)
    raw_court_position: Optional[Point] = None
    stabilized_court_position: Optional[Point] = None
    last_position_frame: Optional[int] = None
    position_outlier: bool = False
    position_source: PositionSource = PositionSource.OBSERVED
    recent_raw_positions: Deque[Point] = field(default_factory=deque)

    @property
    def all_raw_track_ids(self) -> Tuple[int, ...]:
        ids = [*self.previous_raw_track_ids]
        if self.current_raw_track_id is not None:
            ids.append(self.current_raw_track_id)
        return tuple(dict.fromkeys(ids))


@dataclass
class _RawTrackMemory:
    side: TrackSide = TrackSide.UNKNOWN
    pending_side: TrackSide = TrackSide.UNKNOWN
    pending_side_frames: int = 0
    observations: int = 0
    selected_frames: int = 0
    last_seen_frame: Optional[int] = None
    last_selected_frame: Optional[int] = None
    positions: Deque[Point] = field(default_factory=deque)


@dataclass(frozen=True)
class _PairScore:
    raw_track_id: int
    logical_id: int
    score: float
    gap_frames: int
    distance_m: float
    predicted_distance_m: float


@dataclass
class ActivePlayerSelector:
    """Build stable logical identities above raw ByteTrack fragments."""

    court: VolleyballCourtConfiguration
    config: ActivePlayerSelectorConfig = field(default_factory=ActivePlayerSelectorConfig)
    _raw_memory: Dict[int, _RawTrackMemory] = field(default_factory=dict, init=False)
    _logical_players: Dict[int, LogicalPlayerState] = field(default_factory=dict, init=False)
    _raw_to_logical: Dict[int, int] = field(default_factory=dict, init=False)
    _next_logical_id: int = field(default=1, init=False)
    diagnostics: ReconnectionDiagnostics = field(
        default_factory=ReconnectionDiagnostics, init=False)
    events: List[ReconnectionEvent] = field(default_factory=list, init=False)

    @property
    def logical_players(self) -> Mapping[int, LogicalPlayerState]:
        return self._logical_players

    def select(
        self,
        candidates: Iterable[TrackCandidate],
        frame_index: int,
        occluder_bboxes: Sequence[BBox] = (),
    ) -> SelectionResult:
        candidate_list = list(candidates)
        candidate_by_raw = {candidate.track_id: candidate for candidate in candidate_list}
        if len(candidate_by_raw) != len(candidate_list):
            raise ValueError("raw track IDs must be unique within a frame")
        visible_raw_ids = set(candidate_by_raw)
        raw_scores: Dict[int, float] = {}
        referee_suppressed: Set[int] = set()

        for candidate in candidate_list:
            memory = self._raw_memory.setdefault(candidate.track_id, _RawTrackMemory())
            self._update_side(memory, candidate.court_point[1], frame_index)
            memory.observations += 1
            memory.last_seen_frame = frame_index
            memory.positions.append(candidate.court_point)
            while len(memory.positions) > self.config.movement_window_frames:
                memory.positions.popleft()
            penalty = self._stationary_referee_penalty(candidate, memory)
            if penalty > 0:
                referee_suppressed.add(candidate.track_id)
            raw_scores[candidate.track_id] = self._score(candidate, memory, frame_index, penalty)

        self._mark_missing_players(
            frame_index, visible_raw_ids, candidate_by_raw, occluder_bboxes)

        # A previously known raw ID resuming after a brief tracker omission is not a
        # reconnection event: ByteTrack preserved its own identity.
        for raw_id, candidate in candidate_by_raw.items():
            logical_id = self._raw_to_logical.get(raw_id)
            if logical_id is not None:
                self._observe_logical(
                    self._logical_players[logical_id], candidate, frame_index,
                    reconnected=False)

        new_raw_ids = [
            raw_id for raw_id in visible_raw_ids if raw_id not in self._raw_to_logical]
        frame_events = self._reconnect(
            new_raw_ids, candidate_by_raw, referee_suppressed, frame_index)

        visible_decisions: Dict[int, PlayerSelection] = {}
        member_decisions: Dict[int, PlayerSelection] = {}
        selected_logical_ids: Set[int] = set()

        for side in (TrackSide.FAR, TrackSide.NEAR):
            reserved: List[Tuple[float, int]] = []
            visible_pool: List[Tuple[float, int]] = []
            for logical_id, player in self._logical_players.items():
                raw_id = player.current_raw_track_id
                is_visible = raw_id in visible_raw_ids if raw_id is not None else False
                if player.side is not side:
                    continue
                if player.active and not is_visible and self._can_reserve(player, frame_index):
                    age = frame_index - player.last_observed_frame
                    reserved.append((self._incumbent_score(player, age), logical_id))
                elif is_visible and raw_id is not None and raw_id not in referee_suppressed:
                    score = raw_scores[raw_id]
                    if player.active:
                        # A reappearing incumbent occupies its already-reserved slot;
                        # the fresh raw fragment must not lose solely because its raw
                        # track-age score restarted at zero.
                        score = max(score, self._incumbent_score(player, 0))
                    visible_pool.append((score, raw_id))

            assigned_raw_ids = set(self._raw_to_logical)
            for raw_id, candidate in candidate_by_raw.items():
                if (
                    raw_id not in assigned_raw_ids
                    and self._raw_memory[raw_id].side is side
                    and raw_id not in referee_suppressed
                ):
                    visible_pool.append((raw_scores[raw_id], raw_id))

            reserved.sort(key=lambda item: (-item[0], item[1]))
            rank = 1
            for score, logical_id in reserved[:self.config.max_players_per_side]:
                selected_logical_ids.add(logical_id)
                rank = self._select_member(
                    logical_id, score, rank, frame_index, candidate_by_raw,
                    visible_decisions, member_decisions)

            remaining = self.config.max_players_per_side - len(
                [logical_id for logical_id in selected_logical_ids
                 if self._logical_players[logical_id].side is side])
            visible_pool.sort(key=lambda item: (-item[0], item[1]))
            for score, raw_id in visible_pool[:remaining]:
                logical_id = self._raw_to_logical.get(raw_id)
                if logical_id is None:
                    logical_id = self._create_logical(
                        candidate_by_raw[raw_id], frame_index)
                if logical_id in selected_logical_ids:
                    continue
                selected_logical_ids.add(logical_id)
                rank = self._select_member(
                    logical_id, score, rank, frame_index, candidate_by_raw,
                    visible_decisions, member_decisions)

        for logical_id, player in self._logical_players.items():
            player.active = logical_id in selected_logical_ids
            if player.active:
                player.was_ever_active = True

        for candidate in candidate_list:
            if candidate.track_id in visible_decisions:
                continue
            logical_id = self._raw_to_logical.get(candidate.track_id)
            player = self._logical_players.get(logical_id) if logical_id is not None else None
            event = next(
                (item for item in frame_events
                 if item.logical_player_track_id == logical_id
                 and item.new_raw_track_id == candidate.track_id),
                None,
            )
            visible_decisions[candidate.track_id] = PlayerSelection(
                side=self._raw_memory[candidate.track_id].side,
                active=False,
                rank=None,
                score=raw_scores[candidate.track_id],
                raw_track_id=candidate.track_id,
                logical_player_track_id=logical_id,
                last_known_raw_track_id=candidate.track_id,
                last_known_court_point=candidate.court_point,
                last_known_bbox=candidate.bbox,
                reconnected=event is not None,
                reconnection_from_raw_track_id=(event.old_raw_track_id if event else None),
                reconnection_score=(event.score if event else None),
                reconnection_gap_frames=(event.gap_frames if event else None),
                logical_reconnection_count=player.reconnection_count if player else 0,
                raw_court_point=candidate.court_point,
                court_point=(
                    player.stabilized_court_position if player is not None
                    else candidate.court_point),
                position_used=not player.position_outlier if player is not None else True,
                position_outlier=player.position_outlier if player is not None else False,
                position_source=(
                    player.position_source if player is not None
                    else PositionSource.OBSERVED),
            )

        return SelectionResult(
            visible=visible_decisions,
            members=member_decisions,
            reconnection_events=tuple(frame_events),
        )

    def _reconnect(
        self,
        new_raw_ids: Sequence[int],
        candidates: Mapping[int, TrackCandidate],
        referee_suppressed: Set[int],
        frame_index: int,
    ) -> List[ReconnectionEvent]:
        pairs: List[_PairScore] = []
        for raw_id in new_raw_ids:
            if raw_id in referee_suppressed:
                self.diagnostics.rejected_referee += 1
                continue
            candidate = candidates[raw_id]
            side = self._raw_memory[raw_id].side
            for player in self._logical_players.values():
                if player.current_raw_track_id is not None or not player.was_ever_active:
                    continue
                gap = frame_index - player.last_observed_frame
                if gap < 1 or gap > self.config.reconnection.max_gap_frames:
                    continue
                if side is not player.side:
                    self.diagnostics.rejected_side += 1
                    continue
                distance = _distance(candidate.court_point, player.last_observed_court_position)
                predicted = self._predict_position(player, frame_index)
                predicted_distance = _distance(candidate.court_point, predicted)
                if min(distance, predicted_distance) > self.config.reconnection.max_distance_m:
                    self.diagnostics.rejected_distance += 1
                    continue
                elapsed_s = gap / self.config.frame_rate
                if distance / max(elapsed_s, 1e-9) > self.config.reconnection.max_speed_mps:
                    self.diagnostics.rejected_speed += 1
                    continue
                score = self._reconnection_score(
                    player, candidate, gap, distance, predicted_distance)
                if score < self.config.reconnection.min_score:
                    self.diagnostics.rejected_score += 1
                    continue
                pairs.append(_PairScore(
                    raw_id, player.logical_player_track_id, score, gap,
                    distance, predicted_distance))

        ambiguous_pairs: Set[Tuple[int, int]] = set()
        for group_key in ("raw", "logical"):
            keys = {pair.raw_track_id if group_key == "raw" else pair.logical_id for pair in pairs}
            for key in keys:
                group = [
                    pair for pair in pairs
                    if (pair.raw_track_id if group_key == "raw" else pair.logical_id) == key]
                group.sort(key=lambda pair: (-pair.score, pair.logical_id, pair.raw_track_id))
                if (
                    len(group) > 1
                    and group[0].score - group[1].score
                    < self.config.reconnection.ambiguity_margin
                ):
                    ambiguous_pairs.update(
                        (pair.raw_track_id, pair.logical_id) for pair in group)
                    self.diagnostics.ambiguous_rejected += 1

        accepted_raw: Set[int] = set()
        accepted_logical: Set[int] = set()
        events: List[ReconnectionEvent] = []
        for pair in sorted(
            pairs, key=lambda item: (-item.score, item.raw_track_id, item.logical_id)
        ):
            if (pair.raw_track_id, pair.logical_id) in ambiguous_pairs:
                continue
            if pair.raw_track_id in accepted_raw or pair.logical_id in accepted_logical:
                continue
            player = self._logical_players[pair.logical_id]
            old_raw = player.previous_raw_track_ids[-1]
            candidate = candidates[pair.raw_track_id]
            occlusion_evidence = player.occlusion_evidence
            if occlusion_evidence is None and candidate.in_occlusion_zone:
                occlusion_evidence = "static_zone_reappearance"
            self._raw_to_logical[pair.raw_track_id] = pair.logical_id
            player.reconnection_count += 1
            self._observe_logical(player, candidate, frame_index, reconnected=True)
            event = ReconnectionEvent(
                frame=frame_index,
                logical_player_track_id=pair.logical_id,
                old_raw_track_id=old_raw,
                new_raw_track_id=pair.raw_track_id,
                gap_frames=pair.gap_frames,
                side=player.side,
                distance_m=pair.distance_m,
                predicted_distance_m=pair.predicted_distance_m,
                score=pair.score,
                occlusion_evidence=occlusion_evidence is not None,
                occlusion_evidence_type=occlusion_evidence,
            )
            events.append(event)
            self.events.append(event)
            self.diagnostics.accepted += 1
            accepted_raw.add(pair.raw_track_id)
            accepted_logical.add(pair.logical_id)
        return events

    def _create_logical(
        self, candidate: TrackCandidate, frame_index: int,
    ) -> int:
        logical_id = self._next_logical_id
        self._next_logical_id += 1
        state = LogicalPlayerState(
            logical_player_track_id=logical_id,
            current_raw_track_id=candidate.track_id,
            previous_raw_track_ids=[candidate.track_id],
            side=self._raw_memory[candidate.track_id].side,
            visibility_state=VisibilityState.VISIBLE,
            active=False,
            first_seen_frame=frame_index,
            last_seen_frame=frame_index,
            last_observed_frame=frame_index,
            last_observed_court_position=candidate.court_point,
            last_bbox=candidate.bbox,
            last_confidence=candidate.confidence,
            last_seen_in_occlusion_zone=candidate.in_occlusion_zone,
            observed_positions=deque(
                [(frame_index, candidate.court_point)],
                maxlen=self.config.reconnection.motion_history_size),
            raw_court_position=candidate.court_point,
            stabilized_court_position=candidate.court_point,
            last_position_frame=frame_index,
            recent_raw_positions=deque(
                [candidate.court_point], maxlen=self.config.position_median_window),
        )
        self._logical_players[logical_id] = state
        self._raw_to_logical[candidate.track_id] = logical_id
        return logical_id

    def _observe_logical(
        self,
        player: LogicalPlayerState,
        candidate: TrackCandidate,
        frame_index: int,
        reconnected: bool,
    ) -> None:
        self._update_ground_position(player, candidate.court_point, frame_index)
        if reconnected and candidate.track_id not in player.previous_raw_track_ids:
            player.previous_raw_track_ids.append(candidate.track_id)
        player.current_raw_track_id = candidate.track_id
        player.side = self._raw_memory[candidate.track_id].side
        player.visibility_state = VisibilityState.VISIBLE
        player.last_seen_frame = frame_index
        player.last_observed_frame = frame_index
        player.last_observed_court_position = (
            player.stabilized_court_position or candidate.court_point)
        player.last_bbox = candidate.bbox
        player.last_confidence = candidate.confidence
        player.last_seen_in_occlusion_zone = candidate.in_occlusion_zone
        player.observed_positions.append((
            frame_index, player.stabilized_court_position or candidate.court_point))
        player.occlusion_evidence = None

    def _update_ground_position(
        self,
        player: LogicalPlayerState,
        raw_point: Point,
        frame_index: int,
    ) -> None:
        """Reject impossible floor motion, then median/EMA-filter accepted metres."""
        player.raw_court_position = raw_point
        previous = player.stabilized_court_position
        previous_frame = player.last_position_frame
        if previous is None or previous_frame is None:
            player.stabilized_court_position = raw_point
            player.last_position_frame = frame_index
            player.position_outlier = False
            player.position_source = PositionSource.OBSERVED
            player.recent_raw_positions.clear()
            player.recent_raw_positions.append(raw_point)
            return

        gap = frame_index - previous_frame
        if gap > self.config.position_reset_gap_frames:
            player.stabilized_court_position = raw_point
            player.last_position_frame = frame_index
            player.position_outlier = False
            player.position_source = PositionSource.OBSERVED
            player.recent_raw_positions.clear()
            player.recent_raw_positions.append(raw_point)
            return

        elapsed_s = max(gap / self.config.frame_rate, 1.0 / self.config.frame_rate)
        implied_speed = _distance(raw_point, previous) / elapsed_s
        if implied_speed > self.config.position_max_speed_mps:
            predicted = self._predict_stabilized_position(player, frame_index)
            player.stabilized_court_position = predicted
            player.position_outlier = True
            player.position_source = (
                PositionSource.PREDICTED
                if predicted != previous else PositionSource.HELD)
            return

        player.recent_raw_positions.append(raw_point)
        median = np.median(
            np.asarray(player.recent_raw_positions, dtype=np.float64), axis=0)
        alpha = self.config.position_smoothing_alpha
        stabilized = (
            (1.0 - alpha) * np.asarray(previous, dtype=np.float64)
            + alpha * median
        )
        player.stabilized_court_position = (
            float(stabilized[0]), float(stabilized[1]))
        player.last_position_frame = frame_index
        player.position_outlier = False
        player.position_source = PositionSource.SMOOTHED

    def _predict_stabilized_position(
        self, player: LogicalPlayerState, frame_index: int,
    ) -> Point:
        history = list(player.observed_positions)
        previous = player.stabilized_court_position
        if previous is None or len(history) < 2:
            return previous or player.last_observed_court_position
        (first_frame, first), (second_frame, second) = history[-2:]
        elapsed = second_frame - first_frame
        if elapsed <= 0:
            return previous
        velocity = (
            (second[0] - first[0]) / elapsed,
            (second[1] - first[1]) / elapsed,
        )
        prediction_frames = min(
            frame_index - second_frame,
            max(1, int(round(0.25 * self.config.frame_rate))),
        )
        damping = 0.5
        return (
            previous[0] + velocity[0] * prediction_frames * damping,
            previous[1] + velocity[1] * prediction_frames * damping,
        )

    def _mark_missing_players(
        self,
        frame_index: int,
        visible_raw_ids: Set[int],
        candidates: Mapping[int, TrackCandidate],
        occluder_bboxes: Sequence[BBox],
    ) -> None:
        for player in self._logical_players.values():
            raw_id = player.current_raw_track_id
            if raw_id is None or raw_id in visible_raw_ids:
                continue
            player_overlap = False
            if player.active and player.last_bbox is not None:
                for candidate in candidates.values():
                    other_logical_id = self._raw_to_logical.get(candidate.track_id)
                    other = self._logical_players.get(other_logical_id)
                    if (
                        other is None
                        or other.logical_player_track_id == player.logical_player_track_id
                        or not other.active
                        or other.side is not player.side
                        or candidate.bbox is None
                    ):
                        continue
                    if (
                        _distance(
                            player.stabilized_court_position
                            or player.last_observed_court_position,
                            candidate.court_point,
                        ) <= self.config.player_overlap_distance_m
                        and _bbox_overlap_fraction(player.last_bbox, candidate.bbox)
                        >= self.config.dynamic_overlap_threshold
                    ):
                        player_overlap = True
                        break
            if player_overlap:
                player.occlusion_evidence = "player_overlap"
            elif player.last_seen_in_occlusion_zone:
                player.occlusion_evidence = "static_zone"
            elif player.last_bbox is not None and any(
                _bbox_overlap_fraction(player.last_bbox, box)
                >= self.config.dynamic_overlap_threshold
                for box in occluder_bboxes
            ):
                player.occlusion_evidence = "dynamic_person_overlap"
            player.current_raw_track_id = None
            age = frame_index - player.last_observed_frame
            player.visibility_state = (
                VisibilityState.OCCLUDED
                if player.occlusion_evidence is not None
                else VisibilityState.LOST)
            if age > self.config.reconnection.max_gap_frames:
                player.visibility_state = VisibilityState.LOST

    def _select_member(
        self,
        logical_id: int,
        score: float,
        rank: int,
        frame_index: int,
        candidates: Mapping[int, TrackCandidate],
        visible: Dict[int, PlayerSelection],
        members: Dict[int, PlayerSelection],
    ) -> int:
        player = self._logical_players[logical_id]
        raw_id = player.current_raw_track_id
        candidate = candidates.get(raw_id) if raw_id is not None else None
        event = next(
            (item for item in reversed(self.events)
             if item.frame == frame_index and item.logical_player_track_id == logical_id),
            None,
        )
        observed = candidate is not None
        age = 0 if observed else frame_index - player.last_observed_frame
        decision = PlayerSelection(
            side=player.side,
            active=True,
            rank=rank,
            score=score,
            raw_track_id=raw_id if observed else None,
            logical_player_track_id=logical_id,
            last_known_raw_track_id=player.previous_raw_track_ids[-1],
            visibility=VisibilityState.VISIBLE if observed else player.visibility_state,
            position_observed=observed,
            occlusion_age=age,
            last_known_court_point=player.last_observed_court_position,
            last_known_bbox=player.last_bbox,
            occlusion_evidence=player.occlusion_evidence,
            reconnected=event is not None,
            reconnection_from_raw_track_id=(event.old_raw_track_id if event else None),
            reconnection_score=(event.score if event else None),
            reconnection_gap_frames=(event.gap_frames if event else None),
            logical_reconnection_count=player.reconnection_count,
            raw_court_point=(candidate.court_point if candidate is not None else None),
            court_point=player.stabilized_court_position,
            position_used=(candidate is not None and not player.position_outlier),
            position_outlier=(candidate is not None and player.position_outlier),
            position_source=(
                player.position_source if candidate is not None else PositionSource.HELD),
        )
        members[logical_id] = decision
        if observed and raw_id is not None:
            visible[raw_id] = decision
            self._raw_memory[raw_id].selected_frames += 1
            self._raw_memory[raw_id].last_selected_frame = frame_index
        player.selected_frames += 1
        player.last_selected_frame = frame_index
        player.active = True
        return rank + 1

    def _can_reserve(self, player: LogicalPlayerState, frame_index: int) -> bool:
        if (
            not player.active
            or player.last_selected_frame is None
            or player.selected_frames < self.config.active_confirmation_frames
        ):
            return False
        missing_frames = frame_index - player.last_observed_frame
        return 1 <= missing_frames <= self.config.active_occlusion_grace_frames

    def _update_side(
        self, memory: _RawTrackMemory, court_y: float, frame_index: int,
    ) -> None:
        center = self.court.center_line_y
        if memory.side is TrackSide.UNKNOWN:
            memory.side = TrackSide.FAR if court_y < center else TrackSide.NEAR
            return
        if memory.last_seen_frame is not None and frame_index != memory.last_seen_frame + 1:
            memory.pending_side = TrackSide.UNKNOWN
            memory.pending_side_frames = 0
        proposed = TrackSide.UNKNOWN
        if memory.side is TrackSide.FAR and court_y > center + self.config.net_hysteresis_m:
            proposed = TrackSide.NEAR
        elif memory.side is TrackSide.NEAR and court_y < center - self.config.net_hysteresis_m:
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
        self,
        candidate: TrackCandidate,
        memory: _RawTrackMemory,
        frame_index: int,
        stationary_penalty: float,
    ) -> float:
        confidence = max(0.0, min(1.0, candidate.confidence))
        normalized_age = min(
            1.0, memory.observations / self.config.track_age_normalization_frames)
        normalized_history = min(
            1.0, memory.selected_frames / self.config.track_age_normalization_frames)
        recently_active = (
            memory.last_selected_frame is not None
            and frame_index - memory.last_selected_frame
            <= self.config.active_occlusion_grace_frames)
        active_stability = min(
            1.0, memory.selected_frames / self.config.active_confirmation_frames)
        proximity = 1.0 if candidate.inside_court else self._court_proximity(candidate.court_point)
        return (
            self.config.confidence_weight * confidence
            + self.config.continuity_weight * normalized_age
            + self.config.active_history_bonus * float(recently_active) * active_stability
            + self.config.selection_history_weight * normalized_history
            + self.config.court_proximity_weight * proximity
            - stationary_penalty)

    def _stationary_referee_penalty(
        self, candidate: TrackCandidate, memory: _RawTrackMemory,
    ) -> float:
        if len(memory.positions) < self.config.movement_window_frames:
            return 0.0
        positions = np.asarray(memory.positions, dtype=np.float32)
        center = np.median(positions, axis=0)
        footprint = float(np.percentile(np.linalg.norm(positions - center, axis=1), 90))
        suspicious_location = (
            candidate.in_occlusion_zone
            or abs(candidate.court_point[1] - self.court.center_line_y)
            <= self.config.net_referee_band_m)
        if suspicious_location and footprint <= self.config.stationary_footprint_m:
            return self.config.stationary_referee_penalty
        return 0.0

    def _incumbent_score(self, player: LogicalPlayerState, missing_frames: int) -> float:
        history = min(
            1.0, player.selected_frames / self.config.track_age_normalization_frames)
        grace = max(1, self.config.active_occlusion_grace_frames)
        recency = max(0.0, 1.0 - (missing_frames - 1) / grace)
        return self.config.active_history_bonus + self.config.selection_history_weight * history + recency

    def _predict_position(
        self, player: LogicalPlayerState, frame_index: int,
    ) -> Point:
        history = list(player.observed_positions)
        if len(history) < 2:
            return player.last_observed_court_position
        velocities = []
        for (first_frame, first), (second_frame, second) in zip(history, history[1:]):
            elapsed = second_frame - first_frame
            if elapsed > 0:
                velocities.append((
                    (second[0] - first[0]) / elapsed,
                    (second[1] - first[1]) / elapsed,
                ))
        if not velocities:
            return player.last_observed_court_position
        velocity = np.median(np.asarray(velocities, dtype=np.float64), axis=0)
        gap = frame_index - player.last_observed_frame
        damping = max(0.25, 1.0 - gap / self.config.reconnection.max_gap_frames)
        return (
            player.last_observed_court_position[0] + float(velocity[0]) * gap * damping,
            player.last_observed_court_position[1] + float(velocity[1]) * gap * damping,
        )

    def _reconnection_score(
        self,
        player: LogicalPlayerState,
        candidate: TrackCandidate,
        gap: int,
        distance: float,
        predicted_distance: float,
    ) -> float:
        config = self.config.reconnection
        position_score = max(0.0, 1.0 - distance / config.max_distance_m)
        prediction_score = max(0.0, 1.0 - predicted_distance / config.max_distance_m)
        time_score = max(0.0, 1.0 - (gap - 1) / config.max_gap_frames)
        speed = distance / max(gap / self.config.frame_rate, 1e-9)
        motion_score = max(0.0, 1.0 - speed / config.max_speed_mps)
        occlusion_score = float(
            player.occlusion_evidence is not None or candidate.in_occlusion_zone)
        incumbent_score = float(player.active)
        # ``candidate.appearance_score`` is deliberately unused until a trusted
        # appearance provider is introduced; its typed seam prevents identity
        # lifecycle code from needing a later redesign.
        return (
            config.position_weight * position_score
            + config.prediction_weight * prediction_score
            + config.time_weight * time_score
            + config.motion_weight * motion_score
            + config.occlusion_weight * occlusion_score
            + config.incumbent_weight * incumbent_score)

    def _court_proximity(self, point: Point) -> float:
        x, y = point
        if self.court.is_inside_court(point):
            return 1.0
        outside_x = max(-x, 0.0, x - self.court.width)
        outside_y = max(-y, 0.0, y - self.court.length)
        normalized_x = outside_x / max(self.court.side_margin, 1e-9)
        normalized_y = outside_y / max(self.court.baseline_margin, 1e-9)
        return max(0.0, 1.0 - max(normalized_x, normalized_y))


def _distance(first: Point, second: Point) -> float:
    return float(np.linalg.norm(np.asarray(first) - np.asarray(second)))


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
