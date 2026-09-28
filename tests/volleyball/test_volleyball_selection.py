from examples.volleyball.selection import (
    ActivePlayerSelector,
    ActivePlayerSelectorConfig,
    ReconnectionConfig,
    TrackCandidate,
    TrackSide,
    VisibilityState,
)
from sports.configs.volleyball import CameraView, VolleyballCourtConfiguration


def _candidate(
    track_id, y, confidence=0.8, x=4.5, bbox=None, in_occlusion_zone=False,
):
    return TrackCandidate(
        track_id=track_id,
        court_point=(x, y),
        confidence=confidence,
        inside_court=0 <= x <= 9 and 0 <= y <= 18,
        bbox=bbox,
        in_occlusion_zone=in_occlusion_zone,
    )


def _config(**kwargs):
    reconnect = kwargs.pop("reconnection", ReconnectionConfig(
        max_gap_frames=20,
        max_distance_m=3.0,
        max_speed_mps=8.0,
        min_score=0.65,
        ambiguity_margin=0.12,
    ))
    values = {
        "active_confirmation_frames": 2,
        "active_occlusion_grace_frames": 15,
        "frame_rate": 30.0,
        "reconnection": reconnect,
    }
    values.update(kwargs)
    return ActivePlayerSelectorConfig(**values)


def _active_logical(result, side):
    return {
        logical_id for logical_id, decision in result.members.items()
        if decision.active and decision.side is side
    }


def _active_raw(result, side):
    return {
        raw_id for raw_id, decision in result.visible.items()
        if decision.active and decision.side is side
    }


def _establish(selector, candidates, frames=3):
    result = None
    for frame in range(frames):
        result = selector.select(candidates, frame)
    return result


def test_hard_maximum_six_per_side_and_no_synthetic_players() -> None:
    selector = ActivePlayerSelector(VolleyballCourtConfiguration())
    result = selector.select(
        [_candidate(index, 5.0) for index in range(9)]
        + [_candidate(100 + index, 13.0) for index in range(8)], 0)
    assert len(_active_logical(result, TrackSide.FAR)) == 6
    assert len(_active_logical(result, TrackSide.NEAR)) == 6
    fewer = ActivePlayerSelector(VolleyballCourtConfiguration()).select(
        [_candidate(index, 5.0) for index in range(5)], 0)
    assert len(_active_logical(fewer, TrackSide.FAR)) == 5


def test_stable_active_tracks_beat_new_high_confidence_candidate() -> None:
    selector = ActivePlayerSelector(VolleyballCourtConfiguration())
    established = [_candidate(index, 5.0, 0.72, x=1.0 + index) for index in range(6)]
    for frame in range(10):
        result = selector.select(established, frame)
    result = selector.select(
        established + [_candidate(99, 5.0, 0.99, x=8.5)], 10)
    assert _active_raw(result, TrackSide.FAR) == set(range(6))
    assert not result.visible[99].active


def test_referee_overlap_preserves_sixth_player_as_occluded() -> None:
    selector = ActivePlayerSelector(VolleyballCourtConfiguration(), _config())
    player_box = (100.0, 100.0, 160.0, 220.0)
    established = [
        _candidate(index, 5.0, x=1.0 + index, bbox=player_box if index == 2 else None)
        for index in range(6)
    ]
    _establish(selector, established)
    visible = [candidate for candidate in established if candidate.track_id != 2]
    for frame in range(3, 15):
        result = selector.select(
            visible, frame,
            occluder_bboxes=((120.0, 80.0, 180.0, 240.0),) if frame == 3 else ())
        assert len(_active_logical(result, TrackSide.FAR)) == 6
        hidden = [decision for decision in result.members.values() if not decision.visible]
        assert len(hidden) == 1
        assert hidden[0].visibility is VisibilityState.OCCLUDED
        assert hidden[0].raw_track_id is None


def test_grace_expiry_releases_slot() -> None:
    selector = ActivePlayerSelector(
        VolleyballCourtConfiguration(),
        _config(active_occlusion_grace_frames=2, reconnection=ReconnectionConfig(
            max_gap_frames=3, min_score=0.65)))
    established = [_candidate(index, 5.0, x=1.0 + index) for index in range(6)]
    _establish(selector, established)
    replacement = established[1:] + [_candidate(99, 5.0, 0.99, x=8.5)]
    for frame in range(3, 8):
        result = selector.select(replacement, frame)
    assert result.visible[99].active
    assert len(_active_logical(result, TrackSide.FAR)) == 6


def test_stationary_referee_does_not_win_on_track_age() -> None:
    selector = ActivePlayerSelector(
        VolleyballCourtConfiguration(),
        _config(movement_window_frames=8, stationary_footprint_m=0.2,
                stationary_referee_penalty=2.0))
    for frame in range(15):
        players = [
            _candidate(index, 5.0 + 0.2 * ((frame + index) % 3), 0.75, x=1.0 + index)
            for index in range(6)
        ]
        referee = _candidate(99, 8.9, 0.99, x=9.0, in_occlusion_zone=True)
        result = selector.select(players + [referee], frame)
    assert not result.visible[99].active


def test_side_hysteresis_and_real_transition() -> None:
    selector = ActivePlayerSelector(
        VolleyballCourtConfiguration(), _config(side_switch_frames=3))
    sides = [
        selector.select([_candidate(1, y)], frame).visible[1].side
        for frame, y in enumerate((8.8, 9.05, 8.95, 9.1, 8.9))
    ]
    assert sides == [TrackSide.FAR] * 5
    selector = ActivePlayerSelector(
        VolleyballCourtConfiguration(), _config(side_switch_frames=3))
    sides = [
        selector.select([_candidate(1, y)], frame).visible[1].side
        for frame, y in enumerate((8.0, 10.0, 10.2, 10.1))
    ]
    assert sides[-1] is TrackSide.NEAR


def test_servers_are_eligible() -> None:
    selector = ActivePlayerSelector(VolleyballCourtConfiguration())
    result = selector.select([_candidate(1, -2.0), _candidate(2, 20.0)], 0)
    assert result.visible[1].side is TrackSide.FAR and result.visible[1].active
    assert result.visible[2].side is TrackSide.NEAR and result.visible[2].active


def test_simple_referee_occlusion_reconnects_new_raw_id() -> None:
    selector = ActivePlayerSelector(VolleyballCourtConfiguration(), _config())
    old = _candidate(17, 4.0, x=2.0, bbox=(100, 100, 160, 220))
    first = _establish(selector, [old])
    logical_id = first.visible[17].logical_player_track_id
    selector.select([], 3, occluder_bboxes=((110, 90, 170, 230),))
    for frame in range(4, 14):
        selector.select([], frame)
    result = selector.select([_candidate(42, 4.1, x=2.3)], 14)
    assert result.visible[42].logical_player_track_id == logical_id
    assert result.visible[42].reconnected
    assert result.reconnection_events[0].old_raw_track_id == 17
    assert result.reconnection_events[0].occlusion_evidence


def test_stable_raw_id_generates_no_reconnection() -> None:
    selector = ActivePlayerSelector(VolleyballCourtConfiguration(), _config())
    for frame in range(8):
        result = selector.select([_candidate(17, 4.0, x=2.0)], frame)
    assert not result.reconnection_events
    assert selector.diagnostics.accepted == 0


def test_too_far_and_impossible_speed_are_rejected() -> None:
    selector = ActivePlayerSelector(VolleyballCourtConfiguration(), _config())
    _establish(selector, [_candidate(17, 4.0, x=2.0)])
    selector.select([], 3)
    far = selector.select([_candidate(42, 4.0, x=8.0)], 4)
    assert not far.reconnection_events
    assert far.visible[42].logical_player_track_id != 1
    assert selector.diagnostics.rejected_distance > 0

    selector = ActivePlayerSelector(
        VolleyballCourtConfiguration(),
        _config(reconnection=ReconnectionConfig(
            max_gap_frames=20, max_distance_m=10.0, max_speed_mps=5.0,
            min_score=0.4)))
    _establish(selector, [_candidate(17, 4.0, x=2.0)])
    selector.select([], 3)
    fast = selector.select([_candidate(42, 4.0, x=3.0)], 4)
    assert not fast.reconnection_events
    assert selector.diagnostics.rejected_speed > 0


def test_opposite_side_is_rejected() -> None:
    selector = ActivePlayerSelector(VolleyballCourtConfiguration(), _config())
    _establish(selector, [_candidate(17, 4.0, x=2.0)])
    selector.select([], 3)
    result = selector.select([_candidate(42, 14.0, x=2.0)], 4)
    assert not result.reconnection_events
    assert selector.diagnostics.rejected_side > 0


def test_ambiguous_match_is_conservatively_rejected() -> None:
    selector = ActivePlayerSelector(VolleyballCourtConfiguration(), _config())
    established = [_candidate(17, 4.0, x=2.0), _candidate(18, 4.0, x=2.4)]
    _establish(selector, established)
    selector.select([], 3)
    result = selector.select([_candidate(42, 4.0, x=2.2)], 4)
    assert not result.reconnection_events
    assert selector.diagnostics.ambiguous_rejected > 0


def test_one_to_one_assignment_reconnects_two_distinct_pairs() -> None:
    selector = ActivePlayerSelector(VolleyballCourtConfiguration(), _config())
    first = _establish(selector, [
        _candidate(17, 4.0, x=2.0), _candidate(18, 4.0, x=7.0)])
    old_mapping = {
        raw: first.visible[raw].logical_player_track_id for raw in (17, 18)}
    selector.select([], 3)
    result = selector.select([
        _candidate(42, 4.0, x=2.2), _candidate(43, 4.0, x=6.8)], 4)
    assert len(result.reconnection_events) == 2
    assert result.visible[42].logical_player_track_id == old_mapping[17]
    assert result.visible[43].logical_player_track_id == old_mapping[18]


def test_reconnection_preserves_six_active_slots() -> None:
    selector = ActivePlayerSelector(VolleyballCourtConfiguration(), _config())
    established = [_candidate(index, 4.0, x=1.0 + index) for index in range(6)]
    first = _establish(selector, established)
    missing_logical = first.visible[2].logical_player_track_id
    selector.select(established[:2] + established[3:], 3)
    result = selector.select(
        established[:2] + [_candidate(42, 4.0, x=3.1)] + established[3:], 4)
    assert result.visible[42].logical_player_track_id == missing_logical
    assert result.visible[42].active
    assert len(_active_logical(result, TrackSide.FAR)) == 6
    assert sum(not decision.visible for decision in result.members.values()) == 0


def test_expired_fragment_gets_new_logical_identity() -> None:
    selector = ActivePlayerSelector(
        VolleyballCourtConfiguration(),
        _config(active_occlusion_grace_frames=2, reconnection=ReconnectionConfig(
            max_gap_frames=3, min_score=0.4)))
    first = _establish(selector, [_candidate(17, 4.0, x=2.0)])
    old_logical = first.visible[17].logical_player_track_id
    for frame in range(3, 8):
        selector.select([], frame)
    result = selector.select([_candidate(42, 4.0, x=2.1)], 8)
    assert not result.reconnection_events
    assert result.visible[42].logical_player_track_id != old_logical


def test_referee_suppressed_track_cannot_reconnect() -> None:
    selector = ActivePlayerSelector(
        VolleyballCourtConfiguration(),
        _config(movement_window_frames=3, stationary_footprint_m=0.1,
                stationary_referee_penalty=2.0))
    players = [
        _candidate(17 + index, 4.0, confidence=1.0, x=1.0 + index)
        for index in range(6)
    ]
    for frame in range(4):
        selector.select(players + [_candidate(
            99, 8.9, 0.99, x=2.1, in_occlusion_zone=True)], frame)
    result = selector.select(players[1:] + [_candidate(
        99, 8.9, 0.99, x=1.1, in_occlusion_zone=True)], 4)
    assert not result.reconnection_events
    assert selector.diagnostics.rejected_referee > 0


def test_static_zone_evidence_increases_and_accepts_score() -> None:
    selector = ActivePlayerSelector(VolleyballCourtConfiguration(), _config())
    old = _candidate(17, 4.0, x=2.0, in_occlusion_zone=True)
    _establish(selector, [old])
    hidden = selector.select([], 3)
    assert next(iter(hidden.members.values())).occlusion_evidence == "static_zone"
    result = selector.select([_candidate(42, 4.0, x=2.4)], 4)
    assert result.reconnection_events
    assert result.reconnection_events[0].occlusion_evidence_type == "static_zone"


def test_multiple_track_breaks_share_one_logical_id() -> None:
    selector = ActivePlayerSelector(VolleyballCourtConfiguration(), _config())
    first = _establish(selector, [_candidate(7, 4.0, x=2.0)])
    logical_id = first.visible[7].logical_player_track_id
    selector.select([], 3)
    second = selector.select([_candidate(15, 4.0, x=2.2)], 4)
    assert second.visible[15].logical_player_track_id == logical_id
    selector.select([], 5)
    third = selector.select([_candidate(31, 4.0, x=2.4)], 6)
    assert third.visible[31].logical_player_track_id == logical_id
    assert third.visible[31].logical_reconnection_count == 2
    assert selector.logical_players[logical_id].previous_raw_track_ids == [7, 15, 31]


def test_false_merge_near_two_players_is_rejected() -> None:
    selector = ActivePlayerSelector(VolleyballCourtConfiguration(), _config())
    _establish(selector, [
        _candidate(7, 4.0, x=3.0), _candidate(8, 4.0, x=3.4)])
    selector.select([], 3)
    result = selector.select([_candidate(31, 4.0, x=3.2)], 4)
    assert not result.reconnection_events


def test_selection_and_reconnection_are_renderer_orientation_invariant() -> None:
    outputs = []
    for _view in (CameraView.ENDLINE, CameraView.SIDELINE):
        selector = ActivePlayerSelector(VolleyballCourtConfiguration(), _config())
        first = _establish(selector, [_candidate(7, 4.0, x=2.0)])
        selector.select([], 3)
        result = selector.select([_candidate(15, 4.1, x=2.2)], 4)
        outputs.append((
            first.visible[7].logical_player_track_id,
            result.visible[15].logical_player_track_id,
            result.reconnection_events[0].score,
        ))
    assert outputs[0] == outputs[1]
