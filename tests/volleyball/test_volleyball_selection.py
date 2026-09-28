from examples.volleyball.selection import (
    ActivePlayerSelector,
    ActivePlayerSelectorConfig,
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


def _active(decisions, side):
    return [
        track_id for track_id, decision in decisions.items()
        if decision.active and decision.side is side
    ]


def test_hard_maximum_six_per_side_and_no_synthetic_players() -> None:
    selector = ActivePlayerSelector(VolleyballCourtConfiguration())
    decisions = selector.select(
        [_candidate(index, 5.0) for index in range(9)]
        + [_candidate(100 + index, 13.0) for index in range(8)],
        frame_index=0,
    )
    assert len(_active(decisions, TrackSide.FAR)) == 6
    assert len(_active(decisions, TrackSide.NEAR)) == 6

    fewer = ActivePlayerSelector(VolleyballCourtConfiguration()).select(
        [_candidate(index, 5.0) for index in range(5)], frame_index=0)
    assert len(_active(fewer, TrackSide.FAR)) == 5


def test_stable_active_tracks_beat_new_high_confidence_candidate() -> None:
    selector = ActivePlayerSelector(VolleyballCourtConfiguration())
    established = [_candidate(index, 5.0, confidence=0.72) for index in range(6)]
    for frame_index in range(10):
        decisions = selector.select(established, frame_index)
        assert set(_active(decisions, TrackSide.FAR)) == set(range(6))

    decisions = selector.select(
        established + [_candidate(99, 5.0, confidence=0.99)], frame_index=10)
    assert set(_active(decisions, TrackSide.FAR)) == set(range(6))
    assert not decisions[99].active


def test_recent_established_track_reserves_slot_during_short_dropout() -> None:
    selector = ActivePlayerSelector(VolleyballCourtConfiguration())
    established = [_candidate(index, 5.0, confidence=0.75) for index in range(6)]
    for frame_index in range(12):
        selector.select(established, frame_index)

    replacement_frame = established[1:] + [_candidate(99, 5.0, confidence=0.99)]
    missing = selector.select(replacement_frame, 12)
    assert set(_active(missing, TrackSide.FAR)) == set(range(6))
    assert not missing[0].position_observed
    assert missing[0].visibility is VisibilityState.LOST
    assert not missing[99].active

    returned = selector.select(
        established + [_candidate(99, 5.0, confidence=0.99)], frame_index=13)
    assert set(_active(returned, TrackSide.FAR)) == set(range(6))
    assert not returned[99].active


def test_referee_overlap_preserves_sixth_player_as_occluded() -> None:
    selector = ActivePlayerSelector(
        VolleyballCourtConfiguration(),
        ActivePlayerSelectorConfig(active_occlusion_grace_frames=20),
    )
    player_box = (100.0, 100.0, 160.0, 220.0)
    established = [
        _candidate(index, 5.0, bbox=player_box if index == 2 else None)
        for index in range(6)
    ]
    for frame_index in range(12):
        selector.select(established, frame_index)

    visible = [candidate for candidate in established if candidate.track_id != 2]
    visible.append(_candidate(90, 8.9, confidence=0.99, bbox=(120, 80, 180, 240)))
    for offset in range(15):
        decisions = selector.select(
            visible,
            12 + offset,
            occluder_bboxes=(
                ((120.0, 80.0, 180.0, 240.0),) if offset == 0 else ()),
        )
        assert len(_active(decisions, TrackSide.FAR)) == 6
        assert sum(
            decision.active and decision.visible for decision in decisions.values()
        ) == 5
        assert decisions[2].active and decisions[2].occluded
        assert decisions[2].occlusion_evidence == "dynamic_person_overlap"
        assert not decisions[90].active


def test_occluded_player_reappears_without_seventh_member() -> None:
    selector = ActivePlayerSelector(
        VolleyballCourtConfiguration(),
        ActivePlayerSelectorConfig(
            active_occlusion_grace_frames=20, active_confirmation_frames=3),
    )
    established = [
        _candidate(index, 5.0, bbox=(100, 100, 160, 220) if index == 2 else None)
        for index in range(6)
    ]
    for frame_index in range(4):
        selector.select(established, frame_index)
    missing = established[:2] + established[3:] + [_candidate(90, 5.0, 0.99)]
    selector.select(missing, 4, occluder_bboxes=((110, 90, 170, 230),))

    returned = selector.select(established + [_candidate(90, 5.0, 0.99)], 5)
    assert set(_active(returned, TrackSide.FAR)) == set(range(6))
    assert returned[2].visible
    assert not returned[90].active


def test_occlusion_grace_expiry_releases_slot() -> None:
    selector = ActivePlayerSelector(
        VolleyballCourtConfiguration(),
        ActivePlayerSelectorConfig(
            active_occlusion_grace_frames=3, active_confirmation_frames=2),
    )
    established = [_candidate(index, 5.0) for index in range(6)]
    for frame_index in range(3):
        selector.select(established, frame_index)
    replacement = established[1:] + [_candidate(99, 5.0, 0.99)]
    for frame_index in range(3, 7):
        decisions = selector.select(replacement, frame_index)
    assert 0 not in decisions
    assert decisions[99].active


def test_static_zone_is_occlusion_evidence() -> None:
    selector = ActivePlayerSelector(
        VolleyballCourtConfiguration(),
        ActivePlayerSelectorConfig(active_confirmation_frames=2),
    )
    established = [
        _candidate(index, 5.0, in_occlusion_zone=index == 4)
        for index in range(6)
    ]
    selector.select(established, 0)
    selector.select(established, 1)
    decisions = selector.select(established[:4] + established[5:], 2)
    assert decisions[4].active and decisions[4].occluded
    assert decisions[4].occlusion_evidence == "static_zone"


def test_long_lived_stationary_net_track_does_not_win_on_age_alone() -> None:
    selector = ActivePlayerSelector(
        VolleyballCourtConfiguration(),
        ActivePlayerSelectorConfig(
            movement_window_frames=12,
            stationary_footprint_m=0.2,
            stationary_referee_penalty=2.0,
        ),
    )
    for frame_index in range(20):
        players = [
            _candidate(index, 5.0 + 0.2 * ((frame_index + index) % 3), 0.75,
                       x=1.0 + index)
            for index in range(6)
        ]
        referee = _candidate(
            99, 8.9, 0.99, x=9.8 if frame_index == 10 else 9.0,
            in_occlusion_zone=True)
        decisions = selector.select(players + [referee], frame_index)
    assert not decisions[99].active
    assert len(_active(decisions, TrackSide.FAR)) == 6


def test_six_player_invariant_includes_hidden_members() -> None:
    selector = ActivePlayerSelector(
        VolleyballCourtConfiguration(),
        ActivePlayerSelectorConfig(active_confirmation_frames=2),
    )
    established = [_candidate(index, 5.0) for index in range(6)]
    selector.select(established, 0)
    selector.select(established, 1)
    for frame_index in range(2, 12):
        visible = established[1:] + [
            _candidate(100 + index, 5.0, 0.99) for index in range(4)
        ]
        decisions = selector.select(visible, frame_index)
        assert len(_active(decisions, TrackSide.FAR)) <= 6


def test_net_hysteresis_prevents_rapid_side_flipping() -> None:
    selector = ActivePlayerSelector(VolleyballCourtConfiguration())
    sides = []
    for frame_index, y in enumerate((8.8, 9.05, 8.95, 9.1, 8.9)):
        decision = selector.select([_candidate(1, y)], frame_index)[1]
        sides.append(decision.side)
    assert sides == [TrackSide.FAR] * 5


def test_sustained_real_side_transition_eventually_switches() -> None:
    selector = ActivePlayerSelector(
        VolleyballCourtConfiguration(),
        ActivePlayerSelectorConfig(side_switch_frames=3, net_hysteresis_m=0.5),
    )
    sides = []
    for frame_index, y in enumerate((8.0, 10.0, 10.2, 10.1)):
        sides.append(selector.select([_candidate(1, y)], frame_index)[1].side)
    assert sides[:3] == [TrackSide.FAR] * 3
    assert sides[-1] is TrackSide.NEAR


def test_servers_are_eligible_on_their_canonical_side() -> None:
    court = VolleyballCourtConfiguration()
    selector = ActivePlayerSelector(court)
    candidates = [_candidate(1, -2.0), _candidate(2, 20.0)]
    assert all(court.is_inside_analysis_area(candidate.court_point) for candidate in candidates)
    decisions = selector.select(candidates, frame_index=0)
    assert decisions[1].side is TrackSide.FAR and decisions[1].active
    assert decisions[2].side is TrackSide.NEAR and decisions[2].active


def test_selection_is_invariant_to_renderer_camera_view() -> None:
    candidates = [_candidate(index, 4.0 if index < 7 else 14.0) for index in range(14)]
    results = []
    for _camera_view in (CameraView.ENDLINE, CameraView.SIDELINE):
        selector = ActivePlayerSelector(VolleyballCourtConfiguration())
        decisions = selector.select(candidates, frame_index=0)
        results.append({
            track_id: (decision.side, decision.active, decision.rank)
            for track_id, decision in decisions.items()
        })
    assert results[0] == results[1]
