from examples.volleyball.selection import (
    ActivePlayerSelector,
    ActivePlayerSelectorConfig,
    TrackCandidate,
    TrackSide,
)
from sports.configs.volleyball import CameraView, VolleyballCourtConfiguration


def _candidate(track_id, y, confidence=0.8, x=4.5):
    return TrackCandidate(
        track_id=track_id,
        court_point=(x, y),
        confidence=confidence,
        inside_court=0 <= x <= 9 and 0 <= y <= 18,
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


def test_recent_established_track_reclaims_slot_after_short_occlusion() -> None:
    selector = ActivePlayerSelector(VolleyballCourtConfiguration())
    established = [_candidate(index, 5.0, confidence=0.75) for index in range(6)]
    for frame_index in range(12):
        selector.select(established, frame_index)

    replacement_frame = established[1:] + [_candidate(99, 5.0, confidence=0.99)]
    assert 99 in _active(selector.select(replacement_frame, 12), TrackSide.FAR)

    returned = selector.select(
        established + [_candidate(99, 5.0, confidence=0.99)], frame_index=13)
    assert set(_active(returned, TrackSide.FAR)) == set(range(6))
    assert not returned[99].active


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
