from defensive_access.features import AccessConfig, analyse_frame
from defensive_access.models import PlayerPoint, TrackingFrame


def frame_with_defender(lane_y: float) -> TrackingFrame:
    attackers = [PlayerPoint(i, "home", float(i * 4), 10.0) for i in range(1, 9)]
    defenders = [PlayerPoint(100 + i, "away", float(i * 4), -10.0) for i in range(1, 8)]
    defenders.append(PlayerPoint(199, "away", 16.0, lane_y))
    return TrackingFrame(10, 1, "00:00:01.0", 0.0, 0.0, True, "home", tuple(attackers + defenders))


def test_lane_defender_reduces_access():
    cfg = AccessConfig(top_receivers=3)
    blocked = analyse_frame(frame_with_defender(5.0), cfg)
    far = analyse_frame(frame_with_defender(-25.0), cfg)
    assert blocked["valid"] and far["valid"]
    assert blocked["team_access"] < far["team_access"]


def test_counterfactual_contribution_is_reported_for_every_defender():
    result = analyse_frame(frame_with_defender(5.0))
    assert len(result["defender_contributions"]) == 8
    assert max(x["contribution"] for x in result["defender_contributions"]) >= 0


def test_receiver_rows_expose_raw_geometry_for_strong_baselines():
    result = analyse_frame(frame_with_defender(5.0))
    receiver = result["receiver_rows"][0]
    assert receiver["pass_distance_m"] > 0
    assert receiver["receiver_id"] == 2
    assert {"pass_dx_m", "pass_dy_m", "defenders_within_5m", "corridor_defenders"} <= receiver.keys()

