from defensive_access.counterfactual import defender_positioning_gaps
from defensive_access.models import PlayerPoint, TrackingFrame


def test_positioning_grid_excludes_engager_and_contains_observed_position():
    attackers = [PlayerPoint(i, "home", float(i * 4), 5.0) for i in range(1, 9)]
    defenders = [PlayerPoint(100 + i, "away", float(i * 4), -5.0) for i in range(1, 9)]
    frame = TrackingFrame(10, 1, "00:01", 0.0, 0.0, True, "home", tuple(attackers + defenders))
    rows = defender_positioning_gaps(frame, {2: 0.5, 3: 0.8})
    assert len(rows) == 7
    assert all(row["defender_id"] != row["engager_id"] for row in rows)
    assert all(row["positioning_gap"] >= -1e-12 for row in rows)
