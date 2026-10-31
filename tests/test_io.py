import json

from defensive_access.io import iter_tracking


def test_provider_possession_labels_are_normalised(tmp_path):
    metadata = {
        "home_team": {"id": 1},
        "away_team": {"id": 2},
        "players": [
            {"id": 10, "team_id": 1},
            {"id": 20, "team_id": 2},
        ],
    }
    frame = {
        "frame": 1,
        "timestamp": "00:00:00.1",
        "period": 1,
        "ball_data": {"x": 0, "y": 0, "is_detected": True},
        "possession": {"group": "home team"},
        "player_data": [
            {"player_id": 10, "x": 0, "y": 0, "is_detected": True},
            {"player_id": 20, "x": 1, "y": 1, "is_detected": True},
        ],
    }
    metadata_path = tmp_path / "match.json"
    tracking_path = tmp_path / "tracking.jsonl"
    metadata_path.write_text(json.dumps(metadata))
    tracking_path.write_text(json.dumps(frame) + "\n")
    parsed = next(iter_tracking(tracking_path, metadata_path))
    assert parsed.possession_team == "home"

