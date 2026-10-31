from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

from .models import PlayerPoint, TrackingFrame


def load_match_metadata(path: str | Path) -> dict:
    return json.loads(Path(path).read_text())


def player_team_map(metadata: dict) -> dict[int, str]:
    home_id = metadata["home_team"]["id"]
    away_id = metadata["away_team"]["id"]
    result: dict[int, str] = {}
    for player in metadata["players"]:
        team_id = player.get("team_id")
        if team_id == home_id:
            result[int(player["id"])] = "home"
        elif team_id == away_id:
            result[int(player["id"])] = "away"
    return result


def iter_tracking(
    tracking_path: str | Path,
    metadata_path: str | Path,
    *,
    stride: int = 1,
) -> Iterator[TrackingFrame]:
    metadata = load_match_metadata(metadata_path)
    teams = player_team_map(metadata)
    with Path(tracking_path).open() as handle:
        for line in handle:
            raw = json.loads(line)
            frame_id = int(raw["frame"])
            if frame_id % stride:
                continue
            ball = raw.get("ball_data") or {}
            possession = raw.get("possession") or {}
            group = possession.get("group")
            possession_team = {
                "home": "home",
                "home team": "home",
                "away": "away",
                "away team": "away",
            }.get(group)
            players = []
            for item in raw.get("player_data") or []:
                player_id = item.get("player_id")
                team = teams.get(player_id)
                if team is None or item.get("x") is None or item.get("y") is None:
                    continue
                players.append(
                    PlayerPoint(
                        player_id=int(player_id),
                        team=team,
                        x=float(item["x"]),
                        y=float(item["y"]),
                        detected=bool(item.get("is_detected")),
                    )
                )
            yield TrackingFrame(
                frame=frame_id,
                period=raw.get("period"),
                timestamp=raw.get("timestamp"),
                ball_x=ball.get("x"),
                ball_y=ball.get("y"),
                ball_detected=ball.get("is_detected"),
                possession_team=possession_team,
                players=tuple(players),
            )

