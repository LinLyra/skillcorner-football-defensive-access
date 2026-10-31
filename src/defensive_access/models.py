from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Team = Literal["home", "away"]


@dataclass(frozen=True)
class PlayerPoint:
    player_id: int
    team: Team
    x: float
    y: float
    detected: bool = True


@dataclass(frozen=True)
class TrackingFrame:
    frame: int
    period: int | None
    timestamp: str | None
    ball_x: float | None
    ball_y: float | None
    ball_detected: bool | None
    possession_team: Team | None
    players: tuple[PlayerPoint, ...]


@dataclass(frozen=True)
class Spell:
    label: str
    start: int
    end: int

    @property
    def duration(self) -> int:
        return self.end - self.start + 1

