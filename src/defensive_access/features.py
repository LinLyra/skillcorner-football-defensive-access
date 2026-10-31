from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .geometry import convex_hull_area, euclidean, point_segment_distance, sigmoid
from .models import PlayerPoint, TrackingFrame


@dataclass(frozen=True)
class AccessConfig:
    defender_speed_mps: float = 5.5
    pass_speed_mps: float = 15.0
    lane_radius_m: float = 2.0
    min_players_per_team: int = 7
    top_receivers: int = 3
    nearest_distance_cap_m: float = 15.0


def _receiver_access(
    ball: tuple[float, float],
    receiver: PlayerPoint,
    defenders: list[PlayerPoint],
    cfg: AccessConfig,
) -> dict[str, float]:
    target = (receiver.x, receiver.y)
    pass_dx = target[0] - ball[0]
    pass_dy = target[1] - ball[1]
    pass_distance = euclidean(ball, target)
    distances = np.asarray([euclidean(target, (d.x, d.y)) for d in defenders])
    nearest = float(distances.min()) if len(distances) else 99.0
    pressure_arrival = nearest / cfg.defender_speed_mps
    ball_arrival = pass_distance / cfg.pass_speed_mps
    control_margin = pressure_arrival - ball_arrival

    lane_scores = []
    corridor_defenders = 0
    for defender in defenders:
        lane_distance, projection = point_segment_distance((defender.x, defender.y), ball, target)
        centrality = 4.0 * projection * (1.0 - projection)
        lane_scores.append(np.exp(-0.5 * (lane_distance / cfg.lane_radius_m) ** 2) * centrality)
        if 0.0 < projection < 1.0 and lane_distance <= cfg.lane_radius_m:
            corridor_defenders += 1
    lane_block = float(max(lane_scores, default=0.0))
    # Transparent, monotone heuristic. Coefficients are not fitted to event outcomes.
    access = sigmoid(
        -1.2
        + 0.18 * min(nearest, cfg.nearest_distance_cap_m)
        + 0.9 * control_margin
        - 2.0 * lane_block
    )
    return {
        "receiver_id": receiver.player_id,
        "ball_x": float(ball[0]),
        "ball_y": float(ball[1]),
        "receiver_x": float(receiver.x),
        "receiver_y": float(receiver.y),
        "pass_dx_m": float(pass_dx),
        "pass_dy_m": float(pass_dy),
        "pass_distance_m": float(pass_distance),
        "pass_angle_rad": float(np.arctan2(pass_dy, pass_dx)),
        "nearest_defender_m": nearest,
        "defenders_within_5m": float(np.sum(distances <= 5.0)),
        "corridor_defenders": float(corridor_defenders),
        "ball_arrival_s": ball_arrival,
        "pressure_arrival_s": pressure_arrival,
        "control_margin_s": control_margin,
        "lane_block": lane_block,
        "access_probability": access,
    }


def _team_access(receiver_rows: list[dict[str, float]], top_k: int) -> float:
    values = sorted((row["access_probability"] for row in receiver_rows), reverse=True)
    return float(np.mean(values[:top_k])) if values else float("nan")


def analyse_frame(frame: TrackingFrame, cfg: AccessConfig | None = None) -> dict:
    cfg = cfg or AccessConfig()
    if frame.possession_team not in {"home", "away"}:
        return {"valid": False, "reason": "unknown_possession", "frame": frame.frame}
    if frame.ball_x is None or frame.ball_y is None:
        return {"valid": False, "reason": "missing_ball", "frame": frame.frame}

    attackers = [p for p in frame.players if p.team == frame.possession_team]
    defending_team = "away" if frame.possession_team == "home" else "home"
    defenders = [p for p in frame.players if p.team == defending_team]
    if len(attackers) < cfg.min_players_per_team or len(defenders) < cfg.min_players_per_team:
        return {"valid": False, "reason": "insufficient_players", "frame": frame.frame}

    ball = (float(frame.ball_x), float(frame.ball_y))
    ball_carrier = min(attackers, key=lambda p: euclidean(ball, (p.x, p.y)))
    receivers = [p for p in attackers if p.player_id != ball_carrier.player_id]
    receiver_rows = [_receiver_access(ball, receiver, defenders, cfg) for receiver in receivers]
    full_access = _team_access(receiver_rows, cfg.top_receivers)

    defender_contributions = []
    for defender in defenders:
        reduced = [d for d in defenders if d.player_id != defender.player_id]
        cf_rows = [_receiver_access(ball, receiver, reduced, cfg) for receiver in receivers]
        without_access = _team_access(cf_rows, cfg.top_receivers)
        defender_contributions.append(
            {
                "player_id": defender.player_id,
                "access_without": without_access,
                "contribution": without_access - full_access,
                "detected": defender.detected,
                "x": defender.x,
                "y": defender.y,
                "distance_to_ball_m": euclidean(ball, (defender.x, defender.y)),
            }
        )

    def_xy = np.asarray([(p.x, p.y) for p in defenders], dtype=float)
    nearest_ball = min(euclidean(ball, (p.x, p.y)) for p in defenders)
    return {
        "valid": True,
        "frame": frame.frame,
        "period": frame.period,
        "timestamp": frame.timestamp,
        "possession_team": frame.possession_team,
        "defending_team": defending_team,
        "ball_x": ball[0],
        "ball_y": ball[1],
        "ball_detected": frame.ball_detected,
        "n_attackers": len(attackers),
        "n_defenders": len(defenders),
        "observed_player_fraction": float(np.mean([p.detected for p in frame.players])),
        "team_access": full_access,
        "nearest_defender_to_ball_m": nearest_ball,
        "defensive_width_m": float(np.ptp(def_xy[:, 1])),
        "defensive_length_m": float(np.ptp(def_xy[:, 0])),
        "defensive_hull_area_m2": convex_hull_area((p.x, p.y) for p in defenders),
        "defensive_centroid_x": float(def_xy[:, 0].mean()),
        "defensive_centroid_y": float(def_xy[:, 1].mean()),
        "receiver_rows": receiver_rows,
        "defender_contributions": defender_contributions,
    }

