from __future__ import annotations

import numpy as np

from .features import AccessConfig, _receiver_access
from .geometry import euclidean
from .models import TrackingFrame


def _lane_score_grid(
    grid: np.ndarray,
    ball: np.ndarray,
    target: np.ndarray,
    lane_radius_m: float,
) -> np.ndarray:
    segment = target - ball
    denominator = float(segment @ segment)
    if denominator <= 1e-12:
        return np.zeros(len(grid))
    projection = np.clip(((grid - ball) @ segment) / denominator, 0.0, 1.0)
    closest = ball + projection[:, None] * segment
    lane_distance = np.linalg.norm(grid - closest, axis=1)
    centrality = 4.0 * projection * (1.0 - projection)
    return np.exp(-0.5 * (lane_distance / lane_radius_m) ** 2) * centrality


def defender_positioning_gaps(
    frame: TrackingFrame,
    receiver_threat: dict[int, float],
    *,
    config: AccessConfig | None = None,
    grid_radius_m: float = 4.0,
    grid_step_m: float = 1.0,
    reachable_m: float = 7.0,
) -> list[dict]:
    config = config or AccessConfig()
    if frame.ball_x is None or frame.ball_y is None or frame.possession_team is None:
        return []
    attackers = [player for player in frame.players if player.team == frame.possession_team]
    defending_team = "away" if frame.possession_team == "home" else "home"
    defenders = [player for player in frame.players if player.team == defending_team]
    receiver_by_id = {player.player_id: player for player in attackers}
    receivers = [receiver_by_id[player_id] for player_id in receiver_threat if player_id in receiver_by_id]
    if len(receivers) < 2 or len(defenders) < config.min_players_per_team:
        return []

    ball = np.array([float(frame.ball_x), float(frame.ball_y)])
    engager = min(defenders, key=lambda player: euclidean(ball, (player.x, player.y)))
    actual = sum(
        _receiver_access(tuple(ball), receiver, defenders, config)["access_probability"]
        * receiver_threat[receiver.player_id]
        for receiver in receivers
    )
    offsets = np.arange(-grid_radius_m, grid_radius_m + 1e-9, grid_step_m)
    mesh = np.asarray([(dx, dy) for dx in offsets for dy in offsets], dtype=float)
    mesh = mesh[np.linalg.norm(mesh, axis=1) <= reachable_m + 1e-9]
    rows = []
    for moved in defenders:
        if moved.player_id == engager.player_id:
            continue
        origin = np.array([moved.x, moved.y], dtype=float)
        grid = origin + mesh
        inside_pitch = (
            (grid[:, 0] >= -52.5)
            & (grid[:, 0] <= 52.5)
            & (grid[:, 1] >= -34.0)
            & (grid[:, 1] <= 34.0)
        )
        candidate_grid = grid[inside_pitch]
        candidate_offsets = mesh[inside_pitch]
        others = [defender for defender in defenders if defender.player_id != moved.player_id]
        concession = np.zeros(len(candidate_grid), dtype=float)
        for receiver in receivers:
            target = np.array([receiver.x, receiver.y], dtype=float)
            other_distances = np.asarray(
                [euclidean(target, (defender.x, defender.y)) for defender in others]
            )
            nearest_without = float(other_distances.min())
            other_lane = []
            for defender in others:
                defender_grid = np.array([[defender.x, defender.y]], dtype=float)
                other_lane.append(
                    float(_lane_score_grid(defender_grid, ball, target, config.lane_radius_m)[0])
                )
            lane_without = max(other_lane, default=0.0)
            moved_distance = np.linalg.norm(candidate_grid - target, axis=1)
            nearest = np.minimum(nearest_without, moved_distance)
            ball_arrival = np.linalg.norm(target - ball) / config.pass_speed_mps
            control_margin = nearest / config.defender_speed_mps - ball_arrival
            moved_lane = _lane_score_grid(candidate_grid, ball, target, config.lane_radius_m)
            lane_block = np.maximum(lane_without, moved_lane)
            linear = (
                -1.2
                + 0.18 * np.minimum(nearest, config.nearest_distance_cap_m)
                + 0.9 * control_margin
                - 2.0 * lane_block
            )
            access = 1.0 / (1.0 + np.exp(-linear))
            concession += access * receiver_threat[receiver.player_id]

        best = int(np.argmin(concession))
        rows.append(
            {
                "defender_id": moved.player_id,
                "engager_id": engager.player_id,
                "observed_concession": float(actual),
                "optimal_local_concession": float(concession[best]),
                "positioning_gap": float(actual - concession[best]),
                "best_dx_m": float(candidate_offsets[best, 0]),
                "best_dy_m": float(candidate_offsets[best, 1]),
                "distance_to_ball_m": euclidean(ball, (moved.x, moved.y)),
                "observed_x": float(moved.x),
                "observed_y": float(moved.y),
                "best_x": float(candidate_grid[best, 0]),
                "best_y": float(candidate_grid[best, 1]),
            }
        )
    return rows
