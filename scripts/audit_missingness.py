from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from defensive_access.io import iter_tracking
from run_phase01 import HOLDOUT_GAMES, _match_files


def eligibility(frame, min_players: int = 7) -> str:
    if frame.possession_team not in {"home", "away"}:
        return "unknown_possession"
    if frame.ball_x is None or frame.ball_y is None:
        return "missing_ball"
    home = sum(player.team == "home" for player in frame.players)
    away = sum(player.team == "away" for player in frame.players)
    if home < min_players or away < min_players:
        return "insufficient_players"
    return "valid"


def main() -> None:
    data_root = Path("data/raw")
    output = Path("outputs/phase01")
    frame_rows = []
    for match_dir in sorted(path for path in data_root.iterdir() if path.is_dir()):
        game_id = match_dir.name
        if game_id in HOLDOUT_GAMES:
            continue
        tracking, metadata, _ = _match_files(match_dir)
        match_rows = []
        for frame in iter_tracking(tracking, metadata, stride=5):
            match_rows.append(
                {
                    "game_id": game_id,
                    "frame": frame.frame,
                    "period": frame.period,
                    "possession_team": frame.possession_team or "unknown",
                    "ball_x": frame.ball_x,
                    "reason": eligibility(frame),
                }
            )
        match = pd.DataFrame(match_rows)
        match["time_decile"] = pd.cut(
            match["frame"], bins=10, labels=False, include_lowest=True
        ).astype("Int64")
        match["ball_zone"] = pd.cut(
            match["ball_x"],
            bins=[-np.inf, -17.5, 17.5, np.inf],
            labels=["left_third", "middle_third", "right_third"],
        ).astype("object")
        match["ball_zone"] = match["ball_zone"].fillna("unobserved")
        frame_rows.append(match)
        print(f"audited eligibility {game_id}", flush=True)

    frames = pd.concat(frame_rows, ignore_index=True)
    frames.to_csv(output / "frame_eligibility_sample.csv", index=False)
    by_match = (
        frames.groupby(["game_id", "reason"]).size().rename("frames").reset_index()
    )
    totals = frames.groupby("game_id").size().rename("total_frames")
    by_match = by_match.join(totals, on="game_id")
    by_match["fraction"] = by_match["frames"] / by_match["total_frames"]
    by_match.to_csv(output / "missingness_by_match.csv", index=False)
    strata = (
        frames.assign(valid=frames["reason"].eq("valid"))
        .groupby(["period", "time_decile", "ball_zone", "possession_team"], dropna=False)
        .agg(frames=("valid", "size"), valid_fraction=("valid", "mean"))
        .reset_index()
    )
    strata.to_csv(output / "missingness_by_observable_state.csv", index=False)
    print(by_match.groupby("reason")["fraction"].agg(["mean", "min", "max"]).to_string())


if __name__ == "__main__":
    main()
