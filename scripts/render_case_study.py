#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from defensive_access.geometry import point_segment_distance
from defensive_access.io import iter_tracking


GAME_IDS = (
    1874553, 1886347, 1899585, 1925299, 1927964,
    1953632, 1959846, 1986691, 1996435, 1996436,
    2006229, 2006363, 2007448, 2007721, 2010085,
    2011166, 2013725, 2015213, 2016236, 2017461,
)


def choose_case() -> pd.Series:
    candidates = []
    for game_id in GAME_IDS:
        path = Path("outputs") / str(game_id) / "aligned_passing_options.csv"
        frame = pd.read_csv(path)
        frame = frame.loc[frame.received.eq(1)].copy()
        frame["access_rank"] = frame.access_probability.rank(pct=True)
        frame["distance_rank"] = frame.nearest_defender_m.rank(pct=True)
        frame["disagreement"] = frame.access_rank - frame.distance_rank
        frame["game_id"] = game_id
        candidates.append(frame)
    return pd.concat(candidates, ignore_index=True).sort_values("disagreement").iloc[-1]


def tracking_frame(game_id: int, frame_id: int):
    root = Path("data/raw") / str(game_id)
    tracking = root / f"{game_id}_tracking_extrapolated.jsonl"
    metadata = root / f"{game_id}_match.json"
    for frame in iter_tracking(tracking, metadata):
        if frame.frame == frame_id:
            return frame, json.loads(metadata.read_text())
    raise ValueError(f"frame {frame_id} not found")


def draw_pitch(ax) -> None:
    ax.set_xlim(-54, 54)
    ax.set_ylim(-36, 36)
    ax.plot([-52.5, 52.5, 52.5, -52.5, -52.5], [-34, -34, 34, 34, -34], color="#334155")
    ax.axvline(0, color="#94a3b8", lw=1)
    circle = plt.Circle((0, 0), 9.15, fill=False, color="#94a3b8", lw=1)
    ax.add_patch(circle)
    for x, sign in [(-52.5, 1), (52.5, -1)]:
        ax.add_patch(plt.Rectangle((x, -20.16), sign * 16.5, 40.32, fill=False, color="#94a3b8"))
    ax.set_aspect("equal")
    ax.axis("off")


def main() -> None:
    case = choose_case()
    game_id = int(case.game_id)
    frame_id = int(case.aligned_frame)
    frame, metadata = tracking_frame(game_id, frame_id)
    receiver_id = int(case.receiver_id)
    attackers = [p for p in frame.players if p.team == frame.possession_team]
    defenders = [p for p in frame.players if p.team != frame.possession_team]
    receiver = next(p for p in attackers if p.player_id == receiver_id)
    ball = (float(frame.ball_x), float(frame.ball_y))
    blocker = min(
        defenders,
        key=lambda p: point_segment_distance((p.x, p.y), ball, (receiver.x, receiver.y))[0],
    )

    fig, (ax, text_ax) = plt.subplots(
        1, 2, figsize=(12, 5.4), gridspec_kw={"width_ratios": [1.55, 1]}, constrained_layout=True
    )
    draw_pitch(ax)
    ax.scatter([p.x for p in attackers], [p.y for p in attackers], s=55, color="#2563eb", label="In possession")
    ax.scatter([p.x for p in defenders], [p.y for p in defenders], s=55, color="#dc2626", label="Defending")
    ax.scatter(*ball, s=80, color="#111827", edgecolor="white", zorder=5, label="Ball")
    ax.scatter(receiver.x, receiver.y, s=220, marker="*", color="#f59e0b", edgecolor="#111827", zorder=6)
    ax.scatter(blocker.x, blocker.y, s=180, facecolors="none", edgecolors="#f59e0b", lw=2.5, zorder=6)
    ax.plot([ball[0], receiver.x], [ball[1], receiver.y], "--", color="#f59e0b", lw=2)
    ax.set_title(f"Match {game_id} · frame {frame_id}\nReceived option selected by reproducible disagreement rule")
    ax.legend(loc="upper right", frameon=False, fontsize=8)

    text_ax.axis("off")
    text_ax.text(0, 0.96, "Why distance alone is incomplete", fontsize=16, fontweight="bold", va="top")
    lines = [
        ("Receiver access", f"{case.access_probability:.3f}"),
        ("Nearest defender", f"{case.nearest_defender_m:.2f} m"),
        ("Control margin", f"{case.control_margin_s:+.2f} s"),
        ("Lane obstruction", f"{case.lane_block:.3f}"),
        ("Outcome", "Option received"),
    ]
    y = 0.78
    for label, value in lines:
        text_ax.text(0, y, label, color="#64748b", fontsize=11)
        text_ax.text(0.98, y, value, ha="right", fontsize=12, fontweight="bold")
        y -= 0.11
    text_ax.text(
        0,
        0.17,
        "The orange star is the receiver; the orange ring marks the defender closest to the pass lane. "
        "Access combines arrival-time margin and lane geometry, so two options at the same nearest-defender "
        "distance need not be equally usable.",
        fontsize=10.5,
        wrap=True,
        va="top",
    )
    output = Path("reports/figures/tactical_case.png")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=190)
    plt.close(fig)
    print(f"wrote {output}: game={game_id}, frame={frame_id}, receiver={receiver_id}")


if __name__ == "__main__":
    main()

