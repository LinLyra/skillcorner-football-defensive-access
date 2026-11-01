#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.animation import FuncAnimation, PillowWriter

from defensive_access.io import iter_tracking

ATTACKING = "#31688e"
DEFENDING = "#d95f02"
BALL = "#202020"


def draw_pitch(axis) -> None:
    axis.set_facecolor("#f6f3ea")
    axis.plot([-52.5, 52.5, 52.5, -52.5, -52.5], [-34, -34, 34, 34, -34], color="#303030")
    axis.axvline(0, color="#777777", linewidth=0.8)
    axis.add_patch(plt.Circle((0, 0), 9.15, fill=False, color="#777777", linewidth=0.8))
    for sign in (-1, 1):
        x0, x1 = 52.5 * sign, 36.0 * sign
        axis.plot([x0, x1, x1, x0], [-20.16, -20.16, 20.16, 20.16], color="#777777", linewidth=0.8)
    axis.set_xlim(-55, 55)
    axis.set_ylim(-36.5, 36.5)
    axis.set_aspect("equal")
    axis.set_xticks([])
    axis.set_yticks([])


def load_clip(tracking: Path, metadata: Path, start: int, end: int):
    frames = []
    for frame in iter_tracking(tracking, metadata):
        if frame.frame < start:
            continue
        if frame.frame > end:
            break
        frames.append(frame)
    if not frames:
        raise ValueError(f"No tracking frames found in [{start}, {end}]")
    return frames


def main() -> None:
    parser = argparse.ArgumentParser(description="Animate one strictly pre-pass receiver choice set.")
    parser.add_argument("--game-id", default="2013725")
    parser.add_argument("--choice-set-id", default="8_733")
    parser.add_argument(
        "--choices",
        type=Path,
        default=Path("outputs/phase01/secondary_holdout/choice_sets.csv"),
    )
    parser.add_argument("--data-root", type=Path, default=Path("data/raw"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/media/pre_pass_choice_set.gif"),
    )
    parser.add_argument("--seconds-before", type=float, default=2.0)
    parser.add_argument("--seconds-after", type=float, default=1.5)
    parser.add_argument("--fps", type=int, default=10)
    args = parser.parse_args()

    choices = pd.read_csv(args.choices, dtype={"game_id": str, "choice_set_id": str})
    group = choices.loc[
        choices["game_id"].eq(str(args.game_id))
        & choices["choice_set_id"].eq(str(args.choice_set_id))
    ].copy()
    if group.empty:
        raise ValueError(f"Choice set {args.game_id}/{args.choice_set_id} not found")

    aligned = int(group["aligned_frame"].iloc[0])
    decision = int(group["decision_frame"].iloc[0])
    start = aligned - round(args.seconds_before * args.fps)
    end = decision + round(args.seconds_after * args.fps)
    match = args.data_root / str(args.game_id)
    frames = load_clip(
        match / f"{args.game_id}_tracking_extrapolated.jsonl",
        match / f"{args.game_id}_match.json",
        start,
        end,
    )
    aligned_frame = next(frame for frame in frames if frame.frame == aligned)
    option_rows = list(group.sort_values("threat", ascending=False).itertuples(index=False))

    figure = plt.figure(figsize=(11.2, 5.6))
    layout = figure.add_gridspec(1, 2, width_ratios=[2.2, 1.0], wspace=0.08)
    pitch = figure.add_subplot(layout[0, 0])
    summary = figure.add_subplot(layout[0, 1])

    def render(index: int):
        frame = frames[index]
        pitch.clear()
        summary.clear()
        draw_pitch(pitch)

        attackers = [player for player in frame.players if player.team == frame.possession_team]
        defenders = [player for player in frame.players if player.team != frame.possession_team]
        pitch.scatter(
            [player.x for player in attackers],
            [player.y for player in attackers],
            s=58,
            color=ATTACKING,
            label="In possession",
            zorder=3,
        )
        pitch.scatter(
            [player.x for player in defenders],
            [player.y for player in defenders],
            s=58,
            color=DEFENDING,
            label="Defending",
            zorder=3,
        )
        if frame.ball_x is not None and frame.ball_y is not None:
            pitch.scatter(
                [frame.ball_x],
                [frame.ball_y],
                s=52,
                color=BALL,
                edgecolor="white",
                linewidth=0.8,
                label="Ball",
                zorder=5,
            )

        reveal = frame.frame >= aligned - 3
        if reveal:
            for row in option_rows:
                chosen = bool(row.chosen)
                color = "#e63946" if chosen else "#f4a261"
                pitch.plot(
                    [aligned_frame.ball_x, row.receiver_x],
                    [aligned_frame.ball_y, row.receiver_y],
                    color=color,
                    linewidth=3.4 if chosen else 2.2,
                    alpha=0.9,
                    zorder=2,
                )
                pitch.scatter(
                    [row.receiver_x],
                    [row.receiver_y],
                    marker="*" if chosen else "o",
                    s=170 if chosen else 105,
                    facecolor=color,
                    edgecolor="black",
                    linewidth=1.0,
                    zorder=6,
                )
                pitch.text(
                    row.receiver_x + 1.0,
                    row.receiver_y + 1.0,
                    f"T={row.threat:.2f}\nAccess={row.access_probability:.2f}",
                    fontsize=8,
                    weight="bold" if chosen else "normal",
                )

        relative = (frame.frame - decision) / args.fps
        phase = "PRE-PASS OPTIONS" if frame.frame <= decision else "PASS UNFOLDING"
        pitch.set_title(
            f"Game {args.game_id} · frame {frame.frame} · {relative:+.1f}s to decision\n{phase}",
            loc="left",
            fontsize=12,
            weight="bold",
        )
        pitch.legend(frameon=False, loc="lower left", ncol=3, fontsize=8)

        labels = [f"Receiver {row.receiver_id}" for row in option_rows]
        y = np.arange(len(option_rows))
        threat = [row.threat for row in option_rows]
        access = [row.access_probability for row in option_rows]
        summary.barh(y + 0.16, threat, height=0.28, color="#5e3c99", label="Threat T")
        summary.barh(y - 0.16, access, height=0.28, color="#1b9e77", label="Model access")
        summary.set_yticks(y, labels)
        summary.invert_yaxis()
        summary.set_xlim(0, 1.05)
        summary.set_xlabel("Score (0–1)")
        summary.grid(axis="x", color="#dddddd", linewidth=0.7)
        summary.spines[["top", "right", "left"]].set_visible(False)
        summary.set_title("Decision audit", loc="left", weight="bold")
        summary.legend(frameon=False, loc="lower right", fontsize=8)
        for pos, row in enumerate(option_rows):
            if row.chosen:
                summary.text(
                    0.02,
                    pos,
                    "ACTUAL CHOICE",
                    va="center",
                    fontsize=7.5,
                    weight="bold",
                    color="#e63946",
                )
        summary.text(
            0.0,
            -0.12,
            "Threat is destination value, not pass probability.\n"
            "Access is a geometry-based model score.",
            transform=summary.transAxes,
            fontsize=8,
            color="#555555",
            va="top",
        )
        figure.suptitle(
            "Pre-pass choice set: valuable does not always mean accessible",
            x=0.06,
            ha="left",
            fontsize=15,
            weight="bold",
        )
        return []

    animation = FuncAnimation(figure, render, frames=len(frames), interval=1000 / args.fps)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    animation.save(args.output, writer=PillowWriter(fps=args.fps), dpi=105)
    plt.close(figure)
    print(args.output)


if __name__ == "__main__":
    main()
