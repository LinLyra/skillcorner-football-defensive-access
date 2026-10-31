from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from defensive_access.features import AccessConfig, _receiver_access
from defensive_access.io import iter_tracking
from defensive_access.models import PlayerPoint
from run_phase01 import _match_files


FIGURES = Path("reports/figures")


def draw_pitch(axis) -> None:
    axis.set_facecolor("#f6f3ea")
    axis.plot([-52.5, 52.5, 52.5, -52.5, -52.5], [-34, -34, 34, 34, -34], color="#303030")
    axis.axvline(0, color="#777777", linewidth=0.8)
    axis.add_patch(plt.Circle((0, 0), 9.15, fill=False, color="#777777", linewidth=0.8))
    for sign in (-1, 1):
        x0, x1 = (52.5 * sign, 36.0 * sign)
        axis.plot([x0, x1, x1, x0], [-20.16, -20.16, 20.16, 20.16], color="#777777", linewidth=0.8)
    axis.set_xlim(-55, 55)
    axis.set_ylim(-36.5, 36.5)
    axis.set_aspect("equal")
    axis.set_xticks([])
    axis.set_yticks([])


def make_validation_figure() -> None:
    development = pd.read_csv("outputs/phase01/logo_metrics.csv").groupby("model")[
        "conditional_log_loss"
    ].mean()
    holdout = pd.read_csv("outputs/phase01/secondary_holdout/metrics.csv").groupby("model")[
        "conditional_log_loss"
    ].mean()
    order = ["B0", "B1", "B2", "M", "B4"]
    labels = [
        "B0 nearest defender",
        "B1 ordinary geometry",
        "B2 + local/corridor counts",
        "M + lane-aware access",
        "B4 nonlinear raw geometry",
    ]
    y = np.arange(len(order))
    figure, axis = plt.subplots(figsize=(10.5, 5.8))
    axis.scatter(development.loc[order], y - 0.12, s=70, color="#31688e", label="15-match LOGO")
    axis.scatter(
        holdout.loc[order], y + 0.12, s=70, marker="D", color="#d95f02", label="5-match secondary holdout"
    )
    for index, model in enumerate(order):
        axis.plot(
            [development.loc[model], holdout.loc[model]],
            [index - 0.12, index + 0.12],
            color="#b7b7b7",
            linewidth=1,
            zorder=0,
        )
    axis.set_yticks(y, labels)
    axis.invert_yaxis()
    axis.set_xlabel("Conditional log loss per decision (lower is better)")
    figure.suptitle(
        "Strong baselines absorb most of the original Defensive Access gain",
        x=0.21,
        y=0.98,
        ha="left",
        weight="bold",
    )
    figure.text(
        0.21,
        0.925,
        "Strict pre-pass, simultaneous choice sets; every point is evaluated on unseen matches",
        color="#555555",
    )
    axis.grid(axis="x", color="#dddddd", linewidth=0.8)
    axis.legend(frameon=False, loc="lower right")
    axis.text(
        0.99,
        0.16,
        "M − B2 (15-match LOGO): −0.00215\n"
        "match bootstrap 95%: [−0.00390, 0.00000]\n"
        "exact sign-flip p = 0.055\n"
        "B3 omitted: algebraically redundant with B2",
        transform=axis.transAxes,
        ha="right",
        va="bottom",
        fontsize=8.5,
        color="#555555",
        bbox={"boxstyle": "round,pad=0.45", "facecolor": "white", "edgecolor": "#cccccc"},
    )
    for spine in ["top", "right", "left"]:
        axis.spines[spine].set_visible(False)
    figure.tight_layout(rect=[0, 0, 1, 0.89])
    figure.savefig(FIGURES / "choice_set_validation.png", dpi=220, bbox_inches="tight")
    plt.close(figure)


def load_frame(game_id: str, frame_id: int):
    tracking, metadata, _ = _match_files(Path("data/raw") / game_id)
    for frame in iter_tracking(tracking, metadata):
        if frame.frame == frame_id:
            return frame
        if frame.frame > frame_id:
            break
    raise ValueError(f"Frame {frame_id} not found in game {game_id}")


def concession_surface(frame, receiver_threat, defender_id: int):
    cfg = AccessConfig()
    defenders = [player for player in frame.players if player.team != frame.possession_team]
    attackers = {player.player_id: player for player in frame.players if player.team == frame.possession_team}
    moved = next(player for player in defenders if player.player_id == defender_id)
    others = [player for player in defenders if player.player_id != defender_id]
    offsets = np.arange(-4.0, 4.01, 1.0)
    values = np.full((len(offsets), len(offsets)), np.nan)
    ball = (float(frame.ball_x), float(frame.ball_y))
    for ix, dx in enumerate(offsets):
        for iy, dy in enumerate(offsets):
            x, y = moved.x + dx, moved.y + dy
            if not (-52.5 <= x <= 52.5 and -34 <= y <= 34):
                continue
            candidate = PlayerPoint(moved.player_id, moved.team, x, y, moved.detected)
            candidate_defenders = others + [candidate]
            values[iy, ix] = sum(
                _receiver_access(ball, attackers[player_id], candidate_defenders, cfg)[
                    "access_probability"
                ]
                * threat
                for player_id, threat in receiver_threat.items()
                if player_id in attackers
            )
    return offsets, values


def make_hero_figure() -> None:
    gaps = pd.read_csv(
        "outputs/phase03/holdout_defender_gaps.csv",
        dtype={"game_id": str, "choice_set_id": str},
    )
    options = pd.read_csv(
        "outputs/phase01/secondary_holdout/choice_sets.csv",
        dtype={"game_id": str, "choice_set_id": str},
    )
    displacement = np.hypot(gaps["best_dx_m"], gaps["best_dy_m"])
    non_boundary = gaps["best_dx_m"].abs().lt(4.0) & gaps["best_dy_m"].abs().lt(4.0)
    eligible = gaps.loc[
        gaps["observed_concession"].ge(gaps["observed_concession"].median())
        & displacement.ge(1.5)
        & non_boundary
    ].copy()
    target_gap = gaps["positioning_gap"].quantile(0.85)
    case = eligible.iloc[(eligible["positioning_gap"] - target_gap).abs().argsort()[:1]].iloc[0]
    group = options.loc[
        options["game_id"].eq(case.game_id)
        & options["choice_set_id"].eq(case.choice_set_id)
    ]
    frame = load_frame(case.game_id, int(case.aligned_frame))
    receiver_threat = dict(zip(group["receiver_id"].astype(int), group["threat"].astype(float)))
    offsets, surface = concession_surface(frame, receiver_threat, int(case.defender_id))

    figure = plt.figure(figsize=(13.2, 6.4))
    grid = figure.add_gridspec(1, 2, width_ratios=[1.7, 1.0], wspace=0.12)
    pitch = figure.add_subplot(grid[0, 0])
    heat = figure.add_subplot(grid[0, 1])
    draw_pitch(pitch)
    attackers = [player for player in frame.players if player.team == frame.possession_team]
    defenders = [player for player in frame.players if player.team != frame.possession_team]
    pitch.scatter([p.x for p in attackers], [p.y for p in attackers], s=55, color="#31688e", label="In possession")
    pitch.scatter([p.x for p in defenders], [p.y for p in defenders], s=55, color="#d95f02", label="Defending")
    pitch.scatter([frame.ball_x], [frame.ball_y], s=55, color="#202020", edgecolor="white", zorder=5, label="Ball")
    for row in group.itertuples(index=False):
        pitch.plot(
            [frame.ball_x, row.receiver_x],
            [frame.ball_y, row.receiver_y],
            color=plt.cm.magma(0.15 + 0.75 * row.threat),
            linewidth=1.2 + 3.0 * row.threat,
            alpha=0.8,
        )
        pitch.text(row.receiver_x + 0.8, row.receiver_y + 0.8, f"T={row.threat:.2f}", fontsize=8)
    pitch.scatter([case.observed_x], [case.observed_y], s=170, facecolors="none", edgecolors="#111111", linewidth=2.2, zorder=6)
    pitch.annotate(
        "model-local optimum",
        xy=(case.best_x, case.best_y),
        xytext=(-115, 42),
        textcoords="offset points",
        arrowprops={"arrowstyle": "->", "color": "#111111", "lw": 2},
        fontsize=9,
    )
    pitch.set_title("One auditable positioning contrast", loc="left", weight="bold")
    pitch.legend(frameon=False, loc="lower left", ncol=3, fontsize=8)

    extent = [case.observed_x + offsets.min() - 0.5, case.observed_x + offsets.max() + 0.5,
              case.observed_y + offsets.min() - 0.5, case.observed_y + offsets.max() + 0.5]
    image = heat.imshow(surface, origin="lower", extent=extent, cmap="viridis", aspect="equal")
    heat.scatter([case.observed_x], [case.observed_y], s=100, facecolors="white", edgecolors="black", label="Observed")
    heat.scatter([case.best_x], [case.best_y], marker="*", s=170, color="#ffcc33", edgecolor="black", label="Local minimum")
    heat.set_xlabel("Pitch x (m)")
    heat.set_ylabel("Pitch y (m)")
    heat.set_title("Local dangerous-option concession", loc="left", weight="bold")
    heat.legend(frameon=False, loc="upper right")
    colorbar = figure.colorbar(image, ax=heat, fraction=0.045, pad=0.03)
    colorbar.set_label("Concession score (model units)")
    figure.suptitle(
        f"Game {case.game_id}, frame {int(case.aligned_frame)}  |  Non-boundary illustrative gap {case.positioning_gap:.3f}",
        x=0.06,
        ha="left",
        fontsize=13,
        weight="bold",
    )
    figure.text(
        0.06,
        0.01,
        "Non-boundary case nearest the 85th-percentile gap; holdout mean gap = 0.014. The gap did not improve outcome prediction.",
        color="#555555",
        fontsize=9,
    )
    figure.savefig(FIGURES / "positioning_counterfactual.png", dpi=220, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    make_validation_figure()
    make_hero_figure()
    print(FIGURES / "choice_set_validation.png")
    print(FIGURES / "positioning_counterfactual.png")


if __name__ == "__main__":
    main()
