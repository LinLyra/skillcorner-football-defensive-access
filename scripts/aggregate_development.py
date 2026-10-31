#!/usr/bin/env python3
from __future__ import annotations

import itertools
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEVELOPMENT_IDS = (1874553, 1886347, 1899585, 1925299, 1927964)
GAME_IDS = (
    1874553, 1886347, 1899585, 1925299, 1927964,
    1953632, 1959846, 1986691, 1996435, 1996436,
    2006229, 2006363, 2007448, 2007721, 2010085,
    2011166, 2013725, 2015213, 2016236, 2017461,
)


def exact_sign_flip_p(values: np.ndarray) -> float:
    observed = abs(float(values.mean()))
    null = [
        abs(float(np.mean(values * np.asarray(signs))))
        for signs in itertools.product((-1, 1), repeat=len(values))
    ]
    return sum(value >= observed - 1e-12 for value in null) / len(null)


def bootstrap_interval(values: np.ndarray, *, draws: int = 20000) -> tuple[float, float]:
    rng = np.random.default_rng(20261030)
    means = np.asarray([rng.choice(values, len(values), replace=True).mean() for _ in range(draws)])
    return tuple(np.quantile(means, [0.025, 0.975]))


def main() -> None:
    rows = []
    for game_id in GAME_IDS:
        root = Path("outputs") / str(game_id)
        validation = json.loads((root / "passing_option_validation.json").read_text())
        audit = json.loads((root / "audit.json").read_text())
        rows.append({"game_id": game_id, **validation, **audit})
    table = pd.DataFrame(rows)
    table["auc_gain"] = table["access_auc"] - table["distance_auc"]
    differences = table["auc_gain"].to_numpy()

    reports = Path("reports")
    figures = reports / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    table.to_csv(reports / "full_game_results.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(11, 8.5), constrained_layout=True)
    y = np.arange(len(table))
    axes[0].scatter(table["distance_auc"], y, label="Nearest-defender distance", s=55)
    axes[0].scatter(table["access_auc"], y, label="Defensive Access", s=55)
    for index, row in table.iterrows():
        axes[0].plot([row.distance_auc, row.access_auc], [index, index], color="#94a3b8", lw=2)
    axes[0].set_yticks(y, table["game_id"].astype(str))
    axes[0].set_xlabel("AUC: option received")
    axes[0].set_ylabel("Match")
    axes[0].legend(frameon=False, fontsize=8)
    axes[0].grid(axis="x", alpha=0.2)

    colours = ["#94a3b8" if x in DEVELOPMENT_IDS else "#2563eb" for x in table.game_id]
    axes[1].bar(y, table["auc_gain"], color=colours)
    axes[1].axhline(0, color="black", lw=1)
    axes[1].set_xticks(y, table["game_id"].astype(str), rotation=35)
    axes[1].set_ylabel("AUC gain over distance")
    n_positive = int((table["auc_gain"] > 0).sum())
    axes[1].set_title(f"Positive in {n_positive}/{len(table)} matches")
    axes[1].grid(axis="y", alpha=0.2)
    fig.suptitle("Defensive Access match-level validation", fontweight="bold")
    fig.savefig(figures / "full_validation.png", dpi=180)
    plt.close(fig)

    development = table.loc[table.game_id.isin(DEVELOPMENT_IDS)]
    locked = table.loc[~table.game_id.isin(DEVELOPMENT_IDS)]
    total_options = int(table["aligned_passing_options"].sum())
    mean_gain = float(table["auc_gain"].mean())
    p_value = exact_sign_flip_p(differences)
    low, high = bootstrap_interval(differences)
    locked_diff = locked["auc_gain"].to_numpy()
    locked_p = exact_sign_flip_p(locked_diff)
    locked_low, locked_high = bootstrap_interval(locked_diff)
    result = f"""# Full match-level validation

Five matches were used to exercise the pipeline and freeze definitions. The remaining 15 matches were processed without changing feature coefficients, speed assumptions, alignment tolerance, stride or exclusions.

## Result

- {total_options:,} passing options aligned to tracking across 20 matches.
- Defensive Access AUC exceeded nearest-defender-distance AUC in {int((differences > 0).sum())}/20 matches.
- All-match mean AUC gain: {mean_gain:+.4f}; match-bootstrap 95% interval [{low:+.4f}, {high:+.4f}].
- Range: {differences.min():+.4f} to {differences.max():+.4f}.
- Exact two-sided sign-flip p-value across 20 matches: {p_value:.6f}.
- Locked 15-match mean gain: {locked_diff.mean():+.4f}; bootstrap interval [{locked_low:+.4f}, {locked_high:+.4f}], exact sign-flip p = {locked_p:.6f}.
- Development-five mean gain: {development.auc_gain.mean():+.4f}.
- Valid sampled-frame coverage: {table.valid_fraction.min():.1%} to {table.valid_fraction.max():.1%}.

The single negative match is retained. Intervals resample matches, not frames; the 15-match analysis is locked validation but not an organiser-held hidden test.

![Twenty-match validation](figures/full_validation.png)

## Failure discovered and retained

Counterfactual lane contribution of the player recorded as the on-ball engager did not consistently rank engagement success. That estimand conflated the player attacking the ball with teammates protecting passing lanes. The project therefore separates receiver-access validation from a later role-complementarity analysis instead of tuning the sign after seeing outcomes.

## Remaining limits

`received` measures option use, not pass completion or causal defensive value. Event labels and tracking are products of the same provider and may share upstream modelling assumptions. The heuristic phase decoder remains experimental and is excluded from headline claims until it passes leave-one-match-out validation against official phase labels.
"""
    (reports / "full_results.md").write_text(result)
    print(result)


if __name__ == "__main__":
    main()

