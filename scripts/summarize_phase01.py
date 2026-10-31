from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from defensive_access.choice_models import evaluate_choices


def bootstrap_mean(values: np.ndarray, *, draws: int = 20_000, seed: int = 7) -> list[float]:
    rng = np.random.default_rng(seed)
    samples = rng.choice(values, size=(draws, len(values)), replace=True).mean(axis=1)
    return [float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))]


def exact_sign_flip_pvalue(values: np.ndarray) -> float:
    observed = abs(float(values.mean()))
    means = []
    for signs in itertools.product((-1.0, 1.0), repeat=len(values)):
        means.append(abs(float(np.mean(values * np.asarray(signs)))))
    return float(np.mean(np.asarray(means) >= observed - 1e-15))


def distance_sensitivity(predictions: pd.DataFrame) -> pd.DataFrame:
    selected = predictions.loc[predictions["chosen"].eq(1), [
        "game_id", "choice_set_id", "model", "pass_distance_m"
    ]].rename(columns={"pass_distance_m": "selected_distance_m"})
    joined = predictions.merge(selected, on=["game_id", "choice_set_id", "model"])
    rows = []
    bands = {
        "all": np.ones(len(joined), dtype=bool),
        "selected_5m_plus": joined["selected_distance_m"].ge(5).to_numpy(),
        "selected_5_15m": joined["selected_distance_m"].between(5, 15, inclusive="left").to_numpy(),
        "selected_15_30m": joined["selected_distance_m"].between(15, 30, inclusive="left").to_numpy(),
        "selected_30m_plus": joined["selected_distance_m"].ge(30).to_numpy(),
    }
    for band, mask in bands.items():
        subset = joined.loc[mask]
        for (game_id, model), group in subset.groupby(["game_id", "model"]):
            if group.empty:
                continue
            metrics = evaluate_choices(group, group["probability"].to_numpy())
            rows.append({"band": band, "game_id": game_id, "model": model, **metrics})
    return pd.DataFrame(rows)


def main() -> None:
    output = Path("outputs/phase01")
    metrics = pd.read_csv(output / "logo_metrics.csv", dtype={"game_id": str})
    predictions = pd.read_csv(
        output / "logo_predictions.csv", dtype={"game_id": str, "choice_set_id": str}
    )
    pivot = metrics.pivot(index="game_id", columns="model", values="conditional_log_loss")
    comparisons = []
    for reference in ["B0", "B2", "B3", "B4"]:
        difference = (pivot["M"] - pivot[reference]).to_numpy()
        comparisons.append(
            {
                "comparison": f"M - {reference}",
                "mean_log_loss_difference": float(difference.mean()),
                "bootstrap_95_interval": bootstrap_mean(difference),
                "matches_m_better": int((difference < 0).sum()),
                "matches_reference_better": int((difference > 0).sum()),
                "exact_sign_flip_pvalue": exact_sign_flip_pvalue(difference),
            }
        )
    (output / "paired_comparisons.json").write_text(json.dumps(comparisons, indent=2))

    sensitivity = distance_sensitivity(predictions)
    sensitivity.to_csv(output / "distance_sensitivity_by_match.csv", index=False)
    summary = sensitivity.groupby(["band", "model"]).agg(
        matches=("game_id", "nunique"),
        conditional_log_loss=("conditional_log_loss", "mean"),
        top1=("top1", "mean"),
        mrr=("mrr", "mean"),
    )
    summary.to_csv(output / "distance_sensitivity_summary.csv")
    print(json.dumps(comparisons, indent=2))
    print(summary.to_string())


if __name__ == "__main__":
    main()
