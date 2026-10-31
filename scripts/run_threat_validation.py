from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import log_loss, mean_squared_error, r2_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from defensive_access.threat import aggregate_dangerous_options


FEATURES = {
    "E0_opportunity": ["option_count", "sum_threat", "max_threat"],
    "E1_distance": [
        "option_count",
        "sum_threat",
        "max_threat",
        "distance_concession",
    ],
    "E2_access": [
        "option_count",
        "sum_threat",
        "max_threat",
        "distance_concession",
        "access_concession",
    ],
}


def evaluate_logo(table: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for outcome in ["lead_to_shot", "pass_epv_delta_for", "n_opponents_bypassed", "selected_threat"]:
        eligible = table.dropna(subset=[outcome]).copy()
        for game_id in sorted(eligible["game_id"].astype(str).unique()):
            train = eligible.loc[eligible["game_id"].astype(str).ne(game_id)]
            test = eligible.loc[eligible["game_id"].astype(str).eq(game_id)]
            for name, features in FEATURES.items():
                if outcome == "lead_to_shot":
                    model = make_pipeline(
                        StandardScaler(), LogisticRegression(C=1.0, max_iter=1000)
                    )
                    model.fit(train[features], train[outcome].astype(int))
                    prediction = model.predict_proba(test[features])[:, 1]
                    auc = (
                        roc_auc_score(test[outcome], prediction)
                        if test[outcome].nunique() == 2
                        else np.nan
                    )
                    rows.append(
                        {
                            "outcome": outcome,
                            "game_id": game_id,
                            "model": name,
                            "n": len(test),
                            "loss": log_loss(test[outcome], prediction, labels=[0, 1]),
                            "secondary_metric": auc,
                            "secondary_metric_name": "auc",
                        }
                    )
                else:
                    model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
                    model.fit(train[features], train[outcome])
                    prediction = model.predict(test[features])
                    rows.append(
                        {
                            "outcome": outcome,
                            "game_id": game_id,
                            "model": name,
                            "n": len(test),
                            "loss": mean_squared_error(test[outcome], prediction) ** 0.5,
                            "secondary_metric": r2_score(test[outcome], prediction),
                            "secondary_metric_name": "r2",
                        }
                    )
    return pd.DataFrame(rows)


def evaluate_secondary_holdout(train: pd.DataFrame, holdout: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for outcome in ["lead_to_shot", "pass_epv_delta_for", "n_opponents_bypassed", "selected_threat"]:
        outcome_train = train.dropna(subset=[outcome])
        outcome_test = holdout.dropna(subset=[outcome])
        for name, features in FEATURES.items():
            if outcome == "lead_to_shot":
                model = make_pipeline(
                    StandardScaler(), LogisticRegression(C=1.0, max_iter=1000)
                ).fit(outcome_train[features], outcome_train[outcome].astype(int))
                for game_id, test in outcome_test.groupby("game_id"):
                    prediction = model.predict_proba(test[features])[:, 1]
                    rows.append(
                        {
                            "outcome": outcome,
                            "game_id": game_id,
                            "model": name,
                            "n": len(test),
                            "loss": log_loss(test[outcome], prediction, labels=[0, 1]),
                            "secondary_metric": roc_auc_score(test[outcome], prediction),
                            "secondary_metric_name": "auc",
                        }
                    )
            else:
                model = make_pipeline(StandardScaler(), Ridge(alpha=1.0)).fit(
                    outcome_train[features], outcome_train[outcome]
                )
                for game_id, test in outcome_test.groupby("game_id"):
                    prediction = model.predict(test[features])
                    rows.append(
                        {
                            "outcome": outcome,
                            "game_id": game_id,
                            "model": name,
                            "n": len(test),
                            "loss": mean_squared_error(test[outcome], prediction) ** 0.5,
                            "secondary_metric": r2_score(test[outcome], prediction),
                            "secondary_metric_name": "r2",
                        }
                    )
    return pd.DataFrame(rows)


def main() -> None:
    output = Path("outputs/phase02")
    output.mkdir(parents=True, exist_ok=True)
    options = pd.read_csv(
        "outputs/phase01/development_choice_sets.csv",
        dtype={"game_id": str, "choice_set_id": str},
    )
    decisions = aggregate_dangerous_options(options)
    decisions.to_csv(output / "dangerous_option_concession.csv", index=False)
    metrics = evaluate_logo(decisions)
    metrics.to_csv(output / "external_validity_by_match.csv", index=False)
    summary = metrics.groupby(["outcome", "model"]).agg(
        matches=("game_id", "nunique"),
        mean_loss=("loss", "mean"),
        mean_secondary_metric=("secondary_metric", "mean"),
    )
    summary.to_csv(output / "external_validity_summary.csv")

    holdout_options = pd.read_csv(
        "outputs/phase01/secondary_holdout/choice_sets.csv",
        dtype={"game_id": str, "choice_set_id": str},
    )
    holdout_decisions = aggregate_dangerous_options(holdout_options)
    holdout_metrics = evaluate_secondary_holdout(decisions, holdout_decisions)
    holdout_metrics.to_csv(output / "secondary_holdout_by_match.csv", index=False)
    holdout_summary = holdout_metrics.groupby(["outcome", "model"]).agg(
        matches=("game_id", "nunique"),
        mean_loss=("loss", "mean"),
        mean_secondary_metric=("secondary_metric", "mean"),
    )
    holdout_summary.to_csv(output / "secondary_holdout_summary.csv")

    pivot = metrics.pivot(index=["outcome", "game_id"], columns="model", values="loss")
    comparisons = []
    for outcome in pivot.index.get_level_values("outcome").unique():
        difference = (
            pivot.loc[outcome, "E2_access"] - pivot.loc[outcome, "E1_distance"]
        ).to_numpy()
        comparisons.append(
            {
                "outcome": outcome,
                "mean_loss_e2_minus_e1": float(difference.mean()),
                "matches_e2_better": int((difference < 0).sum()),
                "matches_e1_better": int((difference > 0).sum()),
            }
        )
    (output / "external_validity_comparisons.json").write_text(
        json.dumps(comparisons, indent=2)
    )
    print(summary.to_string())
    print(json.dumps(comparisons, indent=2))
    print("\nSecondary holdout")
    print(holdout_summary.to_string())


if __name__ == "__main__":
    main()
