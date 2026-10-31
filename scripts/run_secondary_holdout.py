from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from defensive_access.choice_models import (
    FEATURE_SETS,
    ConditionalSoftmax,
    choice_key,
    evaluate_choices,
)
from defensive_access.choices import build_choice_sets
from run_phase01 import HOLDOUT_GAMES, _match_files, _normalise_within_choice


def main() -> None:
    data_root = Path("data/raw")
    root = Path("outputs/phase01")
    destination = root / "secondary_holdout"
    destination.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(
        root / "development_choice_sets.csv", dtype={"game_id": str, "choice_set_id": str}
    )
    holdout_tables = []
    audits = []
    for game_id in sorted(HOLDOUT_GAMES):
        match_output = destination / "matches" / game_id
        match_output.mkdir(parents=True, exist_ok=True)
        tracking, metadata, events = _match_files(data_root / game_id)
        table, audit = build_choice_sets(
            tracking, metadata, events, game_id=game_id, min_pre_event_lag=1, max_pre_event_lag=5
        )
        table.to_csv(match_output / "choice_sets.csv", index=False)
        (match_output / "choice_set_audit.json").write_text(json.dumps(audit, indent=2))
        holdout_tables.append(table)
        audits.append(audit)
        print(f"opened secondary holdout {game_id}: {audit['eligible_choice_sets']}", flush=True)
    holdout = pd.concat(holdout_tables, ignore_index=True)
    holdout.to_csv(destination / "choice_sets.csv", index=False)
    pd.DataFrame(audits).to_json(destination / "choice_set_audits.json", orient="records", indent=2)

    rows = []
    train_key = choice_key(train).to_numpy()
    for name, features in FEATURE_SETS.items():
        model = ConditionalSoftmax(ridge=1.0).fit(
            train[features].to_numpy(), train["chosen"].to_numpy(), train_key
        )
        for game_id, test in holdout.groupby("game_id"):
            probability = model.predict_proba(
                test[features].to_numpy(), choice_key(test).to_numpy()
            )
            rows.append({"game_id": game_id, "model": name, **evaluate_choices(test, probability)})

    features = FEATURE_SETS["B2"]
    gbm = HistGradientBoostingClassifier(
        max_depth=3, learning_rate=0.05, max_iter=150, min_samples_leaf=30, random_state=7
    ).fit(train[features], train["chosen"])
    for game_id, test in holdout.groupby("game_id"):
        test = test.reset_index(drop=True)
        raw = gbm.predict_proba(test[features])[:, 1]
        probability = _normalise_within_choice(test, raw)
        rows.append({"game_id": game_id, "model": "B4", **evaluate_choices(test, probability)})

    metrics = pd.DataFrame(rows)
    metrics.to_csv(destination / "metrics.csv", index=False)
    summary = metrics.groupby("model").agg(
        matches=("game_id", "nunique"),
        conditional_log_loss=("conditional_log_loss", "mean"),
        top1=("top1", "mean"),
        top3=("top3", "mean"),
        mrr=("mrr", "mean"),
        brier=("brier", "mean"),
    )
    summary.to_csv(destination / "summary.csv")
    print(summary.to_string())


if __name__ == "__main__":
    main()
