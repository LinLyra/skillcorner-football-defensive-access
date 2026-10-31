from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from defensive_access.choice_models import (
    FEATURE_SETS,
    ConditionalSoftmax,
    choice_key,
    evaluate_choices,
)
from defensive_access.choices import build_choice_sets


HOLDOUT_GAMES = {"2011166", "2013725", "2015213", "2016236", "2017461"}


def _match_files(match_dir: Path) -> tuple[Path, Path, Path]:
    game_id = match_dir.name
    return (
        match_dir / f"{game_id}_tracking_extrapolated.jsonl",
        match_dir / f"{game_id}_match.json",
        match_dir / f"{game_id}_dynamic_events.csv",
    )


def build_tables(data_root: Path, output: Path, game_ids: list[str]) -> pd.DataFrame:
    tables = []
    audits = []
    match_output = output / "matches"
    match_output.mkdir(parents=True, exist_ok=True)
    for game_id in game_ids:
        destination = match_output / game_id
        table_path = destination / "choice_sets.csv"
        audit_path = destination / "choice_set_audit.json"
        if table_path.exists() and audit_path.exists():
            table = pd.read_csv(table_path, dtype={"game_id": str, "choice_set_id": str})
            audit = json.loads(audit_path.read_text())
            if "threat" not in table.columns:
                tracking, metadata, events = _match_files(data_root / game_id)
                table, audit = build_choice_sets(
                    tracking,
                    metadata,
                    events,
                    game_id=game_id,
                    min_pre_event_lag=1,
                    max_pre_event_lag=5,
                )
                table.to_csv(table_path, index=False)
                audit_path.write_text(json.dumps(audit, indent=2))
        else:
            destination.mkdir(parents=True, exist_ok=True)
            tracking, metadata, events = _match_files(data_root / game_id)
            table, audit = build_choice_sets(
                tracking, metadata, events, game_id=game_id, min_pre_event_lag=1, max_pre_event_lag=5
            )
            table.to_csv(table_path, index=False)
            audit_path.write_text(json.dumps(audit, indent=2))
        tables.append(table)
        audits.append(audit)
        print(f"built {game_id}: {audit['eligible_choice_sets']} choice sets", flush=True)
    combined = pd.concat(tables, ignore_index=True)
    combined.to_csv(output / "development_choice_sets.csv", index=False)
    pd.DataFrame(audits).to_json(output / "choice_set_audits.json", orient="records", indent=2)
    return combined


def _normalise_within_choice(table: pd.DataFrame, scores: np.ndarray) -> np.ndarray:
    probabilities = np.zeros(len(table), dtype=float)
    key = choice_key(table)
    for _, index in key.groupby(key, sort=False).groups.items():
        local = np.asarray(scores[list(index)], dtype=float)
        local = np.clip(local, 1e-9, None)
        probabilities[list(index)] = local / local.sum()
    return probabilities


def logo_evaluation(table: pd.DataFrame, output: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    metric_rows = []
    prediction_rows = []
    for game_id in sorted(table["game_id"].astype(str).unique()):
        train = table.loc[table["game_id"].astype(str).ne(game_id)].copy()
        test = table.loc[table["game_id"].astype(str).eq(game_id)].copy()
        train_key = choice_key(train).to_numpy()
        test_key = choice_key(test).to_numpy()
        for name, features in FEATURE_SETS.items():
            model = ConditionalSoftmax(ridge=1.0).fit(
                train[features].to_numpy(), train["chosen"].to_numpy(), train_key
            )
            probabilities = model.predict_proba(test[features].to_numpy(), test_key)
            metrics = evaluate_choices(test, probabilities)
            metric_rows.append({"game_id": game_id, "model": name, **metrics})
            prediction_rows.extend(
                {
                    "game_id": game_id,
                    "choice_set_id": choice_set_id,
                    "event_id": event_id,
                    "model": name,
                    "chosen": int(chosen),
                    "probability": float(probability),
                    "pass_distance_m": float(distance),
                }
                for choice_set_id, event_id, chosen, probability, distance in zip(
                    test["choice_set_id"],
                    test["event_id"],
                    test["chosen"],
                    probabilities,
                    test["pass_distance_m"],
                )
            )

        features = FEATURE_SETS["B2"]
        gbm = HistGradientBoostingClassifier(
            max_depth=3, learning_rate=0.05, max_iter=150, min_samples_leaf=30, random_state=7
        )
        gbm.fit(train[features], train["chosen"])
        raw = gbm.predict_proba(test[features])[:, 1]
        probabilities = _normalise_within_choice(test.reset_index(drop=True), raw)
        metrics = evaluate_choices(test.reset_index(drop=True), probabilities)
        metric_rows.append({"game_id": game_id, "model": "B4", **metrics})
        prediction_rows.extend(
            {
                "game_id": game_id,
                "choice_set_id": choice_set_id,
                "event_id": event_id,
                "model": "B4",
                "chosen": int(chosen),
                "probability": float(probability),
                "pass_distance_m": float(distance),
            }
            for choice_set_id, event_id, chosen, probability, distance in zip(
                test["choice_set_id"],
                test["event_id"],
                test["chosen"],
                probabilities,
                test["pass_distance_m"],
            )
        )
        print(f"evaluated held-out match {game_id}", flush=True)

    metrics = pd.DataFrame(metric_rows)
    predictions = pd.DataFrame(prediction_rows)
    metrics.to_csv(output / "logo_metrics.csv", index=False)
    predictions.to_csv(output / "logo_predictions.csv", index=False)
    return metrics, predictions


def audit_short_options(table: pd.DataFrame, output: Path) -> dict:
    selected = table.loc[table["chosen"].eq(1), "pass_distance_m"]
    bands = pd.cut(
        selected,
        bins=[-np.inf, 5.0, 15.0, 30.0, np.inf],
        labels=["0-5", "5-15", "15-30", "30+"],
        right=False,
    )
    audit = {
        "options": int(len(table)),
        "choice_sets": int(table.groupby(["game_id", "choice_set_id"]).ngroups),
        "all_options_below_3m_fraction": float(table["pass_distance_m"].lt(3).mean()),
        "all_options_below_5m_fraction": float(table["pass_distance_m"].lt(5).mean()),
        "selected_below_3m_fraction": float(selected.lt(3).mean()),
        "selected_below_5m_fraction": float(selected.lt(5).mean()),
        "selected_distance_median_m": float(selected.median()),
        "selected_distance_bands": {k: int(v) for k, v in bands.value_counts(sort=False).items()},
    }
    (output / "short_option_audit.json").write_text(json.dumps(audit, indent=2))
    return audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--output", type=Path, default=Path("outputs/phase01"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    game_ids = sorted(path.name for path in args.data_root.iterdir() if path.is_dir())
    development = [game_id for game_id in game_ids if game_id not in HOLDOUT_GAMES]
    table = build_tables(args.data_root, args.output, development)
    audit = audit_short_options(table, args.output)
    metrics, _ = logo_evaluation(table, args.output)
    summary = metrics.groupby("model").agg(
        matches=("game_id", "nunique"),
        conditional_log_loss=("conditional_log_loss", "mean"),
        top1=("top1", "mean"),
        top3=("top3", "mean"),
        mrr=("mrr", "mean"),
        brier=("brier", "mean"),
    )
    summary.to_csv(args.output / "development_model_summary.csv")
    print(json.dumps(audit, indent=2))
    print(summary.to_string())


if __name__ == "__main__":
    main()
