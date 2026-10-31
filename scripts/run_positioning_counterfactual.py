from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from defensive_access.counterfactual import defender_positioning_gaps
from defensive_access.io import iter_tracking
from defensive_access.threat import aggregate_dangerous_options
from run_phase01 import _match_files


BASE_FEATURES = [
    "option_count",
    "sum_threat",
    "max_threat",
    "distance_concession",
    "access_concession",
]
GAP_FEATURES = BASE_FEATURES + ["mean_positioning_gap", "max_positioning_gap"]


def stable_sample(options: pd.DataFrame, per_match: int = 40) -> pd.DataFrame:
    choices = options[["game_id", "choice_set_id", "aligned_frame"]].drop_duplicates().copy()
    choices["sample_key"] = choices.apply(
        lambda row: hashlib.sha256(
            f"{row['game_id']}:{row['choice_set_id']}:phase3".encode()
        ).hexdigest(),
        axis=1,
    )
    return (
        choices.sort_values(["game_id", "sample_key"])
        .groupby("game_id", as_index=False, group_keys=False)
        .head(per_match)
    )


def compute_gaps(options: pd.DataFrame, data_root: Path, per_match: int = 40) -> pd.DataFrame:
    sampled = stable_sample(options, per_match)
    rows = []
    for game_id, selected in sampled.groupby("game_id"):
        match_options = options.loc[options["game_id"].astype(str).eq(str(game_id))]
        targets = set(selected["aligned_frame"].astype(int))
        tracking, metadata, _ = _match_files(data_root / str(game_id))
        frame_by_id = {
            frame.frame: frame
            for frame in iter_tracking(tracking, metadata)
            if frame.frame in targets
        }
        for choice in selected.itertuples(index=False):
            frame = frame_by_id.get(int(choice.aligned_frame))
            if frame is None:
                continue
            group = match_options.loc[
                match_options["choice_set_id"].astype(str).eq(str(choice.choice_set_id))
            ]
            receiver_threat = dict(
                zip(group["receiver_id"].astype(int), group["threat"].astype(float))
            )
            for result in defender_positioning_gaps(frame, receiver_threat):
                rows.append(
                    {
                        "game_id": str(game_id),
                        "choice_set_id": str(choice.choice_set_id),
                        "aligned_frame": int(choice.aligned_frame),
                        **result,
                    }
                )
        print(f"counterfactual grid {game_id}: {len(selected)} sampled choices", flush=True)
    return pd.DataFrame(rows)


def frame_gap_table(options: pd.DataFrame, gaps: pd.DataFrame) -> pd.DataFrame:
    decisions = aggregate_dangerous_options(options)
    gap_summary = gaps.groupby(["game_id", "choice_set_id"], as_index=False).agg(
        defenders_evaluated=("defender_id", "size"),
        mean_positioning_gap=("positioning_gap", "mean"),
        max_positioning_gap=("positioning_gap", "max"),
    )
    return decisions.merge(gap_summary, on=["game_id", "choice_set_id"], how="inner")


def evaluate_development(table: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for game_id in sorted(table["game_id"].astype(str).unique()):
        train = table.loc[table["game_id"].astype(str).ne(game_id)]
        test = table.loc[table["game_id"].astype(str).eq(game_id)]
        for name, features in {"C0_concession": BASE_FEATURES, "C1_gap": GAP_FEATURES}.items():
            model = make_pipeline(
                StandardScaler(), LogisticRegression(C=1.0, max_iter=1000)
            ).fit(train[features], train["lead_to_shot"].astype(int))
            probability = model.predict_proba(test[features])[:, 1]
            rows.append(
                {
                    "game_id": game_id,
                    "model": name,
                    "n": len(test),
                    "log_loss": log_loss(test["lead_to_shot"], probability, labels=[0, 1]),
                    "auc": roc_auc_score(test["lead_to_shot"], probability),
                }
            )
    return pd.DataFrame(rows)


def evaluate_holdout(train: pd.DataFrame, holdout: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name, features in {"C0_concession": BASE_FEATURES, "C1_gap": GAP_FEATURES}.items():
        model = make_pipeline(
            StandardScaler(), LogisticRegression(C=1.0, max_iter=1000)
        ).fit(train[features], train["lead_to_shot"].astype(int))
        for game_id, test in holdout.groupby("game_id"):
            probability = model.predict_proba(test[features])[:, 1]
            rows.append(
                {
                    "game_id": game_id,
                    "model": name,
                    "n": len(test),
                    "log_loss": log_loss(test["lead_to_shot"], probability, labels=[0, 1]),
                    "auc": roc_auc_score(test["lead_to_shot"], probability),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    data_root = Path("data/raw")
    output = Path("outputs/phase03")
    output.mkdir(parents=True, exist_ok=True)
    development_options = pd.read_csv(
        "outputs/phase01/development_choice_sets.csv",
        dtype={"game_id": str, "choice_set_id": str},
    )
    holdout_options = pd.read_csv(
        "outputs/phase01/secondary_holdout/choice_sets.csv",
        dtype={"game_id": str, "choice_set_id": str},
    )
    development_gaps = compute_gaps(development_options, data_root)
    holdout_gaps = compute_gaps(holdout_options, data_root)
    development_gaps.to_csv(output / "development_defender_gaps.csv", index=False)
    holdout_gaps.to_csv(output / "holdout_defender_gaps.csv", index=False)
    development = frame_gap_table(development_options, development_gaps)
    holdout = frame_gap_table(holdout_options, holdout_gaps)
    development.to_csv(output / "development_frame_gaps.csv", index=False)
    holdout.to_csv(output / "holdout_frame_gaps.csv", index=False)

    development_metrics = evaluate_development(development)
    holdout_metrics = evaluate_holdout(development, holdout)
    development_metrics.to_csv(output / "development_metrics.csv", index=False)
    holdout_metrics.to_csv(output / "holdout_metrics.csv", index=False)
    summary = {
        "development_choices": len(development),
        "holdout_choices": len(holdout),
        "development_mean_gap": float(development["mean_positioning_gap"].mean()),
        "holdout_mean_gap": float(holdout["mean_positioning_gap"].mean()),
        "development_metrics": development_metrics.groupby("model")[["log_loss", "auc"]]
        .mean()
        .to_dict("index"),
        "holdout_metrics": holdout_metrics.groupby("model")[["log_loss", "auc"]]
        .mean()
        .to_dict("index"),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
