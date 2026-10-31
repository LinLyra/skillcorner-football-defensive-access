from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from defensive_access.choices import build_choice_sets
from run_phase01 import HOLDOUT_GAMES, _match_files, logo_evaluation


def option_signature(table: pd.DataFrame) -> pd.Series:
    return table.groupby(["game_id", "choice_set_id"])["receiver_id"].apply(
        lambda values: ",".join(map(str, sorted(values.astype(int))))
    )


def main() -> None:
    data_root = Path("data/raw")
    root = Path("outputs/phase01")
    early_root = root / "early_05s"
    match_root = early_root / "matches"
    match_root.mkdir(parents=True, exist_ok=True)
    primary = pd.read_csv(
        root / "development_choice_sets.csv", dtype={"game_id": str, "choice_set_id": str}
    )
    early_tables = []
    audits = []
    game_ids = sorted(
        path.name for path in data_root.iterdir() if path.is_dir() and path.name not in HOLDOUT_GAMES
    )
    for game_id in game_ids:
        destination = match_root / game_id
        destination.mkdir(parents=True, exist_ok=True)
        table_path = destination / "choice_sets.csv"
        audit_path = destination / "choice_set_audit.json"
        if table_path.exists() and audit_path.exists():
            table = pd.read_csv(table_path, dtype={"game_id": str, "choice_set_id": str})
            audit = json.loads(audit_path.read_text())
        else:
            tracking, metadata, events = _match_files(data_root / game_id)
            table, audit = build_choice_sets(
                tracking,
                metadata,
                events,
                game_id=game_id,
                min_pre_event_lag=6,
                max_pre_event_lag=10,
            )
            table.to_csv(table_path, index=False)
            audit_path.write_text(json.dumps(audit, indent=2))
        early_tables.append(table)
        audits.append(audit)
        print(f"built early anchor {game_id}: {audit['eligible_choice_sets']}", flush=True)

    early = pd.concat(early_tables, ignore_index=True)
    primary_signature = option_signature(primary)
    early_signature = option_signature(early)
    common_index = primary_signature.index.intersection(early_signature.index)
    common_index = common_index[
        primary_signature.loc[common_index].to_numpy() == early_signature.loc[common_index].to_numpy()
    ]
    common_keys = {f"{game}:{choice}" for game, choice in common_index}

    def common_subset(table: pd.DataFrame) -> pd.DataFrame:
        keys = table["game_id"].astype(str) + ":" + table["choice_set_id"].astype(str)
        return table.loc[keys.isin(common_keys)].reset_index(drop=True)

    primary_common = common_subset(primary)
    early_common = common_subset(early)
    primary_output = early_root / "primary_common"
    early_output = early_root / "early_common"
    primary_output.mkdir(parents=True, exist_ok=True)
    early_output.mkdir(parents=True, exist_ok=True)
    primary_metrics, _ = logo_evaluation(primary_common, primary_output)
    early_metrics, _ = logo_evaluation(early_common, early_output)
    summary = {
        "primary_choice_sets": int(primary.groupby(["game_id", "choice_set_id"]).ngroups),
        "early_choice_sets": int(early.groupby(["game_id", "choice_set_id"]).ngroups),
        "common_identical_choice_sets": len(common_keys),
        "primary_common": primary_metrics.groupby("model")["conditional_log_loss"].mean().to_dict(),
        "early_common": early_metrics.groupby("model")["conditional_log_loss"].mean().to_dict(),
    }
    summary["m_minus_b2"] = {
        "primary": summary["primary_common"]["M"] - summary["primary_common"]["B2"],
        "early": summary["early_common"]["M"] - summary["early_common"]["B2"],
    }
    (early_root / "summary.json").write_text(json.dumps(summary, indent=2))
    pd.DataFrame(audits).to_json(early_root / "choice_set_audits.json", orient="records", indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
