from __future__ import annotations

from collections import Counter
from pathlib import Path

import pandas as pd

from .features import AccessConfig, analyse_frame
from .io import iter_tracking
from .validation import _as_bool


def _event_choice_groups(events: pd.DataFrame) -> tuple[list[dict], Counter]:
    options = events.loc[events["event_type"].eq("passing_option")].copy()
    possessions = events.loc[events["event_type"].eq("player_possession")].copy()
    possession_by_id = possessions.set_index("event_id")
    exclusions: Counter = Counter()
    groups: list[dict] = []

    options = options.loc[options["associated_player_possession_event_id"].notna()]
    for possession_id, group in options.groupby("associated_player_possession_event_id", sort=False):
        received = group["received"].map(_as_bool)
        if int(received.sum()) != 1:
            exclusions["not_exactly_one_received"] += 1
            continue
        if possession_id not in possession_by_id.index:
            exclusions["missing_possession_end"] += 1
            continue
        parent = possession_by_id.loc[possession_id]
        end = parent.get("frame_end")
        if pd.isna(end):
            exclusions["missing_possession_end"] += 1
            continue
        if parent.get("attacking_side") not in {"left_to_right", "right_to_left"}:
            exclusions["missing_attacking_side"] += 1
            continue
        groups.append(
            {
                "choice_set_id": str(possession_id),
                "decision_frame": int(end),
                "attacking_side": parent.get("attacking_side"),
                "lead_to_shot": int(_as_bool(parent.get("lead_to_shot", False))),
                "pass_epv_delta_for": parent.get("pass_epv_delta_for"),
                "n_opponents_bypassed": parent.get("n_opponents_bypassed"),
                "options": group.assign(_chosen=received.to_numpy()),
            }
        )
    return groups, exclusions


def build_choice_sets(
    tracking_path: str | Path,
    metadata_path: str | Path,
    events_path: str | Path,
    *,
    game_id: str | int | None = None,
    min_pre_event_lag: int = 1,
    max_pre_event_lag: int = 5,
    config: AccessConfig | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Build common-time receiver choice sets using only frames before the pass.

    Candidate frames are t0-1 through t0-max_pre_event_lag. A candidate must be a
    valid tracking frame and contain the selected receiver plus at least one other
    passing option whose documented interval covers that exact frame.
    """
    config = config or AccessConfig()
    events = pd.read_csv(events_path, low_memory=False)
    groups, exclusions = _event_choice_groups(events)
    target_frames = {
        group["decision_frame"] - lag
        for group in groups
        for lag in range(min_pre_event_lag, max_pre_event_lag + 1)
    }

    analysed: dict[int, dict] = {}
    invalid_reasons: Counter = Counter()
    for frame in iter_tracking(tracking_path, metadata_path):
        if frame.frame not in target_frames:
            continue
        row = analyse_frame(frame, config)
        if row["valid"]:
            analysed[frame.frame] = row
        else:
            invalid_reasons[row["reason"]] += 1

    rows: list[dict] = []
    used_groups = 0
    for group in groups:
        decision_frame = group["decision_frame"]
        options = group["options"]
        selected_frame = None
        selected_rows = None
        for lag in range(min_pre_event_lag, max_pre_event_lag + 1):
            frame_id = decision_frame - lag
            frame = analysed.get(frame_id)
            if frame is None:
                continue
            active = options.loc[
                options["frame_start"].le(frame_id) & options["frame_end"].ge(frame_id)
            ].copy()
            if int(active["_chosen"].sum()) != 1 or len(active) < 2:
                continue
            access_by_receiver = {
                int(item["receiver_id"]): item for item in frame["receiver_rows"]
            }
            active = active.loc[active["player_id"].map(access_by_receiver.__contains__)]
            if int(active["_chosen"].sum()) != 1 or len(active) < 2:
                continue
            selected_frame = frame_id
            selected_rows = (active, access_by_receiver)
            break

        if selected_rows is None or selected_frame is None:
            exclusions["no_eligible_pre_event_choice_set"] += 1
            continue

        active, access_by_receiver = selected_rows
        used_groups += 1
        attacking_sign = 1.0 if group["attacking_side"] == "left_to_right" else -1.0
        for _, event in active.iterrows():
            receiver_id = int(event["player_id"])
            access = access_by_receiver[receiver_id]
            progression = attacking_sign * float(access["pass_dx_m"])
            attacking_x = attacking_sign * float(access["receiver_x"])
            final_third = int(attacking_x >= 17.5)
            penalty_area = int(attacking_x >= 36.0 and abs(float(access["receiver_y"])) <= 20.16)
            threat_progression = max(progression, 0.0) / 35.0
            threat_categorical = 0.10 * final_third + 0.25 * penalty_area
            threat = min(max(threat_progression + threat_categorical, 0.0), 1.0)
            rows.append(
                {
                    "game_id": str(game_id) if game_id is not None else "",
                    "choice_set_id": group["choice_set_id"],
                    "decision_frame": decision_frame,
                    "aligned_frame": selected_frame,
                    "pre_event_lag_frames": decision_frame - selected_frame,
                    "event_id": event["event_id"],
                    "receiver_id": receiver_id,
                    "chosen": int(bool(event["_chosen"])),
                    "attacking_sign": attacking_sign,
                    "progression_m": progression,
                    "final_third": final_third,
                    "penalty_area": penalty_area,
                    "threat_progression": threat_progression,
                    "threat_categorical": threat_categorical,
                    "threat": threat,
                    "lead_to_shot": group["lead_to_shot"],
                    "pass_epv_delta_for": group["pass_epv_delta_for"],
                    "n_opponents_bypassed": group["n_opponents_bypassed"],
                    **access,
                }
            )

    table = pd.DataFrame(rows)
    summary = {
        "game_id": str(game_id) if game_id is not None else None,
        "candidate_choice_sets": len(groups),
        "eligible_choice_sets": used_groups,
        "eligible_options": len(table),
        "choice_set_coverage": used_groups / len(groups) if groups else 0.0,
        "exclusions": dict(exclusions),
        "invalid_candidate_frames": dict(invalid_reasons),
    }
    if not table.empty:
        sizes = table.groupby("choice_set_id").size()
        summary.update(
            {
                "mean_choice_set_size": float(sizes.mean()),
                "median_choice_set_size": float(sizes.median()),
                "mean_pre_event_lag_frames": float(
                    table.groupby("choice_set_id")["pre_event_lag_frames"].first().mean()
                ),
            }
        )
    return table, summary
