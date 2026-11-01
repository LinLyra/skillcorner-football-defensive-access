from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

SUCCESS_COLUMNS = (
    "stop_possession_danger",
    "reduce_possession_danger",
    "force_backward",
)


def _as_bool(value) -> bool:
    if pd.isna(value):
        return False
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes"}
    return bool(value)


def roc_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    """AUC via average ranks; returns NaN for a single-class sample."""
    y = np.asarray(y_true, dtype=int)
    scores = np.asarray(scores, dtype=float)
    positives = int(y.sum())
    negatives = len(y) - positives
    if not positives or not negatives:
        return float("nan")
    ranks = pd.Series(scores).rank(method="average").to_numpy()
    positive_rank_sum = float(ranks[y == 1].sum())
    return (positive_rank_sum - positives * (positives + 1) / 2) / (positives * negatives)


def align_engagements(
    contributions: pd.DataFrame,
    events_path: str | Path,
    *,
    max_frame_delta: int = 5,
) -> pd.DataFrame:
    events = pd.read_csv(events_path, low_memory=False)
    events = events.loc[events["event_type"].eq("on_ball_engagement")].copy()
    contributions = contributions.sort_values("frame")
    frames = contributions["frame"].drop_duplicates().to_numpy(dtype=int)
    rows = []
    for event in events.itertuples(index=False):
        frame_start = int(event.frame_start)
        position = int(np.searchsorted(frames, frame_start))
        candidates = frames[max(0, position - 1) : min(len(frames), position + 1)]
        if not len(candidates):
            continue
        aligned_frame = int(min(candidates, key=lambda x: abs(x - frame_start)))
        if abs(aligned_frame - frame_start) > max_frame_delta:
            continue
        player_id = getattr(event, "player_id", np.nan)
        if pd.isna(player_id):
            continue
        match = contributions.loc[
            contributions["frame"].eq(aligned_frame)
            & contributions["player_id"].eq(int(player_id))
        ]
        if match.empty:
            continue
        defender = match.iloc[0]
        success = any(_as_bool(getattr(event, column, False)) for column in SUCCESS_COLUMNS)
        chain_end = getattr(event, "pressing_chain_end_type", None)
        success = success or chain_end in {"regain", "disruption"}
        rows.append(
            {
                "event_id": event.event_id,
                "event_frame": frame_start,
                "aligned_frame": aligned_frame,
                "frame_delta": aligned_frame - frame_start,
                "player_id": int(player_id),
                "event_subtype": getattr(event, "event_subtype", None),
                "success": int(success),
                "contribution": float(defender["contribution"]),
                "distance_to_ball_m": float(defender["distance_to_ball_m"]),
                "detected": bool(defender["detected"]),
            }
        )
    return pd.DataFrame(rows)


def align_passing_options(
    receiver_access: pd.DataFrame,
    events_path: str | Path,
    *,
    max_frame_delta: int = 5,
) -> pd.DataFrame:
    events = pd.read_csv(events_path, low_memory=False)
    events = events.loc[events["event_type"].eq("passing_option")].copy()
    receiver_access = receiver_access.sort_values("frame")
    frames = receiver_access["frame"].drop_duplicates().to_numpy(dtype=int)
    rows = []
    for event in events.itertuples(index=False):
        frame_start = int(event.frame_start)
        position = int(np.searchsorted(frames, frame_start))
        candidates = frames[max(0, position - 1) : min(len(frames), position + 1)]
        if not len(candidates):
            continue
        aligned_frame = int(min(candidates, key=lambda x: abs(x - frame_start)))
        if abs(aligned_frame - frame_start) > max_frame_delta:
            continue
        receiver_id = getattr(event, "player_id", np.nan)
        if pd.isna(receiver_id):
            continue
        match = receiver_access.loc[
            receiver_access["frame"].eq(aligned_frame)
            & receiver_access["receiver_id"].eq(int(receiver_id))
        ]
        if match.empty:
            continue
        access = match.iloc[0]
        rows.append(
            {
                "event_id": event.event_id,
                "event_frame": frame_start,
                "aligned_frame": aligned_frame,
                "frame_delta": aligned_frame - frame_start,
                "receiver_id": int(receiver_id),
                "received": int(_as_bool(getattr(event, "received", False))),
                "access_probability": float(access["access_probability"]),
                "nearest_defender_m": float(access["nearest_defender_m"]),
                "control_margin_s": float(access["control_margin_s"]),
                "lane_block": float(access["lane_block"]),
            }
        )
    return pd.DataFrame(rows)


def validation_summary(aligned: pd.DataFrame) -> dict:
    if aligned.empty:
        return {"aligned_events": 0}
    y = aligned["success"].to_numpy()
    contribution_auc = roc_auc(y, aligned["contribution"].to_numpy())
    proximity_auc = roc_auc(y, -aligned["distance_to_ball_m"].to_numpy())
    return {
        "aligned_events": len(aligned),
        "successful_events": int(y.sum()),
        "success_rate": float(y.mean()),
        "contribution_auc": None if math.isnan(contribution_auc) else contribution_auc,
        "proximity_auc": None if math.isnan(proximity_auc) else proximity_auc,
        "auc_difference": (
            None
            if math.isnan(contribution_auc) or math.isnan(proximity_auc)
            else contribution_auc - proximity_auc
        ),
        "detected_fraction": float(aligned["detected"].mean()),
        "median_abs_alignment_frames": float(aligned["frame_delta"].abs().median()),
    }


def passing_option_summary(aligned: pd.DataFrame) -> dict:
    if aligned.empty:
        return {"aligned_passing_options": 0}
    y = aligned["received"].to_numpy()
    access_auc = roc_auc(y, aligned["access_probability"].to_numpy())
    distance_auc = roc_auc(y, aligned["nearest_defender_m"].to_numpy())
    return {
        "aligned_passing_options": len(aligned),
        "received_options": int(y.sum()),
        "received_rate": float(y.mean()),
        "access_auc": None if math.isnan(access_auc) else access_auc,
        "distance_auc": None if math.isnan(distance_auc) else distance_auc,
        "auc_difference": (
            None
            if math.isnan(access_auc) or math.isnan(distance_auc)
            else access_auc - distance_auc
        ),
        "median_abs_alignment_frames": float(aligned["frame_delta"].abs().median()),
    }

