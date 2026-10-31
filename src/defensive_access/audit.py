from __future__ import annotations

from collections import Counter

from .features import AccessConfig, analyse_frame
from .io import iter_tracking


def audit_match(tracking_path, metadata_path, *, stride: int = 5, config=None) -> tuple[list, dict]:
    config = config or AccessConfig()
    valid_rows, reasons = [], Counter()
    total = 0
    for frame in iter_tracking(tracking_path, metadata_path, stride=stride):
        total += 1
        row = analyse_frame(frame, config)
        if row["valid"]:
            valid_rows.append(row)
        else:
            reasons[row["reason"]] += 1
    summary = {
        "sampled_frames": total,
        "valid_frames": len(valid_rows),
        "valid_fraction": len(valid_rows) / total if total else 0.0,
        "exclusion_reasons": dict(reasons),
    }
    return valid_rows, summary

