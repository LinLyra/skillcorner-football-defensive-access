from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .audit import audit_match
from .choices import build_choice_sets
from .temporal import DefensiveStateDecoder, state_emission_costs
from .validation import (
    align_engagements,
    align_passing_options,
    passing_option_summary,
    validation_summary,
)


def _flatten(rows: list[dict]) -> pd.DataFrame:
    excluded = {"receiver_rows", "defender_contributions"}
    return pd.DataFrame([{k: v for k, v in row.items() if k not in excluded} for row in rows])


def command_audit(args) -> None:
    rows, summary = audit_match(args.tracking, args.metadata, stride=args.stride)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if rows:
        states = DefensiveStateDecoder(switch_penalty=args.switch_penalty).decode(
            state_emission_costs(rows)
        )
        frame_table = _flatten(rows)
        frame_table["heuristic_state_experimental"] = states
        frame_table.to_csv(output / "frame_metrics.csv", index=False)
        defender_rows = []
        receiver_rows = []
        for row in rows:
            for item in row["defender_contributions"]:
                defender_rows.append({"frame": row["frame"], **item})
            for item in row["receiver_rows"]:
                receiver_rows.append({"frame": row["frame"], **item})
        pd.DataFrame(defender_rows).to_csv(output / "defender_contributions.csv", index=False)
        pd.DataFrame(receiver_rows).to_csv(output / "receiver_access.csv", index=False)
    (output / "audit.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


def command_validate(args) -> None:
    contributions = pd.read_csv(args.contributions)
    aligned = align_engagements(contributions, args.events, max_frame_delta=args.max_frame_delta)
    summary = validation_summary(aligned)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    aligned.to_csv(output / "aligned_engagements.csv", index=False)
    (output / "validation.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


def command_validate_options(args) -> None:
    receiver_access = pd.read_csv(args.receiver_access)
    aligned = align_passing_options(receiver_access, args.events, max_frame_delta=args.max_frame_delta)
    summary = passing_option_summary(aligned)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    aligned.to_csv(output / "aligned_passing_options.csv", index=False)
    (output / "passing_option_validation.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


def command_build_choices(args) -> None:
    table, summary = build_choice_sets(
        args.tracking,
        args.metadata,
        args.events,
        game_id=args.game_id,
        min_pre_event_lag=args.min_pre_event_lag,
        max_pre_event_lag=args.max_pre_event_lag,
    )
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    table.to_csv(output / "choice_sets.csv", index=False)
    (output / "choice_set_audit.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="defensive-access")
    sub = parser.add_subparsers(dest="command", required=True)
    audit = sub.add_parser("audit-match", help="Build frame, receiver and defender audit tables")
    audit.add_argument("tracking")
    audit.add_argument("metadata")
    audit.add_argument("--output-dir", default="outputs/audit")
    audit.add_argument("--stride", type=int, default=5, help="Sample every Nth 10 Hz frame")
    audit.add_argument("--switch-penalty", type=float, default=2.2)
    audit.set_defaults(func=command_audit)
    validate = sub.add_parser("validate-events", help="Align DAC with defensive event outcomes")
    validate.add_argument("contributions")
    validate.add_argument("events")
    validate.add_argument("--output-dir", default="outputs/validation")
    validate.add_argument("--max-frame-delta", type=int, default=5)
    validate.set_defaults(func=command_validate)
    options = sub.add_parser("validate-options", help="Validate receiver access on passing options")
    options.add_argument("receiver_access")
    options.add_argument("events")
    options.add_argument("--output-dir", default="outputs/validation")
    options.add_argument("--max-frame-delta", type=int, default=5)
    options.set_defaults(func=command_validate_options)
    choices = sub.add_parser(
        "build-choice-sets", help="Build strict pre-pass, common-frame receiver choice sets"
    )
    choices.add_argument("tracking")
    choices.add_argument("metadata")
    choices.add_argument("events")
    choices.add_argument("--game-id")
    choices.add_argument("--output-dir", default="outputs/choices")
    choices.add_argument("--min-pre-event-lag", type=int, default=1)
    choices.add_argument("--max-pre-event-lag", type=int, default=5)
    choices.set_defaults(func=command_build_choices)
    return parser


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()

