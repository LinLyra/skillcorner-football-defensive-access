#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def run(command: list[str]) -> None:
    subprocess.run(command, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("game_id", type=int)
    parser.add_argument("--stride", type=int, default=5)
    args = parser.parse_args()
    game = str(args.game_id)
    raw = Path("data/raw") / game
    output = Path("outputs") / game
    run(
        [
            "defensive-access",
            "audit-match",
            str(raw / f"{game}_tracking_extrapolated.jsonl"),
            str(raw / f"{game}_match.json"),
            "--stride",
            str(args.stride),
            "--output-dir",
            str(output),
        ]
    )
    events = str(raw / f"{game}_dynamic_events.csv")
    run(
        [
            "defensive-access",
            "validate-options",
            str(output / "receiver_access.csv"),
            events,
            "--output-dir",
            str(output),
        ]
    )
    run(
        [
            "defensive-access",
            "validate-events",
            str(output / "defender_contributions.csv"),
            events,
            "--output-dir",
            str(output),
        ]
    )


if __name__ == "__main__":
    main()

