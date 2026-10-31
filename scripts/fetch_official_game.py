#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path


ROOT = "https://raw.githubusercontent.com/SkillCorner/opendata/{commit}/data/matches/{game_id}"
MEDIA = "https://media.githubusercontent.com/media/SkillCorner/opendata/{commit}/data/matches/{game_id}"
SUFFIXES = ("match.json", "dynamic_events.csv", "phases_of_play.csv")


def download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    print(f"download {url}")
    urllib.request.urlretrieve(url, partial)
    partial.replace(destination)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("game_id", type=int)
    parser.add_argument("--output-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    lock = json.loads(Path("datasets.lock.json").read_text())
    commit = lock["commit"]
    game_dir = args.output_root / str(args.game_id)
    for suffix in SUFFIXES:
        destination = game_dir / f"{args.game_id}_{suffix}"
        if args.force or not destination.exists():
            download(
                ROOT.format(commit=commit, game_id=args.game_id) + f"/{args.game_id}_{suffix}",
                destination,
            )
    tracking = game_dir / f"{args.game_id}_tracking_extrapolated.jsonl"
    if args.force or not tracking.exists():
        download(
            MEDIA.format(commit=commit, game_id=args.game_id)
            + f"/{args.game_id}_tracking_extrapolated.jsonl",
            tracking,
        )


if __name__ == "__main__":
    main()

