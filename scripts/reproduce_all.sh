#!/usr/bin/env bash
set -euo pipefail

game_ids=(
  1874553 1886347 1899585 1925299 1927964
  1953632 1959846 1986691 1996435 1996436
  2006229 2006363 2007448 2007721 2010085
  2011166 2013725 2015213 2016236 2017461
)

python -m pip install -e '.[dev]'
pytest

if [[ "${1:-}" == "--download" ]]; then
  for game_id in "${game_ids[@]}"; do
    python scripts/fetch_official_game.py "$game_id"
  done
fi

for game_id in "${game_ids[@]}"; do
  tracking="data/raw/${game_id}/${game_id}_tracking_extrapolated.jsonl"
  if [[ ! -f "$tracking" ]]; then
    echo "Missing competition data: $tracking" >&2
    echo "Run: bash scripts/reproduce_all.sh --download" >&2
    exit 2
  fi
done

PYTHONPATH=src python scripts/run_phase01.py
PYTHONPATH=src:scripts python scripts/run_leakage_sensitivity.py
PYTHONPATH=src:scripts python scripts/audit_missingness.py
PYTHONPATH=src:scripts python scripts/run_secondary_holdout.py
PYTHONPATH=src python scripts/run_threat_validation.py
PYTHONPATH=src:scripts python scripts/run_positioning_counterfactual.py
PYTHONPATH=src:scripts python scripts/make_submission_figures.py
PYTHONPATH=src python scripts/animate_choice_set.py

echo "Reproduction complete. See outputs/, reports/, reports/figures/, and reports/media/."
