# Data contract

## Inputs

`*_tracking_extrapolated.jsonl` at 10 Hz:

- frame, timestamp and period;
- ball x/y/z and `is_detected`;
- possession group and player identifier;
- player x/y, identifier and `is_detected`.

`*_match.json`:

- player-to-team mapping;
- pitch dimensions;
- period and side metadata.

`*_dynamic_events.csv` for validation only:

- event interval and type/subtype;
- player and team identifiers;
- pressure, danger-reduction, forced-backward and pressing-chain outcomes.

## Coordinate policy

SkillCorner metric coordinates are retained. Algorithms must be invariant to translating or reflecting the pitch. Attack-direction normalisation is applied only for explicitly directional features and must be derived from match period metadata.

## Missingness policy

- no ball coordinate: exclude the frame from spatial access estimation;
- unknown possession team: exclude rather than infer in the primary analysis;
- fewer than seven mapped players for either team: exclude;
- extrapolated player: retain with an observation-quality flag;
- gaps in tactical states: do not linearly interpolate across possession or period changes.

## Output guarantees

Every frame output includes the original frame identifier, observation coverage, player counts, possession and defending team. Counterfactual tables retain player identifiers but never infer stable identity when none exists in the source.

