# Phase 2–3: dangerous-option concession and positioning counterfactuals

Threat and counterfactual definitions were frozen in commit `207d7c6` before these results were inspected. Phase 1's prespecified primary outcome remains M-versus-B2 conditional log loss. All Phase 2 outcomes below are exploratory; no multiplicity-adjusted confirmatory claim is made.

## Dangerous-option concession

Each simultaneously available receiver receives a transparent pre-pass threat value based on forward progression, final-third entry and penalty-area entry. Two frame scores are compared:

- **distance concession:** threat weighted by a monotone nearest-defender accessibility function;
- **access concession:** threat weighted by the lane-aware Defensive Access score.

External-validity models include option count, total/max available threat and distance concession before access concession is added. Every development result is leave-one-match-out. Models are then fit once on all 15 development games and evaluated on the five secondary-holdout games.

### Development (15 matches)

| Outcome | Distance model | + Access model | Increment |
| --- | ---: | ---: | ---: |
| `lead_to_shot` log loss ↓ | 0.22322 | **0.22202** | −0.00120; better in 12/15 matches |
| `lead_to_shot` AUC ↑ | 0.661 | **0.675** | +0.014 |
| Opponents bypassed RMSE ↓ | 2.13694 | **2.13032** | −0.00662 |
| Pass EPV delta RMSE ↓ | 0.014566 | **0.014564** | −0.000002 |
| Selected threat RMSE ↓ | 0.15738 | **0.15594** | −0.00144 |

### Secondary holdout (5 matches)

| Outcome | Distance model | + Access model | Increment |
| --- | ---: | ---: | ---: |
| `lead_to_shot` log loss ↓ | 0.24507 | **0.24381** | −0.00126 |
| `lead_to_shot` AUC ↑ | 0.658 | **0.663** | +0.005 |
| Opponents bypassed RMSE ↓ | 2.21959 | **2.21474** | −0.00485 |
| Pass EPV delta RMSE ↓ | 0.013717 | **0.013698** | −0.000019 |
| Selected threat RMSE ↓ | 0.16567 | **0.16540** | −0.00026 |

For `lead_to_shot`, the development match-bootstrap interval for the E2-minus-E1 log-loss difference is [−0.00220, −0.00014]. On the five secondary-holdout matches it is [−0.00386, +0.00199], crossing zero. The direction replicates in 4/5 holdout matches, but the effect is tiny and difficult to distinguish from zero. `lead_to_shot` and EPV are supplier-derived labels and may share upstream assumptions with the event data. This is a descriptive exploratory signal, not causal defensive value.

## Local defender-position counterfactual

For 40 deterministically sampled choice sets per match, the nearest-to-ball defender is excluded. Every other defender is moved over a ±4 m, 1 m-spaced local grid while all other players remain fixed. Dangerous-option concession is recomputed at each point.

- 600 development and 200 secondary-holdout frames were evaluated;
- mean defender Positioning Gap was 0.0153 in development and 0.0139 in holdout;
- adding mean/max Positioning Gap to the concession model worsened development log loss from 0.23909 to 0.24010;
- it also worsened secondary-holdout log loss from 0.24309 to 0.24532.

The heatmap therefore remains an auditable model diagnostic and video-review interface. It is **not** a validated player rating, tactical prescription or estimate of what would have happened after a defender moved. The failed predictive increment is retained rather than tuned away.

## Evidence decision

Supported:

1. strict decision-time choice sets overturn the original weak-baseline story;
2. lane-aware dangerous-option concession adds a small, replicated association with subsequent danger;
3. constrained local position maps can expose exactly which geometric assumption drives a score.

Not supported:

1. a new state-of-the-art receiver-choice model;
2. stable player ranking from 20 matches;
3. causal or coaching-optimal defender relocation;
4. generalisation beyond this competition dataset without revalidation.

This evidence hierarchy is the submission contribution: a usable audit framework that distinguishes prediction, descriptive value and model counterfactuals instead of presenting all three as the same thing.
