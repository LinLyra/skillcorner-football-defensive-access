# Phase 0–1 validation: from pooled options to real choice sets

Protocol frozen in commit `d361a40` before this analysis. The five secondary-holdout games were opened only after feature definitions, `t0-1` alignment, ridge strength, GBM settings and outputs were fixed. They are not pristine organiser-hidden data because their earlier pooled AUCs had been seen.

## What changed

The earlier analysis aligned each `passing_option` to its nearest sampled tracking frame and pooled options across time. That design could select a post-event frame, compare options that were not simultaneously available, and reward a model for distinguishing easy frames from hard frames.

The replacement design:

- defines `t0` as the associated `player_possession.frame_end`;
- uses only the latest valid 10 Hz frame from `t0-1` through `t0-5`;
- requires every compared option's documented interval to include that same frame;
- requires exactly one received option and at least two tracked options;
- evaluates conditional probability and rank within the decision-time choice set;
- trains on whole matches and tests on unseen matches.

## Cohort audit

The 15 development games produced 6,087 eligible choice sets and 15,012 options. Mean match coverage was 59.4% of possessions with exactly one documented received option (range 53.3%–63.5%). The main exclusion is that two documented options do not coexist at a common pre-pass frame, not missing ball tracking.

Only 3.39% of all eligible options and 3.29% of selected options were shorter than 5 m. Removing choice sets whose selected pass was below 5 m did not materially change the model ordering.

Across all sampled match frames, mean validity was 53.7%. This is mostly a target-population boundary: 45.2% had no team-in-possession label, while only 1.05% had known possession but missing ball coordinates. Among frames with known possession, 98.1% were eligible. Inverse-probability weighting is therefore not used to make dead-ball/unknown-possession frames resemble live passing decisions.

## Baseline ladder: 15-match development LOGO

| Model | Conditional log loss ↓ | Top-1 ↑ | MRR ↑ |
| --- | ---: | ---: | ---: |
| B0 nearest defender | 0.8558 | 0.5102 | 0.7345 |
| B1 ordinary receiver geometry | 0.8444 | 0.5421 | 0.7516 |
| B2 + local/corridor defender counts | 0.8280 | 0.5502 | 0.7573 |
| B3 + control margin | 0.8280 | 0.5502 | 0.7573 |
| M + arrival/lane access | 0.8259 | 0.5467 | 0.7557 |
| B4 nonlinear raw-geometry GBM | **0.8201** | **0.5637** | **0.7638** |

B3 equals B2 because, under fixed ball and defender speeds, control margin is an exact linear combination of nearest-defender distance and pass distance. Presenting it as independent information would be double counting.

M improves mean log loss over B2 by 0.00215 in 11/15 matches. The match-bootstrap 95% interval is [−0.00390, −0.00000] for `M − B2`, but the exact paired sign-flip p-value is 0.055. M does not improve top-1 or MRR over B2 and is worse than B4 by 0.00578 log-loss units on average.

## Prespecified 0.5-second-earlier sensitivity

Moving the anchor from `t0-1` to `t0-6…t0-10` reduced eligible choice sets from 6,087 to 4,685. To avoid a changing-cohort comparison, the analysis retained 3,113 possessions with identical receiver sets at both anchors and refit every model by leave-one-match-out.

On this identical cohort, `M − B2` log loss was −0.00029 immediately before the pass and +0.00038 half a second earlier. The custom access increment therefore does not survive the temporal falsification. The near-pass result may describe late option formation or receiver/defender response, but it is not evidence of stable defensive positioning quality.

## Secondary holdout

Models fit once on all 15 development matches were evaluated on five reserved matches (1,530 choice sets):

| Model | Conditional log loss ↓ | Top-1 ↑ | MRR ↑ |
| --- | ---: | ---: | ---: |
| B0 | 0.8551 | 0.5179 | 0.7384 |
| B1 | 0.8488 | 0.5468 | 0.7522 |
| B2 | 0.8402 | 0.5697 | 0.7647 |
| B3 | 0.8402 | 0.5697 | 0.7647 |
| M | 0.8376 | 0.5664 | 0.7641 |
| B4 | **0.8259** | **0.5759** | **0.7696** |

M's small log-loss improvement over B2 replicated (−0.00259), but its top-1 accuracy remained lower. B4 was the strongest model.

## Decision

The original claim—“Defensive Access predicts received options beyond nearest-defender distance”—was true against a weak pooled baseline but insufficient. Under the frozen strong evaluation, most of the gain is explained by ordinary geometry and nonlinearity. The remaining lane feature produces a small probability-loss improvement close to the pass, not a robust ranking improvement or a validated defensive-position value.

The project therefore pivots from a prediction-win story to an auditable research result:

1. pooled option AUC can overstate defensive-model value;
2. choice-set construction and pre-event timing materially change the conclusion;
3. fixed-speed control margin is redundant with its inputs in a linear model;
4. the next contribution must directly value dangerous options and test constrained defender-position counterfactuals. No causal claim is made.
