# Research protocol v2: Defensive Access

Frozen on 30 October 2026 before the Phase 0–1 re-analysis.

## Research question

Does Defensive Access contain information about a team's available passing options beyond ordinary spatial geometry, and can the same representation support an auditable measure of defensive positioning?

The earlier result—receiver-level AUC against nearest-defender distance—is treated as exploratory. It does not by itself establish defensive value.

## Units and outcomes

The primary unit is a **decision-time choice set**: every eligible teammate available to the player in possession immediately before possession ends in a pass. Exactly one option must be labelled as received. Options from possessions without a documented received option are excluded from choice modelling and retained for coverage audits.

The primary outcome is the selected receiver within that choice set. Secondary pooled outcomes are documented `passing_option.received` labels, used only for comparability with the earlier analysis.

## Temporal eligibility and leakage control

The decision time `t0` is the end frame of the associated `player_possession` event. Every tracking feature must use the latest sampled frame satisfying:

`tracking_frame <= t0 - 1`

and no more than five source frames before `t0`. Nearest-frame alignment, which can select a post-event sample, is prohibited in the primary analysis. A falsification analysis moves the anchor a further 0.5 seconds into the past. Features that improve when sampled after `t0` are evidence of leakage and cannot be used.

An option is eligible only if its event interval includes the selected pre-decision frame, its receiver exists in tracking, and at least two eligible options remain in the choice set.

## Phase 0 audits

Before fitting a new model, report by match:

1. event-to-tracking lag and any excluded post-event alignments from the legacy pipeline;
2. pass-option distance distribution and the fraction below 3 m and 5 m;
3. results by distance bands: 0–5, 5–15, 15–30, and 30+ m;
4. valid and invalid frame coverage by period, field zone, possession side, and match-time decile where those fields are observable;
5. whether missingness is associated with observable state. If it is, report an inverse-probability-weighted sensitivity result and state the overlap limits;
6. choice-set sizes, groups with zero or multiple received options, and exclusion reasons.

`received` combines attacking choice and successful receipt. The choice-set analysis estimates selection among documented successful receipts; it is not a causal estimate of pass completion or of options that were considered but never documented.

## Baseline ladder

All fitted comparisons use identical rows and leave-one-match-out evaluation.

| Model | Information available before the pass |
| --- | --- |
| B0 | Nearest-defender distance |
| B1 | B0 + pass length + relative pass angle + receiver location |
| B2 | B1 + nearby-defender count + defenders intersecting the pass corridor |
| B3 | B2 + classic arrival-time/control-margin feature |
| B4 | Gradient-boosted model on the raw B2 geometry; a capacity reference, not the main claim |
| M | B2 + Defensive Access arrival and lane-geometry features |

Linear B0–B3 and M models use a conditional softmax likelihood over each choice set. Regularisation is fixed inside training folds or tuned with training matches only. B4 is fit only on training matches. No test-match observation participates in scaling, fitting, or tuning.

## Metrics and uncertainty

Primary metrics:

- mean conditional log loss per decision;
- top-1 and top-3 choice accuracy;
- mean reciprocal rank.

Secondary metrics:

- pooled ROC AUC for continuity with the exploratory result;
- Brier score and calibration by predicted choice-probability bins.

Report each test match separately. Aggregate uncertainty is computed by resampling matches, not option rows. With 20 matches, also report the exact sign-flip test for paired match-level differences. No frame-level p-value is permitted.

## Locked decisions and interpretation gates

1. The main incremental comparison is **M versus B2**; M versus B0 is descriptive only.
2. M must improve conditional log loss and ranking on the majority of matches, with a match-level interval compatible with a practically useful effect, before the predictive-increment claim is retained.
3. M versus B3 determines whether the custom representation adds information beyond a classic arrival-time model.
4. If B1/B2 absorbs the earlier gain, the paper pivots to an audit result: simple geometry explains the apparent improvement. It will not be presented as a successful new predictor.
5. If the 0–5 m stratum drives the result, the primary result is repeated after excluding passes shorter than 5 m.
6. An inconclusive result is described as undetected within the study's compatible effect range, not as proof of no effect.

## Secondary holdout and status

Games `2011166`, `2013725`, `2015213`, `2016236`, and `2017461` are reserved from all Phase 0–1 model development and inspected only after the pipeline and hyperparameters are frozen. These games appeared in the earlier aggregate 20-match validation, so this is a **secondary holdout for the new analysis**, not a pristine or organiser-hidden test set.

The remaining 15 games support development diagnostics and leave-one-match-out evaluation. The official competition evaluation, if any, remains the only genuinely unseen assessment.

## Phase 2–3 specification (not yet tested)

If Phase 1 passes its gate, option probabilities will be weighted by a prespecified threat proxy (progression toward goal, final-third/box entry, or a cited open xT grid) to define dangerous-option concession. Counterfactual defender-position maps will then move one non-engaging defender only within a physically reachable local grid while holding all other players fixed.

Those contrasts are model-based counterfactuals, never causal effects. The earlier unsuccessful on-ball-engager attribution remains reported as a negative result.

### Frozen Phase 2 threat definition

Before running Phase 2, the option threat proxy is fixed as:

`threat = clip(max(progression_m, 0) / 35 + 0.10 * final_third + 0.25 * penalty_area, 0, 1)`

where progression is measured toward the opponent goal using the event's documented attacking direction; final-third and penalty-area indicators use standard 105 × 68 m pitch boundaries. The 35 m scale makes a full-third forward reception saturate the continuous component. Sensitivities report progression-only and categorical-only versions.

Frame-level dangerous-option concession is the sum across simultaneously available receivers of `access_probability × threat`. It is a model score, not the probability of a shot. External-validity outcomes are fixed before inspection as the documented selected-option threat, pass EPV delta where available, opponents bypassed, and `lead_to_shot`. Comparisons use nearest-distance weighting and the B2 geometry model on identical choice sets.

### Frozen Phase 3 counterfactual

The on-ball engager is operationally the defender nearest the ball and is excluded. For each other defender, candidate locations form a ±4 m local grid at 1 m spacing, restricted to locations reachable within 1.0 s at 7 m/s from the observed point. Other players and the ball remain fixed. The primary contrast is:

`Positioning Gap = observed dangerous-option concession - minimum grid concession`

Report both the unconstrained model optimum inside that local reachable set and the observed score. This is a one-player, one-frame model contrast: it ignores teammate response, tactical instruction, offside changes and the cost of vacating the defender's current responsibility. No causal or prescriptive coaching claim is allowed without video/role review.
