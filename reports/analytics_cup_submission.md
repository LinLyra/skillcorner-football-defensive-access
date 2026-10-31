# Analytics Cup 2.0 submission brief

## Title

**Defensive Access Under Pressure: Auditing What Tracking-Based Defensive Metrics Really Add**

## One-minute script

Defending is not only proximity to the ball; it is which dangerous options a position removes.

My first result looked impressive: lane-aware access beat nearest-defender distance in nineteen of twenty matches. But that pooled design is now superseded.

I rebuilt the analysis using options available at the same instant before each pass, strong geometry and nonlinear baselines, match-held-out testing, and a half-second timing falsification. Most of the gain disappeared; ordinary nonlinear geometry performed best.

What remained was a transparent dangerous-option concession measure, a reproducible choice-set pipeline, and local maps for video review. The danger association repeated directionally in five reserved matches, but was tiny. Positioning Gap failed outcome validation.

This is not another inflated player rating. It is an open-source framework for testing what tracking-based defensive metrics really add—and what they do not.

## Primary and exploratory outcomes

The prespecified primary Phase 1 comparison is M versus B2 conditional log loss. Its exact paired sign-flip result is `p = 0.055`, so it is treated as difficult to distinguish from zero. Phase 2 outcomes—including `lead_to_shot`, EPV, opponents bypassed and selected threat—are exploratory. The five-game secondary holdout is not organiser-hidden because earlier pooled summaries from those matches had already been seen.

## Supported claims

- Strict timing and stronger baselines explain most of the original pooled result.
- Ordinary nonlinear geometry is the strongest receiver-choice model tested.
- Lane-aware dangerous-option concession has a directionally consistent but tiny exploratory association with subsequent danger.
- The package produces reproducible choice sets, coverage audits and inspectable local position maps.

## Unsupported claims

- a state-of-the-art defensive metric;
- causal defensive value or coaching-optimal relocation;
- reliable player rankings;
- validity outside this competition dataset without revalidation.
