from __future__ import annotations

from collections import defaultdict

from .models import Spell


def labels_to_spells(labels: list[str], frames: list[int] | None = None) -> list[Spell]:
    if not labels:
        return []
    frames = frames or list(range(len(labels)))
    if len(frames) != len(labels):
        raise ValueError("frames and labels must have equal length")
    spells = []
    start = frames[0]
    previous_frame = frames[0]
    current = labels[0]
    for frame, label in zip(frames[1:], labels[1:]):
        if label != current or frame != previous_frame + (frames[1] - frames[0]):
            spells.append(Spell(current, start, previous_frame))
            start, current = frame, label
        previous_frame = frame
    spells.append(Spell(current, start, previous_frame))
    return spells


def temporal_iou(a: Spell, b: Spell) -> float:
    intersection = max(0, min(a.end, b.end) - max(a.start, b.start) + 1)
    union = max(a.end, b.end) - min(a.start, b.start) + 1
    return intersection / union


def spell_f1(
    truth: list[Spell],
    prediction: list[Spell],
    thresholds: tuple[float, ...] = (0.3, 0.4, 0.5, 0.6, 0.7),
) -> float:
    """Greedy one-to-one temporal-IoU F1, averaged across thresholds."""
    truth_by_label: dict[str, list[Spell]] = defaultdict(list)
    pred_by_label: dict[str, list[Spell]] = defaultdict(list)
    for spell in truth:
        truth_by_label[spell.label].append(spell)
    for spell in prediction:
        pred_by_label[spell.label].append(spell)

    scores = []
    for threshold in thresholds:
        matches = 0
        for label in set(truth_by_label) | set(pred_by_label):
            candidates = sorted(
                (
                    (temporal_iou(t, p), ti, pi)
                    for ti, t in enumerate(truth_by_label[label])
                    for pi, p in enumerate(pred_by_label[label])
                ),
                reverse=True,
            )
            used_t, used_p = set(), set()
            for iou, ti, pi in candidates:
                if iou < threshold:
                    break
                if ti not in used_t and pi not in used_p:
                    matches += 1
                    used_t.add(ti)
                    used_p.add(pi)
        denom = len(truth) + len(prediction)
        scores.append(2.0 * matches / denom if denom else 1.0)
    return sum(scores) / len(scores)

