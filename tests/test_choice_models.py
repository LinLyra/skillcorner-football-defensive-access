import numpy as np
import pandas as pd

from defensive_access.choice_models import ConditionalSoftmax, evaluate_choices


def test_conditional_softmax_learns_within_set_ranking():
    x = np.array([[2.0], [0.0], [3.0], [0.0], [4.0], [0.0]])
    y = np.array([1, 0, 1, 0, 1, 0])
    keys = np.array(["a", "a", "b", "b", "c", "c"])
    model = ConditionalSoftmax(ridge=0.01).fit(x, y, keys)
    probabilities = model.predict_proba(x, keys)
    assert all(probabilities[[0, 2, 4]] > 0.5)


def test_choice_metrics_are_computed_per_choice_set():
    table = pd.DataFrame(
        {
            "game_id": [1, 1, 1, 1],
            "choice_set_id": ["a", "a", "b", "b"],
            "chosen": [1, 0, 0, 1],
        }
    )
    metrics = evaluate_choices(table, np.array([0.8, 0.2, 0.6, 0.4]))
    assert metrics["choice_sets"] == 2
    assert metrics["top1"] == 0.5
    assert metrics["mrr"] == 0.75
