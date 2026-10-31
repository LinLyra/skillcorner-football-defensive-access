import numpy as np
import pytest

from defensive_access.validation import roc_auc


def test_auc_is_one_for_perfect_ranking():
    assert roc_auc(np.array([0, 0, 1, 1]), np.array([0.1, 0.2, 0.8, 0.9])) == pytest.approx(1)


def test_auc_is_half_for_ties():
    assert roc_auc(np.array([0, 1]), np.array([0.5, 0.5])) == pytest.approx(0.5)

