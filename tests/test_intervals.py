import pytest

from defensive_access.intervals import labels_to_spells, spell_f1


def test_spell_f1_is_one_for_identical_segmentation():
    spells = labels_to_spells(["a", "a", "b", "b", "a"])
    assert spell_f1(spells, spells) == pytest.approx(1.0)


def test_fragmentation_is_penalized():
    truth = labels_to_spells(["a"] * 10)
    pred = labels_to_spells(["a"] * 4 + ["b"] + ["a"] * 5)
    assert spell_f1(truth, pred) < 1.0

