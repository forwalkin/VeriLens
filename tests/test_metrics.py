import numpy as np
import pytest

from evalkit.metrics import auc, bootstrap_auc_ci, chance_verdict, confusion, prf


def test_confusion_and_prf():
    y = np.array([1, 1, 1, 0, 0, 0, 0, 0])
    p = np.array([1, 1, 0, 1, 0, 0, 0, 0])
    c = confusion(y, p)
    assert (c.tp, c.fp, c.tn, c.fn) == (2, 1, 4, 1)
    m = prf(c)
    assert m["accuracy"] == pytest.approx(6 / 8)
    assert m["precision"] == pytest.approx(2 / 3)
    assert m["recall"] == pytest.approx(2 / 3)
    assert m["f1"] == pytest.approx(2 / 3)
    assert m["fpr"] == pytest.approx(1 / 5)


def test_prf_undefined_is_nan_not_zero():
    m = prf(confusion(np.array([0, 0]), np.array([0, 0])))
    assert np.isnan(m["precision"]) and np.isnan(m["recall"]) and np.isnan(m["f1"])


def test_auc_known_values():
    assert auc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) == 1.0
    assert auc([0, 0, 1, 1], [0.9, 0.8, 0.2, 0.1]) == 0.0
    assert auc([0, 1], [0.5, 0.5]) == 0.5
    assert auc([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8]) == pytest.approx(0.75)


def test_auc_single_class_is_nan():
    assert np.isnan(auc([1, 1, 1], [0.1, 0.2, 0.3]))


def test_chance_detected_for_random_scores():
    rng = np.random.default_rng(1)
    y = np.array([0] * 100 + [1] * 100)
    s = rng.random(200)
    ci = bootstrap_auc_ci(y, s, n_boot=300)
    assert chance_verdict(auc(y, s), ci) == "indistinguível do acaso"


def test_separable_beats_chance_and_inverted_flagged():
    rng = np.random.default_rng(2)
    y = np.array([0] * 100 + [1] * 100)
    s = np.concatenate([rng.normal(0.3, 0.1, 100), rng.normal(0.7, 0.1, 100)])
    assert chance_verdict(auc(y, s), bootstrap_auc_ci(y, s, n_boot=300)) == "melhor que o acaso"
    s_inv = 1 - s
    assert chance_verdict(auc(y, s_inv), bootstrap_auc_ci(y, s_inv, n_boot=300)).startswith("INVERTIDO")


def test_wilson_ci_bounds():
    from evalkit.metrics import wilson_ci
    lo, hi = wilson_ci(50, 100)
    assert 0.40 < lo < 0.41 and 0.59 < hi < 0.60
    assert wilson_ci(0, 10)[0] == 0.0 and wilson_ci(10, 10)[1] == 1.0
    assert np.isnan(wilson_ci(0, 0)[0])
