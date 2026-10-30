import numpy as np
import pytest

from ragkv.allocation import allocate, weights_from_scores


@pytest.mark.parametrize("mode", ["softmax", "rank"])
def test_weights_are_a_distribution_and_follow_the_scores(mode):
    s = np.array([2.0, -1.0, 0.3, 0.9])
    w = weights_from_scores(s, mode, temperature=1.0)
    assert w.sum() == pytest.approx(1.0)
    assert (w > 0).all()
    assert list(np.argsort(-w)) == list(np.argsort(-s))


def test_flat_scores_give_uniform_weights():
    assert weights_from_scores(np.full(5, 3.0)) == pytest.approx(np.full(5, 0.2))


def test_temperature_moves_between_uniform_and_peaked():
    s = np.array([2.0, 0.0, -2.0])
    cold = weights_from_scores(s, temperature=0.2)
    hot = weights_from_scores(s, temperature=50.0)
    assert cold.max() > 0.95
    assert hot == pytest.approx(np.full(3, 1 / 3), abs=0.01)


def test_softmax_is_scale_free():
    s = np.array([1.0, 0.2, -0.5, 0.0])
    assert weights_from_scores(s) == pytest.approx(weights_from_scores(37 * s + 5))


@pytest.mark.parametrize("total", [0, 1, 17, 150, 299, 300])
def test_budgets_add_up_and_respect_caps(total):
    rng = np.random.RandomState(total)
    lengths = rng.randint(5, 120, size=7)
    w = weights_from_scores(rng.randn(7))
    b = allocate(total, lengths, w, floor=4)
    assert b.sum() == min(total, lengths.sum())
    assert (b >= 0).all() and (b <= lengths).all()


def test_short_chunk_overflow_goes_to_the_others():
    b = allocate(120, [20, 200, 200], weights_from_scores(np.array([5.0, 0.0, 0.0])), floor=4)
    assert b[0] == 20
    assert b.sum() == 120


def test_floor_is_honoured_when_the_budget_allows():
    w = weights_from_scores(np.array([9.0, 0.0, 0.0, 0.0]), temperature=0.1)
    b = allocate(100, [80, 80, 80, 80], w, floor=10)
    assert (b >= 10).all()


def test_floor_shrinks_when_budget_is_tiny():
    b = allocate(10, [50] * 4, np.full(4, 0.25), floor=8)
    assert b.sum() == 10
    assert b.max() - b.min() <= 1


def test_higher_score_never_gets_less_when_lengths_match():
    s = np.array([1.5, 0.7, 0.2, -0.3, -1.1])
    b = allocate(90, [100] * 5, weights_from_scores(s), floor=5)
    assert (np.diff(b) <= 0).all()


def test_weights_with_zeros_fall_back_to_uniform_over_whoever_has_room():
    # oracle-style weights: only chunk 0 is "gold" and it is small
    b = allocate(60, [10, 50, 50], np.array([1.0, 0.0, 0.0]), floor=0)
    assert b[0] == 10
    assert b.sum() == 60
    assert b[1] == b[2] == 25
