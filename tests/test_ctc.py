import numpy as np
import pytest

from hanstream import ctc as m


def test_collapse_empty():
    assert m.collapse([], 0) == ()


def test_collapse_blank():
    assert m.collapse([0, 0], 0) == ()


def test_collapse_repeat():
    assert m.collapse([1, 1, 1], 0) == (1,)


def test_collapse_separated():
    assert m.collapse([1, 0, 1], 0) == (1, 1)


def test_collapse_mixed():
    assert m.collapse([0, 1, 1, 2, 0, 2], 0) == (1, 2, 2)


def test_collapse_nonzero_blank():
    assert m.collapse([2, 0, 0, 2, 1], 2) == (0, 1)


def test_bad_path_0():
    with pytest.raises(ValueError):
        m.collapse([-1])


def test_bad_path_1():
    with pytest.raises(ValueError):
        m.collapse([True])


def test_bad_path_2():
    with pytest.raises(ValueError):
        m.collapse([1.2])


def test_bad_blank():
    with pytest.raises(ValueError):
        m.greedy(np.log([[0.5, 0.5]]), 2)


def test_unnormalized():
    with pytest.raises(ValueError):
        m.greedy([[0.0, 0.0]])
