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
