import numpy as np
import pytest

from hanstream import validation as m


def test_positive_int_rejects_0():
    with pytest.raises(ValueError):
        m.positive_int(0)


def test_positive_int_rejects_1():
    with pytest.raises(ValueError):
        m.positive_int(-1)


def test_positive_int_rejects_2():
    with pytest.raises(ValueError):
        m.positive_int(True)
