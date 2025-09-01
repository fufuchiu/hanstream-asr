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


def test_positive_int_rejects_3():
    with pytest.raises(ValueError):
        m.positive_int(1.5)


def test_positive_int_rejects_4():
    with pytest.raises(ValueError):
        m.positive_int('2')


def test_positive_int_rejects_5():
    with pytest.raises(ValueError):
        m.positive_int(None)


def test_finite_rejects_0():
    with pytest.raises(ValueError):
        m.finite(True)


def test_finite_rejects_1():
    with pytest.raises(ValueError):
        m.finite('1')


def test_finite_rejects_2():
    with pytest.raises(ValueError):
        m.finite(None)


def test_finite_rejects_3():
    with pytest.raises(ValueError):
        m.finite(float('nan'))


def test_finite_rejects_4():
    with pytest.raises(ValueError):
        m.finite(float('inf'))


def test_finite_rejects_5():
    with pytest.raises(ValueError):
        m.finite(-float('inf'))


def test_probability_rejects_0():
    with pytest.raises(ValueError):
        m.probability(-0.01)


def test_probability_rejects_1():
    with pytest.raises(ValueError):
        m.probability(1.01)


def test_probability_rejects_2():
    with pytest.raises(ValueError):
        m.probability(True)


def test_probability_rejects_3():
    with pytest.raises(ValueError):
        m.probability(float('nan'))


def test_probability_rejects_4():
    with pytest.raises(ValueError):
        m.probability(float('inf'))
