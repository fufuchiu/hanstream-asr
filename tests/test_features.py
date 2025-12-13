import numpy as np
import pytest

from hanstream import features as m


def test_mel_zero():
    assert m.hz_to_mel(0) == 0


def test_hertz_zero():
    assert m.mel_to_hz(0) == 0


def test_mel_1000():
    assert float(m.hz_to_mel(1000)) == pytest.approx(999.9855371396244)
