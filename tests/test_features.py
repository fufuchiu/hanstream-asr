import numpy as np
import pytest

from hanstream import features as m


def test_mel_zero():
    assert m.hz_to_mel(0) == 0


def test_hertz_zero():
    assert m.mel_to_hz(0) == 0


def test_mel_1000():
    assert float(m.hz_to_mel(1000)) == pytest.approx(999.9855371396244)


def test_empty_features():
    assert m.log_mel([]).shape == (0, 40)


def test_silence_features():
    assert m.log_mel([0] * 160).shape == (1, 40)


def test_cmvn_constant():
    assert m.cmvn([[3, 4], [3, 4]]).tolist() == [[0, 0], [0, 0]]


def test_delta_constant():
    assert m.delta([[3], [3], [3]]).tolist() == [[0], [0], [0]]


def test_delta_ramp():
    assert m.delta([[0], [1], [2]], 1).ravel().tolist() == [0.5, 1, 0.5]


def test_mask_all_time():
    assert m.mask_features([[1, 2], [3, 4]], 2).tolist() == [[0, 0], [0, 0]]


def test_mask_all_frequency():
    assert m.mask_features([[1, 2], [3, 4]], 0, 2).tolist() == [[0, 0], [0, 0]]


def test_negative_hz():
    with pytest.raises(ValueError):
        m.hz_to_mel(-1)
