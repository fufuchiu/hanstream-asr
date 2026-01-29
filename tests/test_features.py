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


def test_nan_hz():
    with pytest.raises(ValueError):
        m.hz_to_mel(float('nan'))


def test_negative_mel():
    with pytest.raises(ValueError):
        m.mel_to_hz(-1)


def test_overflow_mel():
    with pytest.raises(ValueError):
        m.mel_to_hz(100001)


def test_fft_truncation():
    with pytest.raises(ValueError):
        m.power_spectrum([[1] * 8], 4)


def test_zero_fft():
    with pytest.raises(ValueError):
        m.power_spectrum([[1]], 0)


def test_negative_low():
    with pytest.raises(ValueError):
        m.mel_filterbank(f_min=-1)


def test_above_nyquist():
    with pytest.raises(ValueError):
        m.mel_filterbank(f_max=9000)


def test_reversed_bounds():
    with pytest.raises(ValueError):
        m.mel_filterbank(f_min=4000, f_max=2000)


def test_empty_filters():
    with pytest.raises(ValueError):
        m.mel_filterbank(n_fft=4, n_mels=40)


def test_zero_epsilon():
    with pytest.raises(ValueError):
        m.cmvn([[1]], 0)


def test_negative_delta():
    with pytest.raises(ValueError):
        m.delta([[1]], 0)


def test_negative_mask():
    with pytest.raises(ValueError):
        m.mask_features([[1]], -1)


def test_oversized_time_mask():
    with pytest.raises(ValueError):
        m.mask_features([[1]], 2)


def test_oversized_frequency_mask():
    with pytest.raises(ValueError):
        m.mask_features([[1]], 0, 2)


def test_boolean_mask():
    with pytest.raises(ValueError):
        m.mask_features([[1]], True)


def test_mel_inverse():
    x = np.array([0, 100, 1000, 4000, 8000])
    assert m.mel_to_hz(m.hz_to_mel(x)) == pytest.approx(x)
