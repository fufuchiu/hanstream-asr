import numpy as np
import pytest

from hanstream import audio as m


def test_rms_empty():
    assert m.rms([]) == 0


def test_rms_unit():
    assert m.rms([-1, 1]) == 1


def test_peak_empty():
    assert m.peak([]) == 0


def test_peak_signed():
    assert m.peak([-3, 2]) == 3


def test_duration():
    assert m.duration([0] * 160, 16000) == 0.01


def test_pcm_empty():
    assert m.decode_pcm16(b'').tolist() == []


def test_pcm_endpoints():
    assert m.decode_pcm16(bytes.fromhex('00800000ff7f')).tolist() == [-1.0, 0.0, 32767 / 32768]


def test_pcm_saturation():
    assert m.encode_pcm16([-2, 0, 2]).hex() == '00800000ff7f'


def test_trim_silence():
    assert m.trim_silence([0, 0.5, -0.2, 0]).tolist() == [0.5, -0.2]


def test_trim_all():
    assert m.trim_silence([0, 0]).tolist() == []


def test_trim_threshold():
    assert m.trim_silence([0.01, 0.1, 0.01], 0.01).tolist() == [0.1]


def test_normalize_zero():
    assert m.normalize_peak([0, 0]).tolist() == [0, 0]


def test_normalize_peak():
    assert m.normalize_peak([-2, 1], 1).tolist() == [-1, 0.5]


def test_dc_zero_mean():
    assert m.remove_dc([1, 2, 3]).tolist() == [-1, 0, 1]


def test_dc_empty():
    assert m.remove_dc([]).tolist() == []


def test_preemphasis_empty():
    assert m.preemphasis([]).tolist() == []


def test_preemphasis_initial():
    assert m.preemphasis([2]).tolist() == [2]


def test_preemphasis_difference():
    assert m.preemphasis([1, 2, 4], 1).tolist() == [1, 1, 2]


def test_frame_overlap():
    assert m.frame_signal([1, 2, 3], 2, 1).tolist() == [[1, 2], [2, 3], [3, 0]]


def test_frame_no_pad():
    assert m.frame_signal([1, 2, 3], 2, 1, False).tolist() == [[1, 2], [2, 3]]


def test_frame_empty():
    assert m.frame_signal([], 4, 2).shape == (0, 4)


def test_frame_short_no_pad():
    assert m.frame_signal([1], 4, 2, False).shape == (0, 4)


def test_frame_gap():
    assert m.frame_signal([1, 2, 3, 4, 5], 2, 3).tolist() == [[1, 2], [4, 5]]


def test_odd_pcm():
    with pytest.raises(ValueError):
        m.decode_pcm16(b'x')


def test_invalid_rate():
    with pytest.raises(ValueError):
        m.duration([], 0)


def test_invalid_peak():
    with pytest.raises(ValueError):
        m.normalize_peak([1], 1.1)


def test_invalid_emphasis():
    with pytest.raises(ValueError):
        m.preemphasis([1], -1)


def test_zero_frame():
    with pytest.raises(ValueError):
        m.frame_signal([], 0, 1)


def test_zero_hop():
    with pytest.raises(ValueError):
        m.frame_signal([], 1, 0)


def test_noise_snr():
    with pytest.raises(ValueError):
        m.add_noise([1], float('inf'))


def test_excess_noise():
    with pytest.raises(ValueError):
        m.add_noise([1], 121)


def test_negative_threshold():
    with pytest.raises(ValueError):
        m.trim_silence([1], -1)


def test_noise_exact_snr():
    x = np.ones(100)
    y = m.add_noise(x, 20, seed=7)
    assert m.rms(y - x) == pytest.approx(0.1)


def test_noise_deterministic():
    assert np.array_equal(m.add_noise([1] * 20, 5, 8), m.add_noise([1] * 20, 5, 8))


def test_noise_silence():
    assert m.add_noise([0, 0], 0).tolist() == [0, 0]


def test_wav_roundtrip():
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / 'a.wav'
        m.write_wav(p, [-1, 0, 0.5], 8000)
        x, rate = m.read_wav(p)
    assert rate == 8000
    assert x.tolist() == [-1, 0, 0.5]
