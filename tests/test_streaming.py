import pytest

from hanstream import streaming as m


def test_zero_frame():
    with pytest.raises(ValueError):
        m.PCMFramer(0)


def test_zero_start():
    with pytest.raises(ValueError):
        m.EndpointDetector(start_frames=0)


def test_zero_silence():
    with pytest.raises(ValueError):
        m.EndpointDetector(silence_frames=0)


def test_negative_threshold():
    with pytest.raises(ValueError):
        m.EndpointDetector(threshold=-1)


def test_zero_window():
    with pytest.raises(ValueError):
        m.StableTranscript(0)
