import pytest

from hanstream import manifest as m


def test_empty_id():
    with pytest.raises(ValueError):
        m.Utterance('', 'a.wav', '', 's', 1)


def test_empty_audio():
    with pytest.raises(ValueError):
        m.Utterance('u', '', '', 's', 1)
