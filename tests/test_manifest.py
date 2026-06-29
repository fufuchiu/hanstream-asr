import pytest

from hanstream import manifest as m


def test_empty_id():
    with pytest.raises(ValueError):
        m.Utterance('', 'a.wav', '', 's', 1)


def test_empty_audio():
    with pytest.raises(ValueError):
        m.Utterance('u', '', '', 's', 1)


def test_empty_speaker():
    with pytest.raises(ValueError):
        m.Utterance('u', 'a.wav', '', '', 1)


def test_zero_duration():
    with pytest.raises(ValueError):
        m.Utterance('u', 'a.wav', '', 's', 0)


def test_nan_duration():
    with pytest.raises(ValueError):
        m.Utterance('u', 'a.wav', '', 's', float('nan'))


def test_zero_rate():
    with pytest.raises(ValueError):
        m.Utterance('u', 'a.wav', '', 's', 1, 0)


def test_nonstring_text():
    with pytest.raises(ValueError):
        m.Utterance('u', 'a.wav', 1, 's', 1)


def test_unknown_fields():
    with pytest.raises(ValueError):
        m.parse_record({'unexpected': 1})


def test_nonobject():
    with pytest.raises(ValueError):
        m.parse_record([])


def test_negative_fraction():
    with pytest.raises(ValueError):
        m.speaker_split([], -0.1)


def test_large_fraction():
    with pytest.raises(ValueError):
        m.speaker_split([], 1.1)


def test_zero_batch():
    with pytest.raises(ValueError):
        m.duration_batches([], 0)


def test_oversize_batch():
    with pytest.raises(ValueError):
        m.duration_batches([m.Utterance('u', 'u.wav', '你好', 's', 1.0)], 0.5)


def test_empty_summary():
    assert m.corpus_summary([]) == {'utterances': 0, 'hours': 0, 'speakers': 0, 'sample_rates': []}
