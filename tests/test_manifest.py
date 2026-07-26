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


def test_all_training():
    assert len(m.speaker_split([m.Utterance('u', 'u.wav', '你好', 's', 1.0)], 0)[0]) == 1


def test_all_validation():
    assert len(m.speaker_split([m.Utterance('u', 'u.wav', '你好', 's', 1.0)], 1)[1]) == 1


def test_speaker_disjoint_and_complete():
    rows = [m.Utterance(str(i), str(i) + '.wav', '', str(i // 3), 1) for i in range(60)]
    a, c = m.speaker_split(rows, 0.3, 'seed')
    assert not ({r.speaker for r in a} & {r.speaker for r in c})
    assert len(a) + len(c) == 60
    assert a and c


def test_split_order_independence():
    rows = [m.Utterance(str(i), 'a.wav', '', str(i), 1) for i in range(30)]
    a, c = m.speaker_split(rows, 0.5)
    b, d = m.speaker_split(reversed(rows), 0.5)
    assert {r.id for r in a} == {r.id for r in b}
    assert {r.id for r in c} == {r.id for r in d}


def test_duration_budget():
    rows = [m.Utterance('u', 'u.wav', '你好', 's', 1.0) for _ in range(5)]
    assert [len(x) for x in m.duration_batches(rows, 2)] == [2, 2, 1]


def test_manifest_roundtrip():
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / 'data.jsonl'
        m.save_manifest(p, [m.Utterance('u', 'u.wav', '你好', 's', 1.0)])
        assert m.load_manifest(p) == [m.Utterance('u', 'u.wav', '你好', 's', 1.0)]


def test_load_rejects_duplicate():
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / 'data.jsonl'
        p.write_text(
            '{"id":"u","audio":"u.wav","text":"","speaker":"s","duration":1}\n{"id":"u","audio":"u.wav","text":"","speaker":"s","duration":1}\n'
        )
        with pytest.raises(ValueError, match='data.jsonl:'):
            m.load_manifest(p)


def test_load_rejects_invalid_json():
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / 'data.jsonl'
        p.write_text('{broken\n')
        with pytest.raises(ValueError, match='data.jsonl:'):
            m.load_manifest(p)


def test_load_rejects_unknown_field():
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / 'data.jsonl'
        p.write_text('{"extra":1}\n')
        with pytest.raises(ValueError, match='data.jsonl:'):
            m.load_manifest(p)


def test_audio_existence_checked():
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / 'data.jsonl'
        m.save_manifest(p, [m.Utterance('u', 'u.wav', '你好', 's', 1.0)])
        with pytest.raises(ValueError, match='audio file missing'):
            m.load_manifest(p, True)
