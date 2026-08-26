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


def test_common_empty():
    assert m.common_prefix([]) == ()


def test_common_prefix():
    assert m.common_prefix(['abc', 'abd', 'ab']) == ('a', 'b')


def test_common_none():
    assert m.common_prefix(['abc', 'xbc']) == ()


def test_framing_arbitrary_bytes():
    from hanstream.audio import encode_pcm16

    f = m.PCMFramer(2)
    raw = encode_pcm16([0, 0.5, -1])
    assert f.feed(raw[:1]) == []
    frames = f.feed(raw[1:5])
    assert frames[0].tolist() == [0, 0.5]
    assert f.pending_bytes == 1
    assert f.feed(raw[5:]) == []
    assert f.flush()[0].tolist() == [-1]


def test_framing_odd_flush():
    f = m.PCMFramer(2)
    f.feed(b'x')
    with pytest.raises(ValueError):
        f.flush()
    assert not f.closed


def test_framing_closed_feed():
    f = m.PCMFramer(2)
    f.flush()
    with pytest.raises(ValueError):
        f.feed(b'')


def test_framing_closed_flush():
    f = m.PCMFramer(2)
    f.flush()
    with pytest.raises(ValueError):
        f.flush()


def test_endpoint_onset_offset():
    d = m.EndpointDetector(start_frames=2, silence_frames=2)
    events = []
    for x in [[0], [1], [1], [0], [0]]:
        events.extend(d.feed(x))
    assert [(e.kind, e.frame) for e in events] == [('start', 1), ('end', 3)]


def test_endpoint_reset():
    d = m.EndpointDetector(start_frames=1)
    d.feed([1])
    d.reset()
    assert not d.active
    assert d.frame == 0


def test_endpoint_ignores_short_noise():
    d = m.EndpointDetector(start_frames=2)
    assert d.feed([1]) == []
    assert d.feed([0]) == []
    assert not d.active
