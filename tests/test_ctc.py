import numpy as np
import pytest

from hanstream import ctc as m


def test_collapse_empty():
    assert m.collapse([], 0) == ()


def test_collapse_blank():
    assert m.collapse([0, 0], 0) == ()


def test_collapse_repeat():
    assert m.collapse([1, 1, 1], 0) == (1,)


def test_collapse_separated():
    assert m.collapse([1, 0, 1], 0) == (1, 1)


def test_collapse_mixed():
    assert m.collapse([0, 1, 1, 2, 0, 2], 0) == (1, 2, 2)


def test_collapse_nonzero_blank():
    assert m.collapse([2, 0, 0, 2, 1], 2) == (0, 1)


def test_bad_path_0():
    with pytest.raises(ValueError):
        m.collapse([-1])


def test_bad_path_1():
    with pytest.raises(ValueError):
        m.collapse([True])


def test_bad_path_2():
    with pytest.raises(ValueError):
        m.collapse([1.2])


def test_bad_blank():
    with pytest.raises(ValueError):
        m.greedy(np.log([[0.5, 0.5]]), 2)


def test_unnormalized():
    with pytest.raises(ValueError):
        m.greedy([[0.0, 0.0]])


def test_nan():
    with pytest.raises(ValueError):
        m.greedy([[float('nan'), 0]])


def test_positive_inf():
    with pytest.raises(ValueError):
        m.greedy([[float('inf'), 0]])


def test_all_impossible():
    with pytest.raises(ValueError):
        m.greedy([[-float('inf'), -float('inf')]])


def test_zero_beam():
    with pytest.raises(ValueError):
        m.prefix_beam_search(np.log([[0.5, 0.5]]), 0)


def test_impossible_repeat():
    with pytest.raises(ValueError):
        m.forced_align(np.log([[0.5, 0.5]]), [1, 1])


def test_blank_target():
    with pytest.raises(ValueError):
        m.forced_align(np.log([[0.5, 0.5]]), [0])


def test_out_of_range_target():
    with pytest.raises(ValueError):
        m.forced_align(np.log([[0.5, 0.5]]), [2])


def test_logadd_empty():
    assert m.logadd() == -float('inf')


def test_logadd_stable():
    assert m.logadd(-1000, -1000) == pytest.approx(-1000 + np.log(2))


def test_greedy_blank_separated():
    assert m.greedy(np.log([[0.1, 0.9], [0.9, 0.1], [0.1, 0.9]])) == (1, 1)


def test_beam_matches_exhaustive_path_sum():
    import itertools
    from collections import defaultdict

    p = np.array([[0.3, 0.7], [0.6, 0.4], [0.2, 0.8], [0.55, 0.45]])
    expected = defaultdict(float)
    for path in itertools.product(range(2), repeat=4):
        expected[m.collapse(path)] += np.prod([p[t, c] for t, c in enumerate(path)])
    actual = {key: np.exp(value) for key, value in m.prefix_beam_search(np.log(p), 64)}
    for key, value in expected.items():
        assert actual[key] == pytest.approx(value)
    assert sum(actual.values()) == pytest.approx(1)


def test_beam_nonzero_blank():
    p = np.array([[0.8, 0.2], [0.1, 0.9], [0.8, 0.2]])
    result = m.prefix_beam_search(np.log(p), 64, blank=1)
    assert result[0][0] == (0, 0)


def test_alignment_repeat_spans():
    p = np.log([[0.1, 0.9], [0.9, 0.1], [0.1, 0.9]])
    spans = m.forced_align(p, [1, 1])
    assert [(s.token, s.start_frame, s.end_frame) for s in spans] == [(1, 0, 1), (1, 2, 3)]


def test_alignment_impossible_emission():
    with pytest.raises(ValueError):
        m.forced_align([[0, -np.inf]], [1])
