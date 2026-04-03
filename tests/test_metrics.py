import pytest

from hanstream import metrics as m


def test_edit_empty():
    r = m.edit_counts('', '')
    assert (r.substitutions, r.deletions, r.insertions, r.hits) == (0, 0, 0, 0)
    assert r.errors == 0
    assert r.reference_length == 0


def test_edit_insert():
    r = m.edit_counts('', 'ab')
    assert (r.substitutions, r.deletions, r.insertions, r.hits) == (0, 0, 2, 0)
    assert r.errors == 2
    assert r.reference_length == 0


def test_edit_delete():
    r = m.edit_counts('ab', '')
    assert (r.substitutions, r.deletions, r.insertions, r.hits) == (0, 2, 0, 0)
    assert r.errors == 2
    assert r.reference_length == 2


def test_edit_hit():
    r = m.edit_counts('abc', 'abc')
    assert (r.substitutions, r.deletions, r.insertions, r.hits) == (0, 0, 0, 3)
    assert r.errors == 0
    assert r.reference_length == 3


def test_edit_sub():
    r = m.edit_counts('abc', 'adc')
    assert (r.substitutions, r.deletions, r.insertions, r.hits) == (1, 0, 0, 2)
    assert r.errors == 1
    assert r.reference_length == 3


def test_edit_leading_insert():
    r = m.edit_counts('ab', 'xab')
    assert (r.substitutions, r.deletions, r.insertions, r.hits) == (0, 0, 1, 2)
    assert r.errors == 1
    assert r.reference_length == 2


def test_edit_trailing_insert():
    r = m.edit_counts('ab', 'abx')
    assert (r.substitutions, r.deletions, r.insertions, r.hits) == (0, 0, 1, 2)
    assert r.errors == 1
    assert r.reference_length == 2


def test_edit_middle_insert():
    r = m.edit_counts('ab', 'axb')
    assert (r.substitutions, r.deletions, r.insertions, r.hits) == (0, 0, 1, 2)
    assert r.errors == 1
    assert r.reference_length == 2


def test_edit_leading_delete():
    r = m.edit_counts('xab', 'ab')
    assert (r.substitutions, r.deletions, r.insertions, r.hits) == (0, 1, 0, 2)
    assert r.errors == 1
    assert r.reference_length == 3


def test_edit_trailing_delete():
    r = m.edit_counts('abx', 'ab')
    assert (r.substitutions, r.deletions, r.insertions, r.hits) == (0, 1, 0, 2)
    assert r.errors == 1
    assert r.reference_length == 3


def test_edit_middle_delete():
    r = m.edit_counts('axb', 'ab')
    assert (r.substitutions, r.deletions, r.insertions, r.hits) == (0, 1, 0, 2)
    assert r.errors == 1
    assert r.reference_length == 3


def test_edit_repeat_delete():
    r = m.edit_counts('aaa', 'aa')
    assert (r.substitutions, r.deletions, r.insertions, r.hits) == (0, 1, 0, 2)
    assert r.errors == 1
    assert r.reference_length == 3


def test_edit_repeat_insert():
    r = m.edit_counts('aa', 'aaa')
    assert (r.substitutions, r.deletions, r.insertions, r.hits) == (0, 0, 1, 2)
    assert r.errors == 1
    assert r.reference_length == 2


def test_edit_swap():
    r = m.edit_counts('ab', 'ba')
    assert (r.substitutions, r.deletions, r.insertions, r.hits) == (2, 0, 0, 0)
    assert r.errors == 2
    assert r.reference_length == 2


def test_edit_chinese():
    r = m.edit_counts('语音识别', '语音识字')
    assert (r.substitutions, r.deletions, r.insertions, r.hits) == (1, 0, 0, 3)
    assert r.errors == 1
    assert r.reference_length == 4


def test_edit_mixed():
    r = m.edit_counts('ASR语音', 'ASR语音')
    assert (r.substitutions, r.deletions, r.insertions, r.hits) == (0, 0, 0, 5)
    assert r.errors == 0
    assert r.reference_length == 5


def test_edit_all_replace():
    r = m.edit_counts('aaa', 'bbb')
    assert (r.substitutions, r.deletions, r.insertions, r.hits) == (3, 0, 0, 0)
    assert r.errors == 3
    assert r.reference_length == 3


def test_edit_one_to_many():
    r = m.edit_counts('a', 'abc')
    assert (r.substitutions, r.deletions, r.insertions, r.hits) == (0, 0, 2, 1)
    assert r.errors == 2
    assert r.reference_length == 1


def test_edit_many_to_one():
    r = m.edit_counts('abc', 'b')
    assert (r.substitutions, r.deletions, r.insertions, r.hits) == (0, 2, 0, 1)
    assert r.errors == 2
    assert r.reference_length == 3


def test_edit_emoji():
    r = m.edit_counts('🙂x', '🙂y')
    assert (r.substitutions, r.deletions, r.insertions, r.hits) == (1, 0, 0, 1)
    assert r.errors == 1
    assert r.reference_length == 2


def test_empty_reference_insert_rate():
    assert m.score('', 'a b').rate == 2


def test_cer_ignores_spaces():
    assert m.score('广 州', '广州', 'char').rate == 0


def test_micro_average():
    assert m.corpus_score(['a', 'a b c'], ['x', 'a b c']).rate == 0.25


def test_empty_corpus():
    assert m.corpus_score([], []).rate == 0


def test_sentence_rate():
    assert m.sentence_error_rate(['a', 'b'], ['a', 'c']) == 0.5


def test_confusion_counts():
    assert m.confusion_pairs('abab', 'acac') == {('b', 'c'): 2}


def test_corpus_length_mismatch():
    with pytest.raises(ValueError):
        m.corpus_score(['a'], [])
