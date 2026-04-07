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
