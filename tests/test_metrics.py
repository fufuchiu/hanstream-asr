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
