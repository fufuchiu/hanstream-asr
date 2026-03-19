import pytest

from hanstream import metrics as m


def test_edit_empty():
    r = m.edit_counts('', '')
    assert (r.substitutions, r.deletions, r.insertions, r.hits) == (0, 0, 0, 0)
    assert r.errors == 0
    assert r.reference_length == 0
