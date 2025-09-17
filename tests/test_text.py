import pytest

from hanstream import text as m


def test_normalize_ascii_case():
    assert m.normalize('Hello WORLD') == 'hello world'


def test_normalize_spaces():
    assert m.normalize(' a  b ') == 'a b'
