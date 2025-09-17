import pytest

from hanstream import text as m


def test_normalize_ascii_case():
    assert m.normalize('Hello WORLD') == 'hello world'


def test_normalize_spaces():
    assert m.normalize(' a  b ') == 'a b'


def test_normalize_tabs():
    assert m.normalize('a\tb\nc') == 'a b c'


def test_normalize_chinese():
    assert m.normalize('广州大学，语音识别！') == '广州大学 语音识别'


def test_normalize_fullwidth():
    assert m.normalize('ＡＢＣ１２３') == 'abc123'


def test_normalize_accent():
    assert m.normalize('CAFÉ') == 'café'


def test_normalize_combining():
    assert m.normalize('café') == 'café'


def test_normalize_apostrophe():
    assert m.normalize("don't") == 'don t'


def test_normalize_hyphen():
    assert m.normalize('speech-to-text') == 'speech to text'


def test_normalize_emoji():
    assert m.normalize('你好🙂') == '你好🙂'


def test_normalize_japanese():
    assert m.normalize('音声、認識。') == '音声 認識'
