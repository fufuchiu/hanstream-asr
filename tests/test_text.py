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


def test_normalize_korean():
    assert m.normalize('음성 인식') == '음성 인식'


def test_normalize_arabic():
    assert m.normalize('مرحبا، عالم') == 'مرحبا عالم'


def test_normalize_cyrillic():
    assert m.normalize('ПРИВЕТ') == 'привет'


def test_normalize_greek():
    assert m.normalize('ΑΛΦΑ') == 'αλφα'


def test_normalize_nbsp():
    assert m.normalize('a\xa0b') == 'a b'


def test_normalize_ideographic_space():
    assert m.normalize('a\u3000b') == 'a b'


def test_normalize_ligature():
    assert m.normalize('ﬁle') == 'file'


def test_normalize_circled():
    assert m.normalize('①②') == '12'


def test_normalize_math_symbol():
    assert m.normalize('x+y=2') == 'x+y=2'


def test_normalize_only_punctuation():
    assert m.normalize('！？…') == ''
