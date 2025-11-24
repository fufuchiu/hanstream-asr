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


def test_normalize_empty():
    assert m.normalize('') == ''


def test_normalize_newlines():
    assert m.normalize('\n\r\t') == ''


def test_normalize_digits():
    assert m.normalize('123.45') == '123 45'


def test_normalize_quotes():
    assert m.normalize('“语音”') == '语音'


def test_normalize_underscore():
    assert m.normalize('a_b') == 'a b'


def test_normalize_slash():
    assert m.normalize('a/b') == 'a b'


def test_normalize_german():
    assert m.normalize('Straße') == 'straße'


def test_normalize_hindi():
    assert m.normalize('नमस्ते') == 'नमस्ते'


def test_normalize_brackets():
    assert m.normalize('[one] (two)') == 'one two'


def test_preserve_case():
    assert m.normalize('ABC', m.TextPolicy(lowercase=False)) == 'ABC'


def test_preserve_punctuation():
    assert m.normalize('Hi!', m.TextPolicy(punctuation=True)) == 'hi!'


def test_disable_compatibility():
    assert m.normalize('Ａ', m.TextPolicy(compatibility=False, lowercase=False)) == 'Ａ'


def test_characters_remove_spaces():
    assert m.tokens('广 州 大学', 'char') == ['广', '州', '大', '学']


def test_word_tokens():
    assert m.tokens('A, B') == ['a', 'b']


def test_invalid_unit():
    with pytest.raises(ValueError):
        m.tokens('a', 'syllable')


def test_nonstring():
    with pytest.raises(ValueError):
        m.normalize(12)


def test_vocabulary_order():
    assert m.make_vocabulary(['ba', 'a']) == ('<blank>', '<unk>', 'a', 'b')


def test_unknown_encoding():
    assert m.encode('az', ('<blank>', '<unk>', 'a')) == [2, 1]
