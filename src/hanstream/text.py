"""Explicit Unicode text policies and deterministic character vocabularies."""

import unicodedata
from dataclasses import dataclass


@dataclass(frozen=True)
class TextPolicy:
    lowercase: bool = True
    punctuation: bool = False
    compatibility: bool = True


def normalize(text: str, policy: TextPolicy = TextPolicy()) -> str:
    """Normalize Unicode and whitespace; punctuation becomes a word boundary."""
    if not isinstance(text, str):
        raise ValueError('text must be a string')
    value = unicodedata.normalize('NFKC' if policy.compatibility else 'NFC', text)
    if policy.lowercase:
        value = value.lower()
    if not policy.punctuation:
        value = ''.join(' ' if unicodedata.category(c).startswith('P') else c for c in value)
    return ' '.join(value.split())


def tokens(text: str, unit: str = 'word', policy: TextPolicy = TextPolicy()) -> list[str]:
    """Tokenize words or non-whitespace Unicode code points."""
    text = normalize(text, policy)
    if unit == 'word':
        return text.split()
    if unit == 'char':
        return [c for c in text if not c.isspace()]
    raise ValueError('unit must be word or char')


def make_vocabulary(texts: list[str]) -> tuple[str, ...]:
    """Build a sorted vocabulary with blank=0 and unknown=1."""
    alphabet = set(''.join(normalize(text) for text in texts))
    return ('<blank>', '<unk>', *sorted(alphabet))


def check_vocabulary(vocabulary) -> tuple[str, ...]:
    """Validate the blank/unknown contract and unique single-character tokens."""
    value = tuple(vocabulary)
    if len(value) < 2 or value[:2] != ('<blank>', '<unk>'):
        raise ValueError('vocabulary must begin with blank and unknown')
    if any(not isinstance(v, str) or len(v) != 1 for v in value[2:]) or len(set(value)) != len(
        value
    ):
        raise ValueError('vocabulary requires unique character tokens')
    return value


def encode(text: str, vocabulary) -> list[int]:
    """Encode normalized characters; unseen characters become unknown."""
    vocab = check_vocabulary(vocabulary)
    lookup = {token: i for i, token in enumerate(vocab)}
    return [lookup.get(char, 1) for char in normalize(text)]
