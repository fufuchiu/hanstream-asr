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
