"""Explicit Unicode text policies and deterministic character vocabularies."""

import unicodedata
from dataclasses import dataclass


@dataclass(frozen=True)
class TextPolicy:
    lowercase: bool = True
    punctuation: bool = False
    compatibility: bool = True
