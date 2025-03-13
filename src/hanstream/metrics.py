"""Deterministic edit alignment and micro-averaged corpus scoring."""

from collections import Counter
from dataclasses import dataclass

from .text import TextPolicy, tokens


@dataclass(frozen=True)
class EditCounts:
    substitutions: int = 0
    deletions: int = 0
    insertions: int = 0
    hits: int = 0

    @property
    def errors(self) -> int:
        return self.substitutions + self.deletions + self.insertions

    @property
    def reference_length(self) -> int:
        return self.substitutions + self.deletions + self.hits

    @property
    def rate(self) -> float:
        """Empty-reference insertions count as their raw count (denominator one)."""
        return self.errors / max(1, self.reference_length)
