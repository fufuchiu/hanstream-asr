"""Strict corpus manifests and deterministic speaker-disjoint splitting."""

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from .validation import finite, positive_int, probability


@dataclass(frozen=True)
class Utterance:
    id: str
    audio: str
    text: str
    speaker: str
    duration: float
    sample_rate: int = 16000

    def __post_init__(self):
        for key in ('id', 'audio', 'speaker'):
            value = getattr(self, key)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f'{key} must be a nonempty string')
        if not isinstance(self.text, str):
            raise ValueError('text must be a string')
        if finite(self.duration, 'duration') <= 0:
            raise ValueError('duration must be positive')
        positive_int(self.sample_rate, 'sample_rate')


def parse_record(data: dict) -> Utterance:
    """Reject unknown fields and missing required fields."""
    if not isinstance(data, dict):
        raise ValueError('manifest record must be an object')
    try:
        return Utterance(**data)
    except TypeError as exc:
        raise ValueError(f'invalid manifest fields: {exc}') from exc
