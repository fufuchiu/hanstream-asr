"""PCM I/O and deterministic waveform transforms."""

import wave
from pathlib import Path

import numpy as np

from .validation import finite, positive_int, probability, waveform


def decode_pcm16(data: bytes) -> np.ndarray:
    """Decode mono little-endian signed PCM16 into [-1, 1)."""
    if len(data) % 2:
        raise ValueError('PCM16 byte length must be even')
    return np.frombuffer(data, dtype='<i2').astype(np.float64) / 32768.0
