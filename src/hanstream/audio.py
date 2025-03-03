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


def encode_pcm16(samples) -> bytes:
    """Saturate, round and encode normalized mono samples."""
    x = waveform(samples)
    return np.clip(np.rint(x * 32768), -32768, 32767).astype('<i2').tobytes()


def read_wav(path: str | Path) -> tuple[np.ndarray, int]:
    """Read uncompressed PCM16 WAV and average channels."""
    with wave.open(str(path), 'rb') as stream:
        if stream.getsampwidth() != 2 or stream.getcomptype() != 'NONE':
            raise ValueError('only uncompressed PCM16 WAV is supported')
        rate, channels, count = stream.getframerate(), stream.getnchannels(), stream.getnframes()
        raw = stream.readframes(count)
    if len(raw) != count * channels * 2:
        raise ValueError('truncated WAV payload')
    return decode_pcm16(raw).reshape(-1, channels).mean(axis=1), rate


def write_wav(path: str | Path, samples, sample_rate: int = 16000) -> None:
    """Write a mono PCM16 WAV file with a checked sample rate."""
    rate = positive_int(sample_rate, 'sample_rate')
    payload = encode_pcm16(samples)
    with wave.open(str(path), 'wb') as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(rate)
        stream.writeframes(payload)


def rms(samples) -> float:
    """Root mean square amplitude; empty input has zero energy."""
    x = waveform(samples)
    return float(np.sqrt(np.mean(x * x))) if len(x) else 0.0


def peak(samples) -> float:
    """Maximum absolute amplitude, or zero for empty input."""
    return float(np.max(np.abs(waveform(samples)), initial=0))


def normalize_peak(samples, target: float = 0.95) -> np.ndarray:
    """Scale to a peak target; preserve silence without dividing by zero."""
    target = probability(target, 'target')
    x = waveform(samples)
    maximum = peak(x)
    return x * (target / maximum) if maximum else x


def preemphasis(samples, coefficient: float = 0.97) -> np.ndarray:
    """Apply y[t] = x[t] - coefficient*x[t-1], preserving x[0]."""
    coefficient = probability(coefficient, 'coefficient')
    x = waveform(samples)
    return np.concatenate((x[:1], x[1:] - coefficient * x[:-1]))


def frame_signal(samples, frame_size: int, hop_size: int, pad: bool = True) -> np.ndarray:
    """Frame a waveform; padded framing emits every hop start below its end."""
    size, hop = positive_int(frame_size), positive_int(hop_size)
    x = waveform(samples)
    starts = range(0, len(x), hop) if pad else range(0, max(0, len(x) - size + 1), hop)
    rows = [np.pad(x[i : i + size], (0, max(0, i + size - len(x)))) for i in starts]
    return np.stack(rows) if rows else np.empty((0, size), dtype=np.float64)


def remove_dc(samples) -> np.ndarray:
    """Remove the mean without mutating caller-owned samples."""
    x = waveform(samples)
    return x - x.mean() if len(x) else x
