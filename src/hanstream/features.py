"""NumPy log-mel features with explicit framing and normalization."""

import numpy as np

from .audio import frame_signal
from .validation import finite, matrix, positive_int, waveform


def hz_to_mel(hz):
    """Convert nonnegative Hertz to the HTK mel scale."""
    x = np.asarray(hz, dtype=float)
    if not np.isfinite(x).all() or (x < 0).any():
        raise ValueError('frequencies must be finite and nonnegative')
    return 2595.0 * np.log10(1 + x / 700.0)


def mel_to_hz(mel):
    """Invert the HTK mel transform."""
    x = np.asarray(mel, dtype=float)
    if not np.isfinite(x).all() or (x < 0).any() or (x > 100000).any():
        raise ValueError('mel values must lie in [0, 100000]')
    return 700.0 * (10 ** (x / 2595.0) - 1)


def power_spectrum(frames, n_fft: int = 512) -> np.ndarray:
    """Hann-windowed one-sided power divided by FFT length."""
    x = matrix(frames)
    n_fft = positive_int(n_fft)
    if n_fft < x.shape[1]:
        raise ValueError('n_fft cannot truncate a frame')
    return np.abs(np.fft.rfft(x * np.hanning(x.shape[1]), n=n_fft, axis=1)) ** 2 / n_fft


def mel_filterbank(
    sample_rate: int = 16000,
    n_fft: int = 512,
    n_mels: int = 40,
    f_min: float = 0,
    f_max: float | None = None,
) -> np.ndarray:
    """Continuous triangular filters, shape (n_mels, n_fft//2+1)."""
    rate, n_fft, n_mels = positive_int(sample_rate), positive_int(n_fft), positive_int(n_mels)
    low = finite(f_min)
    high = rate / 2 if f_max is None else finite(f_max)
    if not 0 <= low < high <= rate / 2:
        raise ValueError('frequency bounds must satisfy 0 <= min < max <= Nyquist')
    edges = mel_to_hz(np.linspace(hz_to_mel(low), hz_to_mel(high), n_mels + 2))
    frequencies = np.fft.rfftfreq(n_fft, 1 / rate)
    rising = (frequencies[None, :] - edges[:-2, None]) / (edges[1:-1] - edges[:-2])[:, None]
    falling = (edges[2:, None] - frequencies[None, :]) / (edges[2:] - edges[1:-1])[:, None]
    filters = np.maximum(0, np.minimum(rising, falling))
    if (filters.sum(axis=1) == 0).any():
        raise ValueError('FFT resolution is too low for the requested mel filters')
    return filters


def log_mel(
    samples,
    sample_rate: int = 16000,
    frame_size: int = 400,
    hop_size: int = 160,
    n_fft: int = 512,
    n_mels: int = 40,
) -> np.ndarray:
    """Return natural-log mel energies, including padded final frames."""
    x = waveform(samples)
    bank = mel_filterbank(sample_rate, n_fft, n_mels)
    frames = frame_signal(x, frame_size, hop_size)
    if positive_int(n_fft) < positive_int(frame_size):
        raise ValueError('n_fft cannot truncate a frame')
    if not len(frames):
        return np.empty((0, n_mels))
    return np.log(np.maximum(power_spectrum(frames, n_fft) @ bank.T, 1e-10))


def cmvn(features, epsilon: float = 1e-8) -> np.ndarray:
    """Normalize each feature column over time; constant columns become zero."""
    x = matrix(features)
    epsilon = finite(epsilon)
    if epsilon <= 0:
        raise ValueError('epsilon must be positive')
    return (x - x.mean(axis=0)) / np.maximum(x.std(axis=0), epsilon)


def delta(features, width: int = 2) -> np.ndarray:
    """Regression deltas with replicated boundary frames."""
    x = matrix(features)
    width = positive_int(width)
    padded = np.pad(x, ((width, width), (0, 0)), mode='edge')
    result = np.zeros_like(x)
    for i in range(1, width + 1):
        result += i * (
            padded[width + i : width + i + len(x)] - padded[width - i : width - i + len(x)]
        )
    return result / (2 * sum(i * i for i in range(1, width + 1)))
