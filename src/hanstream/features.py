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
