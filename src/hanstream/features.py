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
