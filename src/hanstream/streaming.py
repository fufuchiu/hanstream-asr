"""Bounded PCM framing, endpoint decisions and transcript stabilization."""

from dataclasses import dataclass

from .audio import decode_pcm16, rms
from .validation import positive_int, probability


class PCMFramer:
    """Accumulate arbitrary byte boundaries into fixed mono PCM16 frames."""

    def __init__(self, samples_per_frame: int = 320):
        self.frame_bytes = 2 * positive_int(samples_per_frame)
        self._pending = bytearray()
        self.closed = False

    def feed(self, data: bytes) -> list:
        if self.closed:
            raise ValueError('framer is closed')
        self._pending.extend(data)
        frames = []
        while len(self._pending) >= self.frame_bytes:
            frames.append(decode_pcm16(bytes(self._pending[: self.frame_bytes])))
            del self._pending[: self.frame_bytes]
        return frames

    def flush(self) -> list:
        if self.closed:
            raise ValueError('framer is closed')
        if len(self._pending) % 2:
            raise ValueError('truncated PCM16 sample')
        result = [decode_pcm16(bytes(self._pending))] if self._pending else []
        self._pending.clear()
        self.closed = True
        return result

    @property
    def pending_bytes(self) -> int:
        return len(self._pending)
