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


@dataclass(frozen=True)
class EndpointEvent:
    kind: str
    frame: int


class EndpointDetector:
    """Energy-based speech boundaries with minimum speech and trailing silence."""

    def __init__(self, threshold: float = 0.01, start_frames: int = 2, silence_frames: int = 5):
        self.threshold = probability(threshold)
        self.start_frames = positive_int(start_frames)
        self.silence_frames = positive_int(silence_frames)
        self.reset()

    def reset(self) -> None:
        self.frame = 0
        self.active = False
        self.voiced = 0
        self.silent = 0

    def feed(self, samples) -> list[EndpointEvent]:
        energy = rms(samples)
        self.frame += 1
        speech = energy > self.threshold
        self.voiced = self.voiced + 1 if speech else 0
        self.silent = 0 if speech else self.silent + 1
        if not self.active and self.voiced >= self.start_frames:
            self.active = True
            return [EndpointEvent('start', self.frame - self.start_frames)]
        if self.active and self.silent >= self.silence_frames:
            self.active = False
            return [EndpointEvent('end', self.frame - self.silence_frames)]
        return []


def common_prefix(sequences) -> tuple:
    """Find a prefix shared by all candidate sequences."""
    values = [tuple(v) for v in sequences]
    if not values:
        return ()
    output = []
    for column in zip(*values):
        if len(set(column)) != 1:
            break
        output.append(column[0])
    return tuple(output)
