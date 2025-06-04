import numpy as np

from hanstream.audio import encode_pcm16
from hanstream.streaming import PCMFramer

raw = encode_pcm16(np.sin(np.arange(1000) * 0.1) * 0.2)
framer = PCMFramer(320)
frames = []
for start in range(0, len(raw), 137):
    frames.extend(framer.feed(raw[start : start + 137]))
frames.extend(framer.flush())
assert sum(len(frame) for frame in frames) == 1000
print('frame sizes:', [len(frame) for frame in frames])
