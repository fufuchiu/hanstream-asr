"""A small trainable recurrent CTC baseline; no pretrained weights are bundled."""

from dataclasses import dataclass

import torch
from torch import nn

from .validation import positive_int


@dataclass(frozen=True)
class CTCConfig:
    input_features: int = 40
    hidden_size: int = 64
    vocabulary_size: int = 32
    layers: int = 2
    blank: int = 0

    def __post_init__(self):
        for key in ('input_features', 'hidden_size', 'vocabulary_size', 'layers'):
            positive_int(getattr(self, key), key)
        if (
            isinstance(self.blank, bool)
            or not isinstance(self.blank, int)
            or not 0 <= self.blank < self.vocabulary_size
        ):
            raise ValueError('invalid blank ID')
