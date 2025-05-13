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


class TinyCTC(nn.Module):
    """Unidirectional GRU encoder whose packed batches exclude padded audio."""

    def __init__(self, config: CTCConfig = CTCConfig()):
        super().__init__()
        self.config = config
        self.encoder = nn.GRU(
            config.input_features, config.hidden_size, config.layers, batch_first=True
        )
        self.projection = nn.Linear(config.hidden_size, config.vocabulary_size)

    def forward(self, features: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        if (
            features.ndim != 3
            or features.shape[2] != self.config.input_features
            or not torch.isfinite(features).all()
        ):
            raise ValueError('invalid feature tensor')
        if lengths.dtype not in (torch.int32, torch.int64) or lengths.shape != (features.shape[0],):
            raise ValueError('invalid feature lengths')
        if (lengths <= 0).any() or (lengths > features.shape[1]).any():
            raise ValueError('feature lengths outside padded time dimension')
        packed = nn.utils.rnn.pack_padded_sequence(
            features, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        encoded, _ = self.encoder(packed)
        padded, _ = nn.utils.rnn.pad_packed_sequence(
            encoded, batch_first=True, total_length=features.shape[1]
        )
        return self.projection(padded).log_softmax(-1)
