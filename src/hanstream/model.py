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


def ctc_loss(
    log_probs: torch.Tensor,
    targets: torch.Tensor,
    input_lengths: torch.Tensor,
    target_lengths: torch.Tensor,
    blank: int = 0,
) -> torch.Tensor:
    """Batch-first CTC loss with explicit rejection of impossible targets."""
    if log_probs.ndim != 3 or not torch.isfinite(log_probs).all():
        raise ValueError('log_probs must be a finite batch/time/vocabulary tensor')
    batch, time, vocab = log_probs.shape
    if isinstance(blank, bool) or not isinstance(blank, int) or not 0 <= blank < vocab:
        raise ValueError('invalid blank ID')
    if not torch.allclose(log_probs.logsumexp(-1), torch.zeros_like(log_probs[..., 0]), atol=1e-5):
        raise ValueError('log probabilities must be normalized')
    for lengths in (input_lengths, target_lengths):
        if lengths.dtype not in (torch.int32, torch.int64) or lengths.shape != (batch,):
            raise ValueError('invalid length tensor')
    if (input_lengths <= 0).any() or (input_lengths > time).any() or (target_lengths < 0).any():
        raise ValueError('invalid input or target lengths')
    if (
        targets.ndim != 1
        or targets.dtype not in (torch.int32, torch.int64)
        or targets.numel() != int(target_lengths.sum())
    ):
        raise ValueError('targets must be concatenated integer IDs')
    if (targets < 0).any() or (targets >= vocab).any() or (targets == blank).any():
        raise ValueError('targets must exclude blank and lie in vocabulary')
    offset = 0
    for available, length in zip(input_lengths.tolist(), target_lengths.tolist()):
        sequence = targets[offset : offset + length]
        required = length + int((sequence[1:] == sequence[:-1]).sum())
        if required > available:
            raise ValueError('target cannot be aligned to input frames')
        offset += length
    return nn.functional.ctc_loss(
        log_probs.transpose(0, 1),
        targets,
        input_lengths,
        target_lengths,
        blank=blank,
        zero_infinity=False,
    )


def train_step(
    model: TinyCTC,
    optimizer,
    features,
    lengths,
    targets,
    target_lengths,
    max_grad_norm: float = 1.0,
) -> float:
    """Perform one finite-loss, clipped-gradient update."""
    if not 0 < max_grad_norm < float('inf'):
        raise ValueError('max_grad_norm must be positive and finite')
    model.train()
    optimizer.zero_grad(set_to_none=True)
    loss = ctc_loss(model(features, lengths), targets, lengths, target_lengths, model.config.blank)
    if not torch.isfinite(loss):
        raise ValueError('nonfinite training loss')
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm, error_if_nonfinite=True)
    optimizer.step()
    return float(loss.detach())
