"""CTC path collapse, prefix beam search and forced token alignment."""

import math
import numbers
from dataclasses import dataclass

import numpy as np

from .validation import positive_int


def logadd(*values: float) -> float:
    """Stable log(sum(exp(values))) including the empty sum."""
    if not values:
        return -math.inf
    top = max(values)
    if top == -math.inf:
        return top
    return top + math.log(sum(math.exp(x - top) for x in values))


def checked_log_probs(values, blank: int = 0) -> np.ndarray:
    """Accept normalized log probabilities with impossible (-inf) outcomes."""
    x = np.asarray(values, dtype=float)
    if x.ndim != 2 or x.shape[1] < 1 or np.isnan(x).any() or np.isposinf(x).any():
        raise ValueError('expected a time-by-vocabulary log-probability matrix')
    if (
        isinstance(blank, bool)
        or not isinstance(blank, numbers.Integral)
        or not 0 <= blank < x.shape[1]
    ):
        raise ValueError('blank ID outside vocabulary')
    if any(not math.isclose(logadd(*row), 0, abs_tol=1e-5) for row in x):
        raise ValueError('each log-probability row must sum to one')
    return x


def collapse(path, blank: int = 0) -> tuple[int, ...]:
    """Merge consecutive repetitions before removing blanks."""
    if isinstance(blank, bool) or not isinstance(blank, numbers.Integral) or blank < 0:
        raise ValueError('blank must be a nonnegative integer')
    previous, output = None, []
    for token in path:
        if isinstance(token, bool) or not isinstance(token, numbers.Integral) or token < 0:
            raise ValueError('path IDs must be nonnegative integers')
        if token != blank and token != previous:
            output.append(int(token))
        previous = token
    return tuple(output)


def greedy(log_probs, blank: int = 0) -> tuple[int, ...]:
    """Choose the best frame path and perform CTC collapse."""
    x = checked_log_probs(log_probs, blank)
    return collapse(x.argmax(axis=1), blank)


def prefix_beam_search(
    log_probs, beam_size: int = 8, blank: int = 0
) -> list[tuple[tuple[int, ...], float]]:
    """Aggregate blank/nonblank prefix probabilities, pruning after each frame."""
    x = checked_log_probs(log_probs, blank)
    beam_size = positive_int(beam_size)
    beam = {(): (0.0, -math.inf)}
    for row in x:
        next_beam = {}

        def update(prefix, p_blank=-math.inf, p_nonblank=-math.inf):
            before = next_beam.get(prefix, (-math.inf, -math.inf))
            next_beam[prefix] = (logadd(before[0], p_blank), logadd(before[1], p_nonblank))

        for prefix, (pb, pnb) in beam.items():
            total = logadd(pb, pnb)
            update(prefix, p_blank=total + row[blank])
            for token, probability in enumerate(row):
                if token == blank or probability == -math.inf:
                    continue
                if prefix and token == prefix[-1]:
                    update(prefix, p_nonblank=pnb + probability)
                    update(prefix + (token,), p_nonblank=pb + probability)
                else:
                    update(prefix + (token,), p_nonblank=total + probability)
        beam = dict(
            sorted(next_beam.items(), key=lambda item: (-logadd(*item[1]), item[0]))[:beam_size]
        )
    return sorted(((p, logadd(*v)) for p, v in beam.items()), key=lambda item: (-item[1], item[0]))


@dataclass(frozen=True)
class TokenSpan:
    token: int
    start_frame: int
    end_frame: int
    log_score: float


def forced_align(log_probs, target, blank: int = 0) -> list[TokenSpan]:
    """Viterbi-align a known target; reject impossible alignments explicitly."""
    x = checked_log_probs(log_probs, blank)
    target = list(target)
    if any(
        isinstance(t, bool)
        or not isinstance(t, numbers.Integral)
        or not 0 <= t < x.shape[1]
        or t == blank
        for t in target
    ):
        raise ValueError('target IDs must be in vocabulary and exclude blank')
    if not target:
        if len(x) and not np.isfinite(x[:, blank]).all():
            raise ValueError('empty target is impossible under these emissions')
        return []
    required = len(target) + sum(a == c for a, c in zip(target, target[1:]))
    if len(x) < required:
        raise ValueError('not enough frames for target and repeated-token blanks')
    labels = [blank]
    for token in target:
        labels.extend((token, blank))
    size = len(labels)
    dp = np.full((len(x), size), -np.inf)
    back = np.full((len(x), size), -1, dtype=int)
    dp[0, :2] = x[0, labels[:2]]
    for t in range(1, len(x)):
        for s, label in enumerate(labels):
            previous = [s]
            if s:
                previous.append(s - 1)
            if s > 1 and label != blank and label != labels[s - 2]:
                previous.append(s - 2)
            p = max(previous, key=lambda index: dp[t - 1, index])
            dp[t, s], back[t, s] = dp[t - 1, p] + x[t, label], p
    state = max((size - 1, size - 2), key=lambda s: dp[-1, s])
    if not np.isfinite(dp[-1, state]):
        raise ValueError('target has zero alignment probability')
    states = [state]
    for t in range(len(x) - 1, 0, -1):
        state = int(back[t, state])
        states.append(state)
    states.reverse()
    spans = []
    for i, token in enumerate(target):
        frames = [t for t, s in enumerate(states) if s == 2 * i + 1]
        spans.append(
            TokenSpan(int(token), min(frames), max(frames) + 1, float(x[frames, token].sum()))
        )
    return spans
