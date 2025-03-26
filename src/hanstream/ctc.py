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
