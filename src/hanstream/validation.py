"""Shared strict numeric and array contracts."""

import math
import numbers

import numpy as np


def positive_int(value: int, name: str = 'value') -> int:
    """Require an integer greater than zero, excluding booleans."""
    if isinstance(value, bool) or not isinstance(value, numbers.Integral) or value <= 0:
        raise ValueError(f'{name} must be a positive integer')
    return int(value)


def finite(value: float, name: str = 'value') -> float:
    """Require a finite real scalar."""
    if isinstance(value, bool) or not isinstance(value, numbers.Real) or not math.isfinite(value):
        raise ValueError(f'{name} must be a finite number')
    return float(value)


def probability(value: float, name: str = 'value') -> float:
    """Require a probability in the closed unit interval."""
    value = finite(value, name)
    if not 0 <= value <= 1:
        raise ValueError(f'{name} must be between zero and one')
    return value
