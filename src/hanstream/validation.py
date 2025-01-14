"""Shared strict numeric and array contracts."""

import math
import numbers

import numpy as np


def positive_int(value: int, name: str = 'value') -> int:
    """Require an integer greater than zero, excluding booleans."""
    if isinstance(value, bool) or not isinstance(value, numbers.Integral) or value <= 0:
        raise ValueError(f'{name} must be a positive integer')
    return int(value)
