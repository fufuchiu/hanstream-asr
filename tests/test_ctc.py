import numpy as np
import pytest

from hanstream import ctc as m


def test_collapse_empty():
    assert m.collapse([], 0) == ()
