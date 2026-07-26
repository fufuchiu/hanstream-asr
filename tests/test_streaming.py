import pytest

from hanstream import streaming as m


def test_zero_frame():
    with pytest.raises(ValueError):
        m.PCMFramer(0)
