import math

import pytest

from nv_fix.intervals import normalize_intervals, target_duration_seconds


def test_normalize_sorts_and_merges_intervals():
    assert normalize_intervals([(2.0, 3.0), (0.2, 1.0), (0.9, 2.4), (4.0, 5.0)]) == [
        (0.2, 3.0), (4.0, 5.0)
    ]


def test_normalize_accepts_one_pair():
    assert normalize_intervals((1, 2.5)) == [(1.0, 2.5)]


@pytest.mark.parametrize("value", [[(1, 1)], [(-0.1, 1)], [(0, math.inf)], "0.1:0.2"])
def test_normalize_rejects_invalid_intervals(value):
    with pytest.raises((TypeError, ValueError)):
        normalize_intervals(value)


def test_target_duration_preserves_token_frame():
    assert int(target_duration_seconds(24_001, 24_000, 50) * 50) == 51
