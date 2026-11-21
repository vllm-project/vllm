# SPDX-License-Identifier: Apache-2.0
"""The ratio plot's only real logic is the per-step arithmetic: the raw ratio
is inflated by the sweep's non-uniform steps, and dividing by the concurrency
ratio is what makes the two panels comparable."""

import pytest
from scripts.plot_moe_vs_dense_ratio import steps


def test_raw_ratio_tracks_step_size_and_normalized_ratio_removes_it():
    doubling, plus_eight = steps([(1, 100.0), (2, 180.0), (10, 360.0)])

    assert doubling == ("1→2", pytest.approx(1.8), pytest.approx(0.9))
    assert plus_eight == ("2→10", pytest.approx(2.0), pytest.approx(0.4))


def test_perfect_scaling_normalizes_to_one_regardless_of_step():
    for label, ratio, normalized in steps([(4, 100.0), (8, 200.0), (16, 400.0)]):
        assert ratio == pytest.approx(2.0), label
        assert normalized == pytest.approx(1.0), label


def test_regression_is_reported_as_a_ratio_below_one():
    (_, ratio, normalized), = steps([(64, 100.0), (72, 90.0)])

    assert ratio == pytest.approx(0.9)
    assert normalized == pytest.approx(0.8)


@pytest.mark.parametrize("series", [[], [(1, 100.0)]])
def test_too_few_points_has_no_steps(series):
    assert steps(series) == []
