# SPDX-License-Identifier: Apache-2.0
"""The matched-x plot's only real logic is the axis alignment: the expert
curve and the throughput sweep are measured on different grids, so a point
must land on its own concurrency's tick or the two figures cannot be read
side by side."""

import pytest
from scripts.plot_expert_activation import align, load, sweep_grid

GRID = [1, 2, 4, 8, 16, 24, 32]


def _pareto(path, concurrencies):
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = "\n".join(f"run,{c},1.0" for c in concurrencies)
    path.write_text(f"label,concurrency,output_token_throughput\n{rows}\n")


def test_curve_lands_on_its_own_concurrency_tick():
    xs, ys = align({1: 8.0, 4: 24.0, 32: 78.0}, GRID)

    assert xs == (GRID.index(1), GRID.index(4), GRID.index(32))
    assert ys == (8.0, 24.0, 78.0)


def test_unmeasured_grid_points_are_skipped_not_resampled():
    """A gap must shorten the line, not slide later points onto wrong ticks."""
    xs, _ = align({1: 8.0, 32: 78.0}, GRID)

    assert xs == (0, 6)


def test_curve_points_outside_the_grid_are_dropped():
    xs, ys = align({4: 24.0, 128: 105.0}, GRID)

    assert xs == (GRID.index(4),)
    assert ys == (24.0,)


@pytest.mark.parametrize("curve", [{}, {128: 105.0}])
def test_no_overlap_with_the_grid_draws_nothing(curve):
    assert align(curve, GRID) == ([], [])


def test_grid_is_the_sorted_union_of_every_sweep(tmp_path):
    _pareto(tmp_path / "moe" / "pareto.csv", [1, 2, 4, 8])
    _pareto(tmp_path / "dense" / "pareto.csv", [1, 2, 4, 8, 16, 32])

    assert sweep_grid(("moe", "dense"), tmp_path) == [1, 2, 4, 8, 16, 32]


def test_grid_reads_concurrency_as_a_number_not_a_string(tmp_path):
    """aiperf writes floats; string sort would order 16 before 2."""
    _pareto(tmp_path / "sweep" / "pareto.csv", ["2.0", "16.0", "4.0"])

    assert sweep_grid(("sweep",), tmp_path) == [2, 4, 16]


def test_load_parses_json_keys_back_into_integer_batch_sizes(tmp_path):
    (tmp_path / "m.json").write_text(
        '{"num_experts": 64, "experts_per_layer": {"1": 8.0, "16": 46.25}}'
    )

    assert load("m", tmp_path) == ({1: 8.0, 16: 46.25}, 64)
