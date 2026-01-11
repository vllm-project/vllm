# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Callable
from pathlib import Path
from unittest.mock import patch

from vllm.benchmarks.sweep.param_sweep import ParameterSweepItem
from vllm.benchmarks.sweep.serve_sla import solve_sla
from vllm.benchmarks.sweep.server import ServerProcess
from vllm.benchmarks.sweep.sla_sweep import (
    SLACriterionBase,
    SLALessThan,
    SLALessThanOrEqualTo,
    SLASweepItem,
)


def _set_return_value(
    var2metric: Callable[[ParameterSweepItem], list[dict[str, float]]],
):
    """
    Create a patch for run_sla with a specific function
    indicating the relationship between the benchmark combination
    (which includes the SLA variable) and the SLA criterion.
    """

    def mock_run_sla(
        server: ServerProcess | None,
        bench_cmd: list[str],
        *,
        serve_comb: ParameterSweepItem,
        bench_comb: ParameterSweepItem,
        iter_path: Path,
        num_runs: int,
        dry_run: bool,
    ):
        return var2metric(bench_comb)

    return patch("vllm.benchmarks.sweep.serve_sla.run_sla", side_effect=mock_run_sla)


def _var2metric_linear():
    def wrapped(bench_comb):
        x = float(bench_comb["request_rate"])
        y = x

        return [{"request_throughput": y}]

    return wrapped


def _var2metric_concave(elbow_point: float):
    def wrapped(bench_comb):
        x = float(bench_comb["request_rate"])
        if x < elbow_point:
            y = 0.5 * (x - elbow_point) + elbow_point
        else:
            y = 1.5 * (x - elbow_point) + elbow_point

        return [{"request_throughput": y}]

    return wrapped


def _var2metric_convex(elbow_point: float):
    def wrapped(bench_comb):
        x = float(bench_comb["request_rate"])
        if x < elbow_point:
            y = 1.5 * (x - elbow_point) + elbow_point
        else:
            y = 0.5 * (x - elbow_point) + elbow_point

        return [{"request_throughput": y}]

    return wrapped


def _var2metric_quadratic(y_intercept: float):
    def wrapped(bench_comb):
        x = float(bench_comb["request_rate"])
        y = y_intercept + 0.1 * x**2

        return [{"request_throughput": y}]

    return wrapped


def _var2metric_sqrt(y_intercept: float):
    def wrapped(bench_comb):
        x = float(bench_comb["request_rate"])
        y = y_intercept + 10 * x**0.5

        return [{"request_throughput": y}]

    return wrapped


def _run_solve_sla(
    var2metric: Callable[[ParameterSweepItem], list[dict[str, float]]],
    criterion: SLACriterionBase,
    min_value: int = 1,
    max_value: int = 100,
):
    with _set_return_value(var2metric):
        result = solve_sla(
            server=None,
            bench_cmd=[],
            serve_comb=ParameterSweepItem(),
            bench_comb=ParameterSweepItem(),
            sla_comb=SLASweepItem({"request_throughput": criterion}),
            base_path=Path(""),
            num_runs=1,
            dry_run=False,
            sla_variable="request_rate",
            sla_min_value=min_value,
            sla_max_value=max_value,
        )
        assert result is not None

        return result


def test_solve_linear_sla_le():
    sla_data, history = _run_solve_sla(
        _var2metric_linear(),
        SLALessThanOrEqualTo(target=32),
    )

    assert history.get_max_passing() == 32

    assert {val: margin <= 0 for val, margin in history.items()} == {
        100: False,
        1: True,
        32: True,
        33: False,
    }


def test_solve_linear_sla_lt():
    sla_data, history = _run_solve_sla(
        _var2metric_linear(),
        SLALessThan(target=32),
    )

    assert history.get_max_passing() == 31

    assert {val: margin <= 0 for val, margin in history.items()} == {
        100: False,
        1: True,
        31: True,
        32: False,
    }


def test_solve_linear_sla_oob():
    sla_data, history = _run_solve_sla(
        _var2metric_linear(),
        SLALessThanOrEqualTo(target=32),
        min_value=64,
    )

    assert history.get_max_passing() == 64
    assert history.get_min_failing() == 64

    assert {val: margin <= 0 for val, margin in history.items()} == {
        100: False,
        64: False,
    }


def test_solve_concave_sla_le():
    sla_data, history = _run_solve_sla(
        _var2metric_concave(elbow_point=32),
        SLALessThanOrEqualTo(target=24),
    )

    assert history.get_max_passing() == 16

    assert {val: margin <= 0 for val, margin in history.items()} == {
        100: False,
        1: True,
        7: True,
        13: True,
        15: True,
        16: True,
        17: False,
    }


def test_solve_convex_sla_le():
    sla_data, history = _run_solve_sla(
        _var2metric_convex(elbow_point=32),
        SLALessThanOrEqualTo(target=24),
    )

    assert history.get_max_passing() == 26

    assert {val: margin <= 0 for val, margin in history.items()} == {
        100: False,
        1: True,
        48: False,
        30: False,
        24: True,
        26: True,
        27: False,
    }


def test_solve_quadratic_sla_le():
    sla_data, history = _run_solve_sla(
        _var2metric_quadratic(y_intercept=10),
        SLALessThanOrEqualTo(target=50),
    )

    assert history.get_max_passing() == 20

    assert {val: margin <= 0 for val, margin in history.items()} == {
        100: False,
        1: True,
        4: True,
        20: True,
        21: False,
    }


def test_solve_sqrt_sla_le():
    sla_data, history = _run_solve_sla(
        _var2metric_sqrt(y_intercept=10),
        SLALessThanOrEqualTo(target=100),
    )

    assert history.get_max_passing() == 81

    assert {val: margin <= 0 for val, margin in history.items()} == {
        100: False,
        1: True,
        89: False,
        81: True,
        82: False,
    }
