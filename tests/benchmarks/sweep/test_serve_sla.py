# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Callable
from pathlib import Path
from unittest.mock import patch

from vllm.benchmarks.sweep.param_sweep import ParameterSweepItem
from vllm.benchmarks.sweep.serve_sla import _estimate_sla_bounds, _find_sla_value
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


def _var2metric_identity(bench_comb):
    return [{"request_throughput": float(bench_comb["request_rate"])}]


def _run_estimate_sla_bounds(
    var2metric: Callable[[ParameterSweepItem], list[dict[str, float]]],
    criterion: SLACriterionBase,
    init_value: int,
    max_value: int,
):
    with _set_return_value(var2metric):
        return _estimate_sla_bounds(
            server=None,
            bench_cmd=[],
            serve_comb=ParameterSweepItem(),
            bench_comb=ParameterSweepItem(),
            sla_comb=SLASweepItem({"request_throughput": criterion}),
            base_path=Path(""),
            num_runs=1,
            dry_run=False,
            sla_variable="request_rate",
            init_value=init_value,
            max_value=max_value,
        )


def test_estimate_sla_bounds_le():
    sla_data, (max_passing, min_failing), history = _run_estimate_sla_bounds(
        _var2metric_identity,
        SLALessThanOrEqualTo(target=32),
        init_value=1,
        max_value=100,
    )

    assert max_passing == 32
    assert min_failing == 64

    assert {val: margin <= 0 for val, margin in history.items()} == {
        1: True,
        2: True,
        4: True,
        8: True,
        16: True,
        32: True,
        64: False,
    }


def test_estimate_sla_bounds_lt():
    sla_data, (max_passing, min_failing), history = _run_estimate_sla_bounds(
        _var2metric_identity,
        SLALessThan(target=32),
        init_value=1,
        max_value=100,
    )

    assert max_passing == 16
    assert min_failing == 32

    assert {val: margin <= 0 for val, margin in history.items()} == {
        1: True,
        2: True,
        4: True,
        8: True,
        16: True,
        32: False,
    }


def test_estimate_sla_bounds_oob():
    sla_data, (max_passing, min_failing), history = _run_estimate_sla_bounds(
        _var2metric_identity,
        SLALessThanOrEqualTo(target=32),
        init_value=64,
        max_value=128,
    )

    assert max_passing == 0
    assert min_failing == 64

    assert {val: margin <= 0 for val, margin in history.items()} == {
        64: False,
    }


def _run_test_find_sla_value_le(
    var2metric: Callable[[ParameterSweepItem], list[dict[str, float]]],
    criterion: SLACriterionBase,
    min_value: int,
    max_value: int,
):
    with _set_return_value(var2metric):
        return _find_sla_value(
            server=None,
            bench_cmd=[],
            serve_comb=ParameterSweepItem(),
            bench_comb=ParameterSweepItem(),
            sla_comb=SLASweepItem({"request_throughput": criterion}),
            base_path=Path(""),
            num_runs=1,
            dry_run=False,
            sla_variable="request_rate",
            min_value=min_value,
            max_value=max_value,
        )


def test_find_sla_value_le():
    sla_data, sla_value, history = _run_test_find_sla_value_le(
        _var2metric_identity,
        SLALessThanOrEqualTo(target=50.0),
        min_value=32,
        max_value=64,
    )

    assert sla_value == 50
    assert {val: margin <= 0 for val, margin in history.items()} == {
        48: True,
        56: False,
        52: False,
        50: True,
        51: False,
    }


def test_find_sla_value_lt():
    sla_data, sla_value, history = _run_test_find_sla_value_le(
        _var2metric_identity,
        SLALessThan(target=50.0),
        min_value=32,
        max_value=64,
    )

    assert sla_value == 49
    assert {val: margin <= 0 for val, margin in history.items()} == {
        48: True,
        56: False,
        52: False,
        50: False,
        49: True,
    }


def test_find_sla_value_oob():
    sla_data, sla_value, history = _run_test_find_sla_value_le(
        _var2metric_identity,
        SLALessThanOrEqualTo(target=50.0),
        min_value=64,
        max_value=128,
    )

    assert sla_value == 64
    assert {val: margin <= 0 for val, margin in history.items()} == {
        96: False,
        80: False,
        72: False,
        68: False,
        66: False,
        65: False,
        64: False,
    }
