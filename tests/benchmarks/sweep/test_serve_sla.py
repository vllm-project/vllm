# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Callable
from pathlib import Path
from unittest.mock import patch

from vllm.benchmarks.sweep.param_sweep import ParameterSweepItem
from vllm.benchmarks.sweep.serve_sla import _estimate_sla_bounds, _find_sla_value
from vllm.benchmarks.sweep.server import ServerProcess
from vllm.benchmarks.sweep.sla_sweep import (
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


def test_estimate_sla_bounds_le():
    sla_comb = SLASweepItem({"request_throughput": SLALessThanOrEqualTo(target=32)})

    with _set_return_value(_var2metric_identity):
        sla_data, (max_passing, min_failing), history = _estimate_sla_bounds(
            server=None,
            bench_cmd=[],
            serve_comb=ParameterSweepItem(),
            bench_comb=ParameterSweepItem(),
            sla_comb=sla_comb,
            base_path=Path(""),
            num_runs=1,
            dry_run=False,
            sla_variable="request_rate",
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
    sla_comb = SLASweepItem({"request_throughput": SLALessThan(target=32)})

    with _set_return_value(_var2metric_identity):
        sla_data, (max_passing, min_failing), history = _estimate_sla_bounds(
            server=None,
            bench_cmd=[],
            serve_comb=ParameterSweepItem(),
            bench_comb=ParameterSweepItem(),
            sla_comb=sla_comb,
            base_path=Path(""),
            num_runs=1,
            dry_run=False,
            sla_variable="request_rate",
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


def test_find_sla_value_le():
    sla_comb = SLASweepItem({"request_throughput": SLALessThanOrEqualTo(target=50.0)})

    with _set_return_value(_var2metric_identity):
        sla_data, sla_value, history = _find_sla_value(
            server=None,
            bench_cmd=[],
            serve_comb=ParameterSweepItem(),
            bench_comb=ParameterSweepItem(),
            sla_comb=sla_comb,
            base_path=Path(""),
            num_runs=1,
            dry_run=False,
            sla_variable="request_rate",
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
    sla_comb = SLASweepItem({"request_throughput": SLALessThan(target=50.0)})

    with _set_return_value(_var2metric_identity):
        sla_data, sla_value, history = _find_sla_value(
            server=None,
            bench_cmd=[],
            serve_comb=ParameterSweepItem(),
            bench_comb=ParameterSweepItem(),
            sla_comb=sla_comb,
            base_path=Path(""),
            num_runs=1,
            dry_run=False,
            sla_variable="request_rate",
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
