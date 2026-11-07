# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from benchmarks.kernels import benchmark_device_communicators as benchmark
from benchmarks.kernels.benchmark_device_communicators import (
    _build_trial_record,
    _delay_us_to_cycles,
    _parse_tensor_shape,
    _require_calibration_results,
)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("8627x8192", (8627, 8192)),
        ("16X4096", (16, 4096)),
    ],
)
def test_parse_tensor_shape(value, expected):
    assert _parse_tensor_shape(value) == expected


@pytest.mark.parametrize("value", ["8192", "1x2x3", "0x8192", "axb"])
def test_parse_tensor_shape_rejects_invalid_values(value):
    with pytest.raises(ValueError):
        _parse_tensor_shape(value)


def test_delay_us_to_cycles_uses_reported_clock_rate():
    assert _delay_us_to_cycles(250, 1_980_000) == 495_000


@pytest.mark.parametrize(
    ("delayed_rank", "residence", "measured_delay", "gap"),
    [(0, 1.2, 0.003, 0.0), (1, 0.9, 0.259, 0.3)],
)
def test_build_trial_record_summarizes_injected_rank_without_assuming_arrival_order(
    delayed_rank,
    residence,
    measured_delay,
    gap,
):
    records = [
        {
            "rank": 1,
            "host": "node0",
            "host_enqueue_ns": 1_001_500,
            "gpu_collective_ms": 0.9,
            "measured_gpu_delay_ms": 0.259,
        },
        {
            "rank": 0,
            "host": "node0",
            "host_enqueue_ns": 1_000_000,
            "gpu_collective_ms": 1.2,
            "measured_gpu_delay_ms": 0.003,
        },
    ]

    trial = _build_trial_record(3, records, delayed_rank)

    assert trial["trial"] == 3
    assert trial["host_enqueue_spread_us"] == 1.5
    assert trial["max_gpu_collective_ms"] == 1.2
    assert trial["delayed_rank_gpu_collective_ms"] == residence
    assert trial["measured_gpu_delay_ms"] == measured_delay
    assert trial["max_minus_delayed_rank_ms"] == pytest.approx(gap)
    assert [record["rank"] for record in trial["ranks"]] == [0, 1]


def test_build_trial_record_does_not_compare_cross_node_host_clocks():
    records = [
        {
            "rank": 0,
            "host": "node0",
            "host_enqueue_ns": 1_000_000,
            "gpu_collective_ms": 1.0,
            "measured_gpu_delay_ms": 0.003,
        },
        {
            "rank": 1,
            "host": "node1",
            "host_enqueue_ns": 2_000_000,
            "gpu_collective_ms": 1.1,
            "measured_gpu_delay_ms": 0.003,
        },
    ]

    assert _build_trial_record(0, records, 1)["host_enqueue_spread_us"] is None


def test_print_calibration_reports_measured_delay_and_injected_rank(capsys):
    benchmark.print_skew_calibration_results(
        [
            {
                "backend": "pynccl",
                "shape": [8627, 8192],
                "execution_mode": "eager",
                "delayed_rank": 1,
                "requested_delay_us": 250,
                "trials": [
                    {
                        "max_gpu_collective_ms": 0.740,
                        "delayed_rank_gpu_collective_ms": 0.488,
                        "measured_gpu_delay_ms": 0.259,
                        "max_minus_delayed_rank_ms": 0.252,
                    }
                ],
            }
        ]
    )

    output = capsys.readouterr().out
    assert "delay=250 us (measured p50=259.0 us)" in output
    assert "max-rank residence p50=0.7400 ms" in output
    assert "injected-rank (1) residence p50=0.4880 ms" in output
    assert "max-minus-injected p50=0.2520 ms" in output
    assert "zero-delay baselines include natural skew" in output


@pytest.mark.parametrize(("delay", "clock"), [(-1, 1_980_000), (1, 0)])
def test_delay_rejects_invalid_calibration_inputs(delay, clock):
    with pytest.raises(ValueError):
        _delay_us_to_cycles(delay, clock)


def test_calibration_rejects_partial_explicit_requests():
    results = [{"backend": "pynccl", "shape": [8, 8192], "requested_delay_us": 0}]
    _require_calibration_results(results, ["pynccl"], [(8, 8192)], [0])
    with pytest.raises(RuntimeError, match="Unavailable or unsupported"):
        _require_calibration_results(results, ["pynccl"], [(8, 8192)], [0, 1000])
    with pytest.raises(RuntimeError, match="Unavailable or unsupported"):
        _require_calibration_results(results, ["pynccl-symm"], [(8, 8192)], [0])


def test_calibration_rejects_empty_auto_selection():
    with pytest.raises(RuntimeError, match="No supported"):
        _require_calibration_results([], None, [(8, 8192)], [0])


@pytest.mark.parametrize(
    "args",
    [
        ["--skew-rank", "1"],
        ["--calibration-execution-mode", "eager"],
        ["--calibration-backends", "pynccl"],
        ["--skew-delays-us", "-1"],
        ["--skew-delays-us", "0", "--num-trials", "0"],
        ["--skew-delays-us", "0", "--num-warmup", "-1"],
        ["--skew-delays-us", "0", "--sequence-lengths", "0"],
    ],
)
def test_invalid_calibration_cli_fails_before_distributed_init(monkeypatch, args):
    def unexpected_init(*args, **kwargs):
        pytest.fail("invalid input reached distributed initialization")

    monkeypatch.setattr("sys.argv", ["benchmark", *args])
    monkeypatch.setattr(benchmark.dist, "init_process_group", unexpected_init)
    with pytest.raises(SystemExit) as error:
        benchmark.main()
    assert error.value.code == 2
