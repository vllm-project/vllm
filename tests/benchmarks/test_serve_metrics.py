# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU-only regression tests for ``bench serve`` metric aggregation."""

from vllm.benchmarks.lib.endpoint_request_func import RequestFuncOutput
from vllm.benchmarks.serve import calculate_metrics


def _successful_output(start_time: float, latency: float) -> RequestFuncOutput:
    return RequestFuncOutput(
        success=True,
        start_time=start_time,
        latency=latency,
        output_tokens=1,
        prompt_len=1,
    )


def _peak_concurrency(outputs: list[RequestFuncOutput]) -> int:
    metrics, _ = calculate_metrics(
        input_requests=[],
        outputs=outputs,
        dur_s=2.0,
        tokenizer=None,
        selected_percentiles=[],
        goodput_config_dict={},
    )
    return metrics.max_concurrent_requests


def test_calculate_metrics_does_not_count_adjacent_requests_as_concurrent() -> None:
    outputs = [
        _successful_output(start_time=0.0, latency=1.0),
        _successful_output(start_time=1.0, latency=1.0),
    ]
    assert _peak_concurrency(outputs) == 1


def test_calculate_metrics_counts_overlapping_requests() -> None:
    outputs = [
        _successful_output(start_time=0.0, latency=1.5),
        _successful_output(start_time=1.0, latency=1.0),
    ]
    assert _peak_concurrency(outputs) == 2


def test_calculate_metrics_subsecond_sequential_requests() -> None:
    # 5 non-overlapping sub-second requests inside second 0
    outputs = [
        _successful_output(start_time=0.0, latency=0.1),
        _successful_output(start_time=0.2, latency=0.1),
        _successful_output(start_time=0.4, latency=0.1),
        _successful_output(start_time=0.6, latency=0.1),
        _successful_output(start_time=0.8, latency=0.1),
    ]
    assert _peak_concurrency(outputs) == 1


def test_calculate_metrics_zero_latency_request() -> None:
    # Instantaneous or cached requests with zero latency
    outputs = [
        _successful_output(start_time=0.0, latency=0.0),
    ]
    assert _peak_concurrency(outputs) == 1
