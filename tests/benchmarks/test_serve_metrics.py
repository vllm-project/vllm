# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.benchmarks.lib.endpoint_request_func import RequestFuncOutput
from vllm.benchmarks.serve import (
    _calculate_peak_concurrency,
    _calculate_peak_output_tokens_per_s_approx,
    calculate_metrics,
)


def test_sliding_window_finds_peak_across_fixed_bucket_boundary():
    outputs = [
        RequestFuncOutput(success=True, start_time=0.0, ttft=0.6),
        RequestFuncOutput(success=True, start_time=0.0, ttft=1.4),
    ]

    peak, event_times, rolling_rates = _calculate_peak_output_tokens_per_s_approx(
        outputs
    )

    assert peak == 2.0
    assert event_times == pytest.approx([0.6, 1.4, 1.6, 2.4])
    assert rolling_rates == pytest.approx([1.0, 2.0, 1.0, 0.0])


def test_calculate_metrics_uses_sliding_window_peak():
    outputs = [
        RequestFuncOutput(
            success=True,
            start_time=0.0,
            latency=2.0,
            ttft=0.6,
            output_tokens=1,
        ),
        RequestFuncOutput(
            success=True,
            start_time=0.0,
            latency=2.0,
            ttft=1.4,
            output_tokens=1,
        ),
    ]

    metrics, _ = calculate_metrics(
        input_requests=[],
        outputs=outputs,
        dur_s=2.0,
        tokenizer=None,
        selected_percentiles=[99],
        goodput_config_dict={},
    )

    assert metrics.max_output_tokens_per_s == 2.0
    assert metrics.max_concurrent_requests == 2


def test_sliding_window_excludes_chunk_at_exact_window_boundary():
    outputs = [
        RequestFuncOutput(success=True, start_time=0.0, ttft=0.5),
        RequestFuncOutput(success=True, start_time=0.0, ttft=1.5),
    ]

    peak, event_times, rolling_rates = _calculate_peak_output_tokens_per_s_approx(
        outputs
    )

    assert peak == 1.0
    assert event_times == pytest.approx([0.5, 1.5, 2.5])
    assert rolling_rates == pytest.approx([1.0, 1.0, 0.0])


def test_peak_concurrency_uses_half_open_request_intervals():
    outputs = [
        RequestFuncOutput(start_time=0.0, latency=1.0),
        RequestFuncOutput(start_time=1.0, latency=1.0),
        RequestFuncOutput(start_time=0.5, latency=1.0),
        RequestFuncOutput(start_time=0.0, latency=0.0),
    ]

    peak, event_times, concurrency = _calculate_peak_concurrency(outputs)

    assert peak == 2
    assert event_times == pytest.approx([0.0, 0.5, 1.0, 1.5, 2.0])
    assert concurrency == [1, 2, 2, 1, 0]
