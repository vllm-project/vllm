# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest

from vllm.benchmarks.lib.endpoint_request_func import RequestFuncOutput
from vllm.benchmarks.serve import calculate_metrics


@pytest.mark.parametrize(
    "intervals,expected",
    [
        ([(0.0, 0.125), (0.25, 0.375), (0.5, 0.625)], 1),
        ([(0.0, 0.5), (0.5, 1.0)], 1),
        ([(0.0, 0.75), (0.25, 0.5), (0.5, 1.0)], 2),
        ([(0.0, 2.0), (0.5, 1.5), (0.75, 1.0)], 3),
    ],
)
def test_peak_concurrency_counts_overlapping_requests(intervals, expected):
    outputs = [
        RequestFuncOutput(
            success=True,
            start_time=start,
            latency=end - start,
            ttft=end - start,
            output_tokens=1,
        )
        for start, end in intervals
    ]
    metrics, _ = calculate_metrics(
        input_requests=[],
        outputs=outputs,
        dur_s=2.0,
        tokenizer=None,
        selected_percentiles=[],
        goodput_config_dict={},
    )
    assert metrics.max_concurrent_requests == expected
