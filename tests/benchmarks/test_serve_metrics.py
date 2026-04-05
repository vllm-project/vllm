# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.benchmarks.datasets import SampleRequest
from vllm.benchmarks.lib.endpoint_request_func import RequestFuncOutput
from vllm.benchmarks.serve import calculate_metrics


def test_peak_output_throughput_not_less_than_average_with_chunked_streaming():
    # Simulate two successful requests where streaming events are chunked
    # (few ITL events) but each request still generates many output tokens.
    input_requests = [
        SampleRequest(prompt='a', prompt_len=16, expected_output_len=0),
        SampleRequest(prompt='b', prompt_len=16, expected_output_len=0),
    ]
    outputs = [
        RequestFuncOutput(
            success=True,
            start_time=0.0,
            latency=4.0,
            ttft=1.0,
            itl=[1.5],  # chunk-level events, not per-token events
            output_tokens=600,
            generated_text='x' * 600,
        ),
        RequestFuncOutput(
            success=True,
            start_time=0.5,
            latency=4.0,
            ttft=1.0,
            itl=[1.0],
            output_tokens=600,
            generated_text='y' * 600,
        ),
    ]

    metrics, _ = calculate_metrics(
        input_requests=input_requests,
        outputs=outputs,
        dur_s=4.5,
        tokenizer=None,  # output_tokens already present
        selected_percentiles=[50],
        goodput_config_dict={},
    )

    assert metrics.output_throughput > 0
    assert metrics.max_output_tokens_per_s >= metrics.output_throughput
