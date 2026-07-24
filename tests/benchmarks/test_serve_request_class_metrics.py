# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.benchmarks.datasets import SampleRequest
from vllm.benchmarks.lib.endpoint_request_func import RequestFuncOutput
from vllm.benchmarks.serve import calculate_request_class_metrics


@pytest.mark.benchmark
@pytest.mark.skip_global_cleanup
def test_calculate_request_class_metrics_groups_latency() -> None:
    requests = [
        SampleRequest(
            prompt="a",
            prompt_len=10,
            expected_output_len=3,
            request_metadata={"workload_class": "long_prefill"},
        ),
        SampleRequest(
            prompt="b",
            prompt_len=5,
            expected_output_len=8,
            request_metadata={"workload_class": "long_decode"},
        ),
        SampleRequest(
            prompt="c",
            prompt_len=6,
            expected_output_len=8,
            request_metadata={"workload_class": "long_decode"},
        ),
    ]
    outputs = [
        RequestFuncOutput(
            success=True,
            latency=0.30,
            output_tokens=3,
            ttft=0.10,
            itl=[0.10, 0.10],
            prompt_len=10,
        ),
        RequestFuncOutput(
            success=True,
            latency=0.90,
            output_tokens=8,
            ttft=0.20,
            itl=[0.10] * 7,
            prompt_len=5,
        ),
        RequestFuncOutput(
            success=False,
            latency=0.0,
            output_tokens=0,
            ttft=0.0,
            prompt_len=6,
            error="boom",
        ),
    ]

    metrics = calculate_request_class_metrics(
        input_requests=requests,
        outputs=outputs,
        actual_output_lens=[3, 8, 0],
        selected_percentiles=[50, 99],
    )

    assert metrics["long_prefill"]["requests"] == 1
    assert metrics["long_prefill"]["completed"] == 1
    assert metrics["long_prefill"]["mean_ttft_ms"] == 100.0
    assert metrics["long_prefill"]["total_output_tokens"] == 3

    assert metrics["long_decode"]["requests"] == 2
    assert metrics["long_decode"]["completed"] == 1
    assert metrics["long_decode"]["failed"] == 1
    assert metrics["long_decode"]["total_input_tokens"] == 11
    assert metrics["long_decode"]["total_output_tokens"] == 8
    assert metrics["long_decode"]["mean_e2el_ms"] == 900.0
    assert metrics["long_decode"]["ttft_percentiles_ms"]["p99"] == 200.0
