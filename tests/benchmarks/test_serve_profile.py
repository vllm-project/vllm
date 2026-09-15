# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
from unittest.mock import AsyncMock

import pytest

from vllm.benchmarks import serve
from vllm.benchmarks.datasets import SampleRequest
from vllm.benchmarks.lib.endpoint_request_func import RequestFuncOutput


@pytest.mark.parametrize(
    "headers", [None, {"Authorization": "Bearer test-key", "X-Tenant": "test"}]
)
def test_benchmark_profile_headers(monkeypatch, headers):
    request_func = AsyncMock(
        return_value=RequestFuncOutput(
            success=True, latency=0.1, ttft=0.1, prompt_len=1, output_tokens=1
        )
    )
    monkeypatch.setitem(serve.ASYNC_REQUEST_FUNCS, "test", request_func)
    monkeypatch.setattr(
        serve, "fetch_spec_decode_metrics", AsyncMock(return_value=None)
    )
    monkeypatch.setattr(serve, "fetch_diffusion_metrics", AsyncMock(return_value=None))

    asyncio.run(
        serve.benchmark(
            task_type=serve.TaskType.GENERATION,
            endpoint_type="test",
            api_url="http://localhost/v1/completions",
            base_url="http://localhost",
            model_id="test-model",
            model_name="test-model",
            tokenizer=None,
            input_requests=[
                SampleRequest(prompt="hello", prompt_len=1, expected_output_len=1)
            ],
            logprobs=None,
            request_rate=float("inf"),
            burstiness=1.0,
            disable_tqdm=True,
            num_warmups=0,
            profile=True,
            selected_percentile_metrics=[],
            selected_percentiles=[99.0],
            ignore_eos=False,
            goodput_config_dict={},
            max_concurrency=None,
            lora_modules=None,
            extra_headers=headers,
            extra_body=None,
            ready_check_timeout_sec=0,
        )
    )

    inputs = [
        call.kwargs["request_func_input"] for call in request_func.await_args_list
    ]
    assert [request.api_url for request in inputs] == [
        "http://localhost/start_profile",
        "http://localhost/v1/completions",
        "http://localhost/stop_profile",
    ]
    assert [request.extra_headers for request in inputs] == [headers] * 3
