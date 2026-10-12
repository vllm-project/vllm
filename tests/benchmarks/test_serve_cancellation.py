# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Cancellation workload scheduling and result accounting, without a model."""

import asyncio
import time
import warnings
from unittest.mock import AsyncMock

import pytest

from vllm.benchmarks import serve
from vllm.benchmarks.datasets import SampleRequest
from vllm.benchmarks.lib.endpoint_request_func import RequestFuncOutput

pytestmark = pytest.mark.skip_global_cleanup


def _benchmark_kwargs():
    return dict(
        task_type=serve.TaskType.GENERATION,
        endpoint_type="openai",
        api_url="http://test/v1/completions",
        base_url="http://test",
        model_id="test",
        model_name="test",
        tokenizer=None,
        input_requests=[SampleRequest(str(i), 7, 2) for i in range(4)],
        logprobs=None,
        request_rate=float("inf"),
        burstiness=1.0,
        disable_tqdm=True,
        num_warmups=1,
        profile=False,
        selected_percentile_metrics=["e2el"],
        selected_percentiles=[50.0],
        ignore_eos=False,
        goodput_config_dict={},
        max_concurrency=1,
        lora_modules=None,
        extra_headers=None,
        extra_body=None,
        ready_check_timeout_sec=2,
        cancellation_rate=0.5,
        cancellation_delay=0.02,
        cancellation_seed=0,
    )


@pytest.mark.parametrize(
    "rate, expected",
    [
        (0.0, []),
        (0.5, [1, 3]),
        (1.0, [0, 1, 2, 3]),
    ],
)
def test_cancellation_starts_after_admission_and_leaves_warmups_alone(
    monkeypatch, rate, expected
):
    finished = []
    cancelled = []

    async def request(request_func_input, session, pbar=None):
        started = time.perf_counter()
        try:
            await asyncio.sleep(0.06)
        except asyncio.CancelledError:
            cancelled.append(request_func_input.prompt)
            raise
        finished.append(request_func_input.prompt)
        return RequestFuncOutput(
            success=True,
            start_time=started,
            latency=time.perf_counter() - started,
            prompt_len=7,
            output_tokens=2,
            ttft=0.01,
            itl=[0.05],
        )

    monkeypatch.setitem(serve.ASYNC_REQUEST_FUNCS, "openai", request)
    for name in ("fetch_spec_decode_metrics", "fetch_diffusion_metrics"):
        monkeypatch.setattr(serve, name, AsyncMock(return_value=None))

    kwargs = _benchmark_kwargs()
    kwargs["cancellation_rate"] = rate
    result = asyncio.run(serve.benchmark(**kwargs))
    assert cancelled == [str(i) for i in expected]
    # Readiness and warmup use prompt 0 and are never selected.
    assert finished == ["0", "0"] + [str(i) for i in range(4) if i not in expected]
    assert result["completed"] == 4 - len(expected) and result["failed"] == 0
    assert result["total_output_tokens"] == 2 * (4 - len(expected))
    if rate:
        assert result["cancellations"] == [i in expected for i in range(4)]
        assert result["cancellation_targeted"] == result["cancelled"] == len(expected)
        if rate == 0.5:
            assert result["queue_times"][1] > result["cancellation_delay"]
        for i in expected:
            assert result["latencies"][i] >= result["cancellation_delay"]
    else:
        assert "cancelled" not in result and "cancellations" not in result


def test_cancelled_requests_are_not_failures_or_completed_tokens():
    outputs = [
        RequestFuncOutput(success=True, output_tokens=2, prompt_len=7, latency=1),
        RequestFuncOutput(cancelled=True, latency=0.1),
        RequestFuncOutput(error="HTTP 500"),
    ]
    metrics, lengths = serve.calculate_metrics([], outputs, 2, None, [50], {})
    assert (metrics.completed, metrics.failed, metrics.cancelled) == (1, 1, 1)
    assert lengths == [2, 0, 0]
    assert metrics.total_output == 2 and metrics.request_throughput == 0.5
    with warnings.catch_warnings(record=True) as caught:
        metrics, _ = serve.calculate_metrics([], outputs[1:2], 2, None, [50], {})
    assert metrics.cancelled == 1 and metrics.failed == metrics.completed == 0
    assert not any("All requests failed" in str(w.message) for w in caught)


@pytest.mark.parametrize(
    "rate,delay,backend",
    [
        (-0.1, 1, "openai"),
        (1.1, 1, "openai"),
        (float("nan"), 1, "openai"),
        (0.5, 0, "openai"),
        (0.5, float("inf"), "openai"),
        (0.5, float("nan"), "openai"),
        (0.5, 1, "openai-embeddings"),
    ],
)
def test_invalid_cancellation_workload_fails_before_opening_session(
    rate, delay, backend
):
    kwargs = _benchmark_kwargs()
    kwargs.update(
        cancellation_rate=rate, cancellation_delay=delay, endpoint_type=backend
    )
    with pytest.raises(ValueError):
        asyncio.run(serve.benchmark(**kwargs))
