# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for benchmark-client concurrency accounting."""

import asyncio

import numpy as np
import pytest
from aiohttp import web

from vllm.benchmarks.datasets import SampleRequest
from vllm.benchmarks.serve import TaskType, benchmark


@pytest.mark.asyncio
async def test_benchmark_reports_client_concurrency_queue_time():
    async def handler(request: web.Request) -> web.StreamResponse:
        await request.json()
        response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await response.prepare(request)
        await asyncio.sleep(0.05)
        await response.write(b'data: {"choices":[{"index":0,"text":"x"}]}\n\n')
        await response.write(
            b'data: {"choices":[],"usage":{"prompt_tokens":1,'
            b'"completion_tokens":1,"total_tokens":2}}\n\n'
        )
        await response.write(b"data: [DONE]\n\n")
        await response.write_eof()
        return response

    app = web.Application()
    app.router.add_post("/v1/completions", handler)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]

    try:
        result = await benchmark(
            task_type=TaskType.GENERATION,
            endpoint_type="openai",
            api_url=f"http://127.0.0.1:{port}/v1/completions",
            base_url=f"http://127.0.0.1:{port}",
            model_id="mock",
            model_name="mock",
            tokenizer=None,
            input_requests=[
                SampleRequest(prompt="hi", prompt_len=1, expected_output_len=1)
                for _ in range(3)
            ],
            logprobs=None,
            request_rate=float("inf"),
            burstiness=float("inf"),
            disable_tqdm=True,
            num_warmups=0,
            profile=False,
            selected_percentile_metrics=["ttft", "e2el"],
            selected_percentiles=[99.0],
            ignore_eos=False,
            goodput_config_dict={},
            max_concurrency=1,
            lora_modules=None,
            extra_headers=None,
            extra_body=None,
        )
    finally:
        await runner.cleanup()

    queue_times = result["queue_times"]
    assert queue_times[0] < 0.03
    assert queue_times[1] > 0.03
    assert queue_times[2] > queue_times[1] + 0.03

    e2els_including_queue = [
        latency + queue_time
        for latency, queue_time in zip(result["latencies"], queue_times)
    ]
    assert result["mean_client_queue_time_ms"] == pytest.approx(
        np.mean(queue_times) * 1000
    )
    assert result["p99_client_queue_time_ms"] == pytest.approx(
        np.percentile(queue_times, 99) * 1000
    )
    assert result["mean_e2el_including_client_queue_ms"] == pytest.approx(
        np.mean(e2els_including_queue) * 1000
    )
    assert result["p99_e2el_including_client_queue_ms"] == pytest.approx(
        np.percentile(e2els_including_queue, 99) * 1000
    )
