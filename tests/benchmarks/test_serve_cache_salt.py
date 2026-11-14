# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The ready check and warmups of `vllm bench serve` send request 0 with a unique
cache_salt, so they don't prime the prefix cache for the run, and drop the salt
only if the server rejects it.

Driven against a stub OpenAI completions server: no GPU or model.
"""

import asyncio
import json
from collections.abc import Callable

import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

from vllm.benchmarks.datasets import SampleRequest
from vllm.benchmarks.serve import TaskType, benchmark

pytestmark = pytest.mark.skip_global_cleanup


async def _salted_requests(status_for: Callable[[int, bool], int]) -> list[bool]:
    """Runs the ready check, 2 warmups and 1 measured request. Returns, for each
    request the server got, whether it carried a cache_salt.

    `status_for(n, salted)` is the HTTP status for the server's n-th request.
    """
    salted: list[bool] = []

    async def completions(request: web.Request) -> web.StreamResponse:
        salted.append("cache_salt" in await request.json())
        status = status_for(len(salted), salted[-1])
        if status != 200:
            return web.Response(status=status)
        response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await response.prepare(request)
        for chunk in (
            {"choices": [{"text": "hi"}]},
            {"usage": {"prompt_tokens": 1, "completion_tokens": 1}},
        ):
            await response.write(f"data: {json.dumps(chunk)}\n\n".encode())
        await response.write(b"data: [DONE]\n\n")
        return response

    app = web.Application()
    app.router.add_post("/v1/completions", completions)
    async with TestServer(app) as server:
        base_url = str(server.make_url("")).rstrip("/")
        await benchmark(
            task_type=TaskType.GENERATION,
            endpoint_type="openai",
            api_url=f"{base_url}/v1/completions",
            base_url=base_url,
            model_id="stub",
            model_name="stub",
            tokenizer=None,
            input_requests=[
                SampleRequest(prompt="hello", prompt_len=1, expected_output_len=1)
            ],
            logprobs=None,
            request_rate=float("inf"),
            burstiness=1.0,
            disable_tqdm=True,
            num_warmups=2,
            profile=False,
            selected_percentile_metrics=["ttft"],
            selected_percentiles=[99.0],
            ignore_eos=False,
            goodput_config_dict={},
            max_concurrency=1,
            lora_modules=None,
            extra_headers=None,
            extra_body=None,
            ready_check_timeout_sec=30,
        )
    return salted


def test_transient_ready_check_error_keeps_cache_salt():
    salted = asyncio.run(_salted_requests(lambda n, _: 503 if n == 1 else 200))
    # ready check (503), ready check retry, 2 warmups, measured request
    assert salted == [True, True, True, True, False]


def test_rejected_cache_salt_falls_back_to_unsalted():
    salted = asyncio.run(_salted_requests(lambda _, has_salt: 400 if has_salt else 200))
    # ready check (400) + unsalted, 2 warmups (400) + 2 unsalted, measured request
    assert salted == [True, False, True, True, False, False, False]
