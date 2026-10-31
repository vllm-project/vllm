# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio

import aiohttp
import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

from vllm.benchmarks.lib.endpoint_request_func import (
    RequestFuncInput,
    async_request_vllm_rerank,
)

pytestmark = pytest.mark.skip_global_cleanup


@pytest.mark.parametrize(
    "extra_body",
    [None, {}, {"top_n": 1, "truncate_prompt_tokens": 16}],
)
def test_rerank_request_preserves_body_overrides(extra_body):
    """Rerank benchmarks must send the requested workload settings over HTTP."""
    received = []

    async def rerank(request):
        received.append(await request.json())
        return web.json_response({"results": [], "usage": {"prompt_tokens": 12}})

    async def run():
        app = web.Application()
        app.router.add_post("/v1/rerank", rerank)
        async with TestServer(app) as server, aiohttp.ClientSession() as session:
            return await async_request_vllm_rerank(
                RequestFuncInput(
                    prompt=["query", "document one", "document two"],
                    api_url=str(server.make_url("/v1/rerank")),
                    prompt_len=12,
                    output_len=0,
                    model="reranker",
                    extra_body=extra_body,
                ),
                session,
            )

    output = asyncio.run(run())
    assert output.success, output.error
    assert output.num_input_sequences == 2
    assert output.prompt_len == 12
    assert received == [
        {
            "model": "reranker",
            "query": "query",
            "documents": ["document one", "document two"],
            "truncate_prompt_tokens": -1,
            **(extra_body or {}),
        }
    ]
