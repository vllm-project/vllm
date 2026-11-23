# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import asyncio

import aiohttp
import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

from vllm.benchmarks.lib.endpoint_request_func import (
    ASYNC_REQUEST_FUNCS,
    RequestFuncInput,
)

pytestmark = pytest.mark.skip_global_cleanup


@pytest.mark.parametrize(
    ("backend", "endpoint"),
    [("vllm-systemone", "/v1/systemone"), ("openai-decisions", "/v1/decisions")],
)
def test_decisions_request_is_accepted_by_the_server_schema(backend, endpoint):
    from vllm.entrypoints.generate.structured_decisions.protocol import (
        StructuredDecisionRequest,
    )
    from vllm.entrypoints.generate.structured_decisions.serving import parse_questions
    from vllm.entrypoints.openai.decisions.adapters import make_read_question
    from vllm.entrypoints.openai.decisions.protocol import DecisionRequest
    from vllm.entrypoints.openai.decisions.strategies import DecisionLimits

    seen = []

    async def handler(request: web.Request) -> web.Response:
        seen.append(await request.json())
        return web.json_response({"usage": {"input_tokens": 42}})

    async def run():
        app = web.Application()
        app.router.add_post(endpoint, handler)
        server = TestServer(app)
        await server.start_server()
        try:
            request_input = RequestFuncInput(
                prompt="My card was charged twice.",
                api_url=str(server.make_url(endpoint)),
                prompt_len=6,
                output_len=0,
                model="Qwen/Qwen3-0.6B",
                ignore_eos=True,
            )
            async with aiohttp.ClientSession() as session:
                return await ASYNC_REQUEST_FUNCS[backend](request_input, session)
        finally:
            await server.close()

    output = asyncio.run(run())

    assert output.success
    assert (output.prompt_len, output.num_input_sequences) == (42, 3)
    if backend == "vllm-systemone":
        request = StructuredDecisionRequest.model_validate(seen[0])
        parse_questions(request, DecisionLimits(max_questions=64, max_options=26))
    else:
        request = DecisionRequest.model_validate(seen[0])
        for index, question in enumerate(request.questions):
            make_read_question(index, question)
