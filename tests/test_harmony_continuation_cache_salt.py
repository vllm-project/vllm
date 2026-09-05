# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest

from vllm.entrypoints.openai.responses.context import HarmonyContext
from vllm.entrypoints.openai.responses.protocol import ResponsesRequest
from vllm.entrypoints.openai.responses.serving import OpenAIServingResponses
from vllm.inputs import tokens_input
from vllm.sampling_params import SamplingParams

pytestmark = [pytest.mark.cpu_test, pytest.mark.skip_global_cleanup]


class _EngineClient:
    def __init__(self):
        self.inputs = []

    def generate(self, engine_input, *_args, **_kwargs):
        self.inputs.append(engine_input)

        async def empty_generator():
            if False:
                yield None

        return empty_generator()


@pytest.mark.asyncio
async def test_harmony_tool_continuation_preserves_cache_salt():
    serving = OpenAIServingResponses.__new__(OpenAIServingResponses)
    serving.model_config = SimpleNamespace(max_model_len=128)
    serving.engine_client = _EngineClient()
    serving._log_inputs = lambda *_args, **_kwargs: None

    context = HarmonyContext.__new__(HarmonyContext)
    context.request = ResponsesRequest(
        model="m",
        input="hello",
        cache_salt="victim-salt",
    )
    context.append_output = lambda _output: None
    context.append_tool_output = lambda _output: None
    context.render_for_completion = lambda: [10, 11, 12]

    needs_tool = iter([True, False])
    context.need_builtin_tool_call = lambda: next(needs_tool)

    async def call_tool():
        return []

    context.call_tool = call_tool

    async for _ in serving._generate_with_builtin_tools(
        request_id="resp-test",
        engine_input=tokens_input([1, 2], cache_salt="victim-salt"),
        sampling_params=SamplingParams(max_tokens=4),
        context=context,
    ):
        pass

    assert len(serving.engine_client.inputs) == 2
    assert serving.engine_client.inputs[1]["cache_salt"] == "victim-salt"
