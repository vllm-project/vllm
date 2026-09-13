# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from collections.abc import AsyncGenerator

import pytest

from vllm.entrypoints.openai.engine.protocol import RequestResponseMetadata
from vllm.entrypoints.scale_out.token_in_token_out.protocol import GenerateRequest
from vllm.entrypoints.scale_out.token_in_token_out.serving import ServingTokens
from vllm.outputs import CompletionOutput, RequestOutput
from vllm.sampling_params import SamplingParams


async def _single_result(
    output: RequestOutput,
) -> AsyncGenerator[RequestOutput, None]:
    yield output


def _serving_tokens() -> ServingTokens:
    serving = object.__new__(ServingTokens)
    serving.enable_prompt_tokens_details = False
    serving.enable_log_outputs = False
    serving.request_logger = None
    return serving


def _request_output(stop_reason: int | str) -> RequestOutput:
    return RequestOutput(
        request_id="rid",
        prompt=None,
        prompt_token_ids=[1, 2],
        prompt_logprobs=None,
        outputs=[
            CompletionOutput(
                index=0,
                text="",
                token_ids=[42],
                cumulative_logprob=None,
                logprobs=None,
                finish_reason="stop",
                stop_reason=stop_reason,
            )
        ],
        finished=True,
    )


@pytest.mark.asyncio
async def test_generate_full_response_includes_stop_reason_from_engine():
    request = GenerateRequest(token_ids=[1, 2], sampling_params=SamplingParams())

    response = await _serving_tokens().serve_tokens_full_generator(
        request,
        _single_result(_request_output(42)),
        "rid",
        "test-model",
        RequestResponseMetadata(request_id="rid"),
    )

    choice = response.model_dump()["choices"][0]
    assert choice["finish_reason"] == "stop"
    assert choice["stop_reason"] == 42


@pytest.mark.asyncio
async def test_generate_stream_response_includes_stop_reason_from_engine():
    request = GenerateRequest(
        token_ids=[1, 2],
        sampling_params=SamplingParams(),
        stream=True,
    )

    chunks = []
    async for line in _serving_tokens().serve_tokens_stream_generator(
        request,
        _single_result(_request_output("done")),
        "rid",
        "test-model",
        RequestResponseMetadata(request_id="rid"),
    ):
        if not line.startswith("data: "):
            continue
        payload = line[len("data: ") :].strip()
        if payload == "[DONE]":
            break
        chunks.append(json.loads(payload))

    assert chunks
    choice = chunks[0]["choices"][0]
    assert choice["finish_reason"] == "stop"
    assert choice["stop_reason"] == "done"
