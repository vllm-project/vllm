# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json

import pytest

from vllm.entrypoints.generate.base.protocol import RequestResponseMetadata
from vllm.entrypoints.openai.completion.protocol import CompletionRequest
from vllm.entrypoints.openai.completion.serving import OpenAIServingCompletion
from vllm.outputs import CompletionOutput, RequestOutput

PROMPT = "Hi there"
PROMPT_TOKEN_IDS = [9, 10]


def _output() -> RequestOutput:
    # echo with max_tokens=0 runs the engine with max_tokens=1, so the
    # output carries one sampled token that the response does not return.
    return RequestOutput(
        request_id="r",
        prompt=PROMPT,
        prompt_token_ids=PROMPT_TOKEN_IDS,
        prompt_logprobs=None,
        outputs=[
            CompletionOutput(
                index=0,
                text=" X",
                token_ids=[42],
                cumulative_logprob=None,
                logprobs=None,
                finish_reason="length",
            )
        ],
        finished=True,
    )


def _serving() -> OpenAIServingCompletion:
    serving = OpenAIServingCompletion.__new__(OpenAIServingCompletion)
    serving.enable_prompt_tokens_details = False
    serving.enable_force_include_usage = False
    serving.enable_per_request_metrics = False
    serving.system_fingerprint = None
    serving.return_tokens_as_token_ids = False
    return serving


def _request(stream: bool) -> CompletionRequest:
    return CompletionRequest(
        model="m",
        prompt=PROMPT,
        max_tokens=0,
        echo=True,
        return_token_ids=True,
        stream=stream,
        stream_options={"include_usage": True} if stream else None,
    )


def test_echo_without_generation_counts_no_completion_tokens():
    request = _request(stream=False)
    assert request.to_sampling_params(0).max_tokens == 1

    response = _serving().request_output_to_completion_response(
        [_output()],
        request,
        "r",
        0,
        "m",
        None,
        RequestResponseMetadata(request_id="r"),
    )

    assert response.usage.completion_tokens == 0
    assert response.usage.total_tokens == len(PROMPT_TOKEN_IDS)
    assert response.choices[0].token_ids == []


@pytest.mark.asyncio
async def test_stream_echo_without_generation_counts_no_completion_tokens():
    async def results():
        yield 0, _output()

    chunks = []
    async for line in _serving().completion_stream_generator(
        _request(stream=True),
        [{"prompt": PROMPT}],
        results(),
        "r",
        0,
        "m",
        num_prompts=1,
        tokenizer=None,
        request_metadata=RequestResponseMetadata(request_id="r"),
    ):
        data = line.removeprefix("data: ").strip()
        if data != "[DONE]":
            chunks.append(json.loads(data))

    assert chunks[-1]["usage"]["completion_tokens"] == 0
    assert chunks[-1]["usage"]["total_tokens"] == len(PROMPT_TOKEN_IDS)
    assert chunks[0]["choices"][0]["token_ids"] == []
