# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json

import pytest

from vllm.entrypoints.generate.base.protocol import RequestResponseMetadata
from vllm.entrypoints.openai.completion.protocol import CompletionRequest
from vllm.entrypoints.openai.completion.serving import OpenAIServingCompletion
from vllm.logprobs import Logprob
from vllm.outputs import CompletionOutput, RequestOutput

TOKENS = [(1, "abc"), (2, " de"), (3, "!")]
# Text deltas as the detokenizer streams them with stop=["\n\n"]: one char is
# held back until the request finishes, so the text lags the tokens.
TEXT_DELTAS = ["ab", "c d", "e!"]


def _logprobs(i: int) -> dict[int, Logprob]:
    token_id, token = TOKENS[i]
    return {token_id: Logprob(logprob=-1.0, rank=1, decoded_token=token)}


def _output(text: str, idxs: list[int], finished: bool) -> RequestOutput:
    return RequestOutput(
        request_id="r",
        prompt="Hi",
        prompt_token_ids=[9],
        prompt_logprobs=None,
        outputs=[
            CompletionOutput(
                index=0,
                text=text,
                token_ids=[TOKENS[i][0] for i in idxs],
                cumulative_logprob=None,
                logprobs=[_logprobs(i) for i in idxs],
                finish_reason="stop" if finished else None,
            )
        ],
        finished=finished,
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
        model="m", prompt="Hi", max_tokens=8, logprobs=0, stop=["\n\n"], stream=stream
    )


@pytest.mark.asyncio
async def test_stream_text_offset_matches_non_stream_with_stop_holdback():
    async def results():
        for i, text in enumerate(TEXT_DELTAS):
            yield 0, _output(text, [i], finished=i == len(TEXT_DELTAS) - 1)

    serving = _serving()
    stream_offsets = []
    async for line in serving.completion_stream_generator(
        _request(stream=True),
        [{"prompt": "Hi"}],
        results(),
        "r",
        0,
        "m",
        num_prompts=1,
        tokenizer=None,
        request_metadata=RequestResponseMetadata(request_id="r"),
    ):
        data = line.removeprefix("data: ").strip()
        if data == "[DONE]":
            continue
        for choice in json.loads(data)["choices"]:
            stream_offsets += choice["logprobs"]["text_offset"]

    response = serving.request_output_to_completion_response(
        [_output("".join(TEXT_DELTAS), [0, 1, 2], finished=True)],
        _request(stream=False),
        "r",
        0,
        "m",
        None,
        RequestResponseMetadata(request_id="r"),
    )
    assert response.choices[0].logprobs is not None
    assert response.choices[0].logprobs.text_offset == [0, 3, 6]
    assert stream_offsets == [0, 3, 6]
