# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json

import pytest

from vllm.entrypoints.generate.base.protocol import RequestResponseMetadata
from vllm.entrypoints.openai.completion.protocol import CompletionRequest
from vllm.entrypoints.openai.completion.serving import OpenAIServingCompletion
from vllm.inputs import tokens_input
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
    serving.enable_log_outputs = False
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
        [tokens_input([9], prompt="Hi")],
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


def test_non_stream_response_omits_null_prompt_tokens_details():
    """The OpenAI spec types `usage.prompt_tokens_details` as an object, not
    nullable, so it is omitted when `--enable-prompt-tokens-details` is off."""
    response = _serving().request_output_to_completion_response(
        [_output("abc", [0], finished=True)],
        _request(stream=False),
        "r",
        0,
        "m",
        None,
        RequestResponseMetadata(request_id="r"),
    )

    assert "prompt_tokens_details" not in response.model_dump()["usage"]


PROMPT_IDS = [9, 1, 2, 3]


def _aborted_output(prompt_logprobs) -> RequestOutput:
    """A request aborted mid-prefill: prompt logprobs cover only the processed
    prompt tokens, or none of them, and nothing was generated."""
    return RequestOutput(
        request_id="r",
        prompt="Hi",
        prompt_token_ids=PROMPT_IDS,
        prompt_logprobs=prompt_logprobs,
        outputs=[
            CompletionOutput(
                index=0,
                text="",
                token_ids=[],
                cumulative_logprob=None,
                logprobs=[],
                finish_reason="abort",
            )
        ],
        finished=True,
    )


def _echo_request(stream: bool) -> CompletionRequest:
    return CompletionRequest(
        model="m",
        prompt=PROMPT_IDS,
        max_tokens=8,
        logprobs=0,
        echo=True,
        stream=stream,
        return_tokens_as_token_ids=True,
    )


@pytest.mark.parametrize("prompt_logprobs", [[None], [None, _logprobs(0)], None])
def test_echo_logprobs_of_request_aborted_mid_prefill(prompt_logprobs):
    response = _serving().request_output_to_completion_response(
        [_aborted_output(prompt_logprobs)],
        _echo_request(stream=False),
        "r",
        0,
        "m",
        None,
        RequestResponseMetadata(request_id="r"),
    )

    choice = response.choices[0]
    assert choice.finish_reason == "abort"
    assert choice.logprobs is not None
    assert len(choice.logprobs.tokens) == len(PROMPT_IDS)
    # Prompt positions the engine never reached have no logprob
    assert choice.logprobs.token_logprobs[2:] == [None, None]


@pytest.mark.asyncio
@pytest.mark.parametrize("prompt_logprobs", [[None], [None, _logprobs(0)], None])
async def test_stream_echo_logprobs_of_request_aborted_mid_prefill(prompt_logprobs):
    async def results():
        yield 0, _aborted_output(prompt_logprobs)

    tokens = []
    async for line in _serving().completion_stream_generator(
        _echo_request(stream=True),
        [tokens_input(PROMPT_IDS, prompt="Hi")],
        results(),
        "r",
        0,
        "m",
        num_prompts=1,
        tokenizer=None,
        request_metadata=RequestResponseMetadata(request_id="r"),
    ):
        data = line.removeprefix("data: ").strip()
        assert '"error"' not in data
        if data == "[DONE]":
            continue
        for choice in json.loads(data)["choices"]:
            if choice["logprobs"] is not None:
                tokens += choice["logprobs"]["tokens"]

    assert len(tokens) == len(PROMPT_IDS)


@pytest.mark.asyncio
@pytest.mark.parametrize("max_tokens", [0, 8])
async def test_stream_echo_without_logprobs_skips_prompt_padding(
    monkeypatch, max_tokens
):
    def pad(*args):
        raise AssertionError("prompt logprobs padded without logprobs requested")

    monkeypatch.setattr(
        "vllm.entrypoints.openai.completion.serving._padded_prompt_logprobs", pad
    )
    request = _echo_request(stream=True)
    request.logprobs = None
    request.max_tokens = max_tokens

    async def results():
        yield 0, _aborted_output(None)

    async for line in _serving().completion_stream_generator(
        request,
        [tokens_input(PROMPT_IDS, prompt="Hi")],
        results(),
        "r",
        0,
        "m",
        num_prompts=1,
        tokenizer=None,
        request_metadata=RequestResponseMetadata(request_id="r"),
    ):
        data = line.removeprefix("data: ").strip()
        assert '"error"' not in data
        if data != "[DONE]":
            assert all(c["logprobs"] is None for c in json.loads(data)["choices"])
