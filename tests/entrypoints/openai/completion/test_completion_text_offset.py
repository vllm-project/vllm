# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import math

import numpy as np
import pytest

from vllm.entrypoints.generate.base.protocol import RequestResponseMetadata
from vllm.entrypoints.openai.completion.protocol import (
    CompletionLogProbs,
    CompletionRequest,
    CompletionResponseStreamChoice,
    CompletionStreamResponse,
)
from vllm.entrypoints.openai.completion.serving import (
    OpenAIServingCompletion,
    _logprobs_frame_templates_match,
    _LogprobsFrameTemplate,
)
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
    """The OpenAI spec types ``usage.prompt_tokens_details`` as an object, not
    nullable, so it is omitted when ``--enable-prompt-tokens-details`` is off."""
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


@pytest.mark.parametrize(
    "text", [' "q" \\ back\nslash\t', "\x00\x1f\x7f\x85", "café 中文 🚀", ""]
)
def test_logprobs_frame_template_matches_response_model(text):
    """Template bytes retain model field ordering, escaping and float encoding."""
    fields = (
        [0, 3],
        [-0.0, -math.inf],
        [text, "é"],
        [{text: math.nan, "é": -1e-45}, None],
    )
    template = _LogprobsFrameTemplate.build("cmpl-test", 123, "model", 2)
    assert template is not None
    chunk = CompletionStreamResponse(
        id="cmpl-test",
        object="text_completion",
        created=123,
        model="model",
        choices=[
            CompletionResponseStreamChoice(
                index=2,
                text=text,
                logprobs=CompletionLogProbs(
                    text_offset=fields[0],
                    token_logprobs=fields[1],
                    tokens=fields[2],
                    top_logprobs=fields[3],
                ),
                finish_reason=None,
                stop_reason=None,
                prompt_token_ids=None,
                token_ids=None,
                weight_versions=None,
            )
        ],
        kv_transfer_params=None,
    )
    assert (
        template.render(text, fields)
        == f"data: {chunk.model_dump_json(exclude_unset=True)}\n\n"
    )


@pytest.mark.parametrize(
    "text,value", [("lone \ud800", -0.5), ("token", np.float32(-0.5))]
)
def test_logprobs_frame_template_rejects_values_requiring_model_validation(text, value):
    template = _LogprobsFrameTemplate.build("cmpl-test", 123, "model", 0)
    assert template is not None
    assert template.render(text, ([0], [value], ["token"], [None])) is None


def test_logprobs_frame_templates_match_installed_pydantic():
    assert _logprobs_frame_templates_match()
