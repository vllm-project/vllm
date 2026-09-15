# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from io import BytesIO

import pytest
from fastapi import UploadFile

from vllm.entrypoints.openai.chat_completion.protocol import (
    BatchChatCompletionRequest,
    ChatCompletionRequest,
)
from vllm.entrypoints.openai.completion.protocol import CompletionRequest
from vllm.entrypoints.openai.responses.protocol import ResponsesRequest
from vllm.entrypoints.speech_to_text.transcription.protocol import (
    TranscriptionRequest,
)
from vllm.entrypoints.speech_to_text.translation.protocol import TranslationRequest
from vllm.exceptions import VLLMValidationError
from vllm.sampling_params import StructuredOutputsParams


def test_chat_request_preserves_watermarking_opt_out():
    request = ChatCompletionRequest(
        messages=[{"role": "user", "content": "hello"}], watermarking=False
    )

    params = request.to_sampling_params(max_tokens=1, default_sampling_params={})

    assert not params.watermarking


def test_completion_request_preserves_watermarking_opt_out():
    request = CompletionRequest(prompt="hello", watermarking=False)

    params = request.to_sampling_params(max_tokens=1)

    assert not params.watermarking


def test_responses_request_preserves_watermarking_opt_out():
    request = ResponsesRequest(input="hello", watermarking=False)

    params = request.to_sampling_params(default_max_tokens=1)

    assert not params.watermarking


def test_batch_chat_request_preserves_watermarking_opt_out():
    request = BatchChatCompletionRequest(
        messages=[[{"role": "user", "content": "hello"}]], watermarking=False
    )

    converted = request.to_chat_completion_request(request.messages[0])
    params = converted.to_sampling_params(max_tokens=1, default_sampling_params={})

    assert not params.watermarking


@pytest.mark.parametrize(
    "request_cls,kwargs",
    [
        (ChatCompletionRequest, {"messages": [{"role": "user", "content": "hi"}]}),
        (CompletionRequest, {"prompt": "hi"}),
        (
            BatchChatCompletionRequest,
            {"messages": [[{"role": "user", "content": "hi"}]]},
        ),
    ],
)
@pytest.mark.parametrize("watermarking", [True, False])
def test_best_of_is_rejected(request_cls, kwargs, watermarking):
    with pytest.raises(VLLMValidationError, match="best_of.*not supported"):
        request_cls(**kwargs, best_of=2, watermarking=watermarking)


@pytest.mark.parametrize(
    "api_request,conversion,kwargs",
    [
        (
            ChatCompletionRequest(
                messages=[{"role": "user", "content": "hello"}],
                use_beam_search=True,
                watermarking=False,
            ),
            "to_beam_search_params",
            {"max_tokens": 1, "default_sampling_params": {}},
        ),
        (
            CompletionRequest(prompt="hello", use_beam_search=True, watermarking=False),
            "to_beam_search_params",
            {"max_tokens": 1, "default_sampling_params": {}},
        ),
        (
            TranscriptionRequest(
                file=UploadFile(file=BytesIO(), filename="audio.wav"),
                watermarking=False,
            ),
            "to_sampling_params",
            {"default_max_tokens": 1},
        ),
        (
            TranscriptionRequest(
                file=UploadFile(file=BytesIO(), filename="audio.wav"),
                watermarking=False,
            ),
            "to_beam_search_params",
            {"default_max_tokens": 1},
        ),
        (
            TranslationRequest(
                file=UploadFile(file=BytesIO(), filename="audio.wav"),
                watermarking=False,
            ),
            "to_sampling_params",
            {"default_max_tokens": 1},
        ),
        (
            TranslationRequest(
                file=UploadFile(file=BytesIO(), filename="audio.wav"),
                watermarking=False,
            ),
            "to_beam_search_params",
            {"default_max_tokens": 1},
        ),
    ],
)
def test_request_preserves_watermarking_opt_out(api_request, conversion, kwargs):
    params = getattr(api_request, conversion)(**kwargs)

    assert not params.watermarking


@pytest.mark.parametrize(
    "api_request,kwargs",
    [
        (
            ChatCompletionRequest(
                messages=[{"role": "user", "content": "choose A or B"}],
                temperature=0.8,
                structured_outputs=StructuredOutputsParams(choice=["A", "B"]),
            ),
            {"max_tokens": 8, "default_sampling_params": {}},
        ),
        (
            CompletionRequest(
                prompt="choose A or B",
                temperature=0.8,
                structured_outputs=StructuredOutputsParams(choice=["A", "B"]),
            ),
            {"max_tokens": 8},
        ),
        (
            ResponsesRequest(
                input="choose A or B",
                temperature=0.8,
                structured_outputs=StructuredOutputsParams(choice=["A", "B"]),
            ),
            {"default_max_tokens": 8},
        ),
    ],
)
def test_request_preserves_watermarking_with_structured_outputs(api_request, kwargs):
    params = api_request.to_sampling_params(**kwargs)

    assert params.watermarking
    assert params.structured_outputs is not None
    assert params.structured_outputs.choice == ["A", "B"]
