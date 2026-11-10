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
from vllm.sampling_params import StructuredOutputsParams


def test_chat_request_preserves_watermarking_opt_out():
    request = ChatCompletionRequest(
        messages=[{"role": "user", "content": "hello"}], watermarking=False
    )

    params = request.to_sampling_params(max_tokens=1, default_sampling_params={})

    assert params.watermarking is False


def test_completion_request_preserves_watermarking_opt_out():
    request = CompletionRequest(prompt="hello", watermarking=False)

    params = request.to_sampling_params(max_tokens=1)

    assert params.watermarking is False


def test_responses_request_preserves_watermarking_opt_out():
    request = ResponsesRequest(input="hello", watermarking=False)

    params = request.to_sampling_params(default_max_tokens=1)

    assert params.watermarking is False


def test_batch_chat_request_preserves_watermarking_opt_out():
    request = BatchChatCompletionRequest(
        messages=[[{"role": "user", "content": "hello"}]], watermarking=False
    )

    converted = request.to_chat_completion_request(request.messages[0])
    params = converted.to_sampling_params(max_tokens=1, default_sampling_params={})

    assert params.watermarking is False


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

    assert params.watermarking is False


@pytest.mark.parametrize("request_cls", [TranscriptionRequest, TranslationRequest])
def test_speech_requests_leave_watermarking_unspecified(request_cls):
    request = request_cls(
        file=UploadFile(file=BytesIO(), filename="audio.wav"),
        temperature=0.8,
    )

    sampling_params = request.to_sampling_params(default_max_tokens=1)
    beam_params = request.to_beam_search_params(default_max_tokens=1)

    assert request.watermarking is None
    assert sampling_params.watermarking is None
    assert beam_params.watermarking is None


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
def test_request_preserves_unspecified_watermarking_with_structured_outputs(
    api_request, kwargs
):
    params = api_request.to_sampling_params(**kwargs)

    assert params.watermarking is None
    assert params.structured_outputs is not None
    assert params.structured_outputs.choice == ["A", "B"]


@pytest.mark.parametrize(
    "api_request,conversion,kwargs",
    [
        (
            ChatCompletionRequest(
                messages=[{"role": "user", "content": "hello"}], watermarking=True
            ),
            "to_sampling_params",
            {"max_tokens": 1, "default_sampling_params": {}},
        ),
        (
            ChatCompletionRequest(
                messages=[{"role": "user", "content": "hello"}],
                use_beam_search=True,
                watermarking=True,
            ),
            "to_beam_search_params",
            {"max_tokens": 1, "default_sampling_params": {}},
        ),
        (
            CompletionRequest(prompt="hello", watermarking=True),
            "to_sampling_params",
            {"max_tokens": 1},
        ),
        (
            CompletionRequest(prompt="hello", use_beam_search=True, watermarking=True),
            "to_beam_search_params",
            {"max_tokens": 1},
        ),
        (
            ResponsesRequest(input="hello", watermarking=True),
            "to_sampling_params",
            {"default_max_tokens": 1},
        ),
        (
            TranscriptionRequest(
                file=UploadFile(file=BytesIO(), filename="audio.wav"),
                watermarking=True,
            ),
            "to_sampling_params",
            {"default_max_tokens": 1},
        ),
        (
            TranslationRequest(
                file=UploadFile(file=BytesIO(), filename="audio.wav"),
                watermarking=True,
            ),
            "to_sampling_params",
            {"default_max_tokens": 1},
        ),
    ],
)
def test_request_preserves_explicit_watermarking(api_request, conversion, kwargs):
    params = getattr(api_request, conversion)(**kwargs)

    assert params.watermarking is True


def test_batch_chat_request_preserves_explicit_watermarking():
    request = BatchChatCompletionRequest(
        messages=[[{"role": "user", "content": "hello"}]], watermarking=True
    )

    converted = request.to_chat_completion_request(request.messages[0])
    params = converted.to_sampling_params(max_tokens=1, default_sampling_params={})

    assert converted.watermarking is True
    assert params.watermarking is True


def test_batch_chat_request_inherits_watermarking():
    request = BatchChatCompletionRequest(
        messages=[[{"role": "user", "content": "hello"}]]
    )

    converted = request.to_chat_completion_request(request.messages[0])
    params = converted.to_sampling_params(max_tokens=1, default_sampling_params={})

    assert converted.watermarking is None
    assert params.watermarking is None
