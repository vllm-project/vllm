# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Callable
from typing import Protocol

import pytest
from pydantic import ValidationError

from vllm.entrypoints.openai.chat_completion.protocol import (
    BatchChatCompletionRequest,
    ChatCompletionRequest,
)
from vllm.entrypoints.openai.completion.protocol import CompletionRequest
from vllm.entrypoints.openai.responses.protocol import ResponsesRequest


class _StopRequest(Protocol):
    stop: str | list[str] | None


def _completion_request(stop: list[str]) -> _StopRequest:
    return CompletionRequest(model="test-model", prompt="hello", stop=stop)


def _chat_request(stop: list[str]) -> _StopRequest:
    return ChatCompletionRequest(
        model="test-model",
        messages=[{"role": "user", "content": "hello"}],
        stop=stop,
    )


def _batch_chat_request(stop: list[str]) -> _StopRequest:
    return BatchChatCompletionRequest(
        model="test-model",
        messages=[[{"role": "user", "content": "hello"}]],
        stop=stop,
    )


def _responses_request(stop: list[str]) -> _StopRequest:
    return ResponsesRequest(model="test-model", input="hello", stop=stop)


REQUEST_BUILDERS: list[Callable[[list[str]], _StopRequest]] = [
    _completion_request,
    _chat_request,
    _batch_chat_request,
    _responses_request,
]


@pytest.mark.parametrize("build_request", REQUEST_BUILDERS)
def test_public_requests_accept_four_stop_strings(
    build_request: Callable[[list[str]], _StopRequest],
):
    stop = ["one", "two", "three", "four"]

    request = build_request(stop)

    assert request.stop == stop


@pytest.mark.parametrize("build_request", REQUEST_BUILDERS)
def test_public_requests_reject_more_than_four_stop_strings(
    build_request: Callable[[list[str]], _StopRequest],
):
    with pytest.raises(ValidationError, match="at most 4"):
        build_request(["one", "two", "three", "four", "five"])
