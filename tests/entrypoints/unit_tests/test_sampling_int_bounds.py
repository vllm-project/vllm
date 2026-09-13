# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Reject sampling integers that cannot round-trip through engine transport."""

import pytest
from pydantic import ValidationError

from vllm.entrypoints.anthropic.protocol import AnthropicMessagesRequest
from vllm.entrypoints.openai.chat_completion.protocol import (
    BatchChatCompletionRequest,
    ChatCompletionRequest,
)
from vllm.entrypoints.openai.completion.protocol import CompletionRequest
from vllm.entrypoints.openai.responses.protocol import ResponsesRequest

pytestmark = pytest.mark.cpu_test

_INT64_MAX = 2**63 - 1
_OVERFLOW = 2**64
_CHAT_MESSAGES = [{"role": "user", "content": "hi"}]


@pytest.mark.parametrize(
    ("cls", "kwargs"),
    [
        (CompletionRequest, {"model": "m", "prompt": "hi"}),
        (
            ChatCompletionRequest,
            {"model": "m", "messages": _CHAT_MESSAGES},
        ),
    ],
)
def test_oversized_stream_interval_rejected(cls, kwargs):
    with pytest.raises(ValidationError):
        cls(**kwargs, stream_interval=_OVERFLOW)


@pytest.mark.parametrize(
    ("cls", "kwargs"),
    [
        (CompletionRequest, {"model": "m", "prompt": "hi"}),
        (
            ChatCompletionRequest,
            {"model": "m", "messages": _CHAT_MESSAGES},
        ),
    ],
)
def test_max_stream_interval_accepted(cls, kwargs):
    req = cls(**kwargs, stream_interval=_INT64_MAX)
    assert req.stream_interval == _INT64_MAX


@pytest.mark.parametrize(
    ("cls", "kwargs"),
    [
        (CompletionRequest, {"model": "m", "prompt": "hi"}),
        (
            ChatCompletionRequest,
            {"model": "m", "messages": _CHAT_MESSAGES},
        ),
        (
            BatchChatCompletionRequest,
            {"model": "m", "messages": [_CHAT_MESSAGES]},
        ),
        (ResponsesRequest, {"model": "m", "input": "hi"}),
        (
            AnthropicMessagesRequest,
            {"model": "m", "max_tokens": 1, "messages": _CHAT_MESSAGES},
        ),
    ],
)
def test_oversized_top_k_rejected(cls, kwargs):
    with pytest.raises(ValidationError):
        cls(**kwargs, top_k=_OVERFLOW)


@pytest.mark.parametrize(
    ("cls", "kwargs"),
    [
        (CompletionRequest, {"model": "m", "prompt": "hi"}),
        (
            ChatCompletionRequest,
            {"model": "m", "messages": _CHAT_MESSAGES},
        ),
    ],
)
def test_negative_stream_interval_rejected(cls, kwargs):
    with pytest.raises(ValidationError):
        cls(**kwargs, stream_interval=-1)
