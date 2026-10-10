# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for ``--fingerprint-mode=none`` response serialization.

Regression tests for https://github.com/vllm-project/vllm/issues/57376:
with ``--fingerprint-mode=none`` the non-streaming ``/v1/chat/completions``
and ``/v1/completions`` responses must omit the ``system_fingerprint`` key
entirely instead of emitting an explicit ``null``. Other fingerprint modes
keep the full payload unchanged (see #53349).
"""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from vllm.entrypoints.openai.chat_completion.api_router import (
    create_chat_completion,
)
from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionResponseChoice,
    ChatMessage,
)
from vllm.entrypoints.openai.completion.api_router import create_completion
from vllm.entrypoints.openai.completion.protocol import (
    CompletionRequest,
    CompletionResponse,
    CompletionResponseChoice,
)
from vllm.entrypoints.serve.engine.protocol import UsageInfo


def _raw_request(handler_attr: str, handler) -> SimpleNamespace:
    """Minimal stand-in for fastapi's ``Request`` usable by the endpoint
    decorators: ``load_aware_call`` reads ``app.state`` and
    ``with_cancellation`` listens on ``request.stream()``."""

    async def _never_disconnects():
        await asyncio.Event().wait()
        yield  # pragma: no cover

    raw_request = SimpleNamespace()
    raw_request.headers = {}
    raw_request.stream = _never_disconnects
    raw_request.app = SimpleNamespace(
        state=SimpleNamespace(
            **{
                handler_attr: handler,
                "args": None,
            }
        )
    )
    return raw_request


def _make_handler(method_name: str, system_fingerprint, response):
    handler = SimpleNamespace()
    handler.system_fingerprint = system_fingerprint
    setattr(handler, method_name, AsyncMock(return_value=response))
    return handler


def _body(response) -> dict:
    return json.loads(response.body)


def _chat_response(system_fingerprint):
    return ChatCompletionResponse(
        model="test-model",
        choices=[
            ChatCompletionResponseChoice(
                index=0,
                message=ChatMessage(role="assistant", content="Hi"),
                finish_reason="stop",
            )
        ],
        usage=UsageInfo(),
        system_fingerprint=system_fingerprint,
    )


def _completion_response(system_fingerprint):
    return CompletionResponse(
        model="test-model",
        choices=[
            CompletionResponseChoice(
                index=0,
                text="Hi",
                finish_reason="stop",
            )
        ],
        usage=UsageInfo(),
        system_fingerprint=system_fingerprint,
    )


@pytest.mark.asyncio
async def test_chat_none_mode_omits_system_fingerprint():
    handler = _make_handler(
        "create_chat_completion", None, _chat_response(None)
    )
    raw_request = _raw_request("openai_serving_chat", handler)

    result = await create_chat_completion(
        ChatCompletionRequest(
            model="test-model", messages=[{"role": "user", "content": "Hi"}]
        ),
        raw_request,
    )

    assert "system_fingerprint" not in _body(result)


@pytest.mark.asyncio
async def test_chat_full_mode_keeps_system_fingerprint():
    fingerprint = "vllm-0.10.1-deadbeef"
    handler = _make_handler(
        "create_chat_completion", fingerprint, _chat_response(fingerprint)
    )
    raw_request = _raw_request("openai_serving_chat", handler)

    result = await create_chat_completion(
        ChatCompletionRequest(
            model="test-model", messages=[{"role": "user", "content": "Hi"}]
        ),
        raw_request,
    )

    body = _body(result)
    assert body["system_fingerprint"] == fingerprint
    # Other unset fields keep their explicit ``null`` outside ``none`` mode.
    assert body["prompt_token_ids"] is None


@pytest.mark.asyncio
async def test_completion_none_mode_omits_system_fingerprint():
    handler = _make_handler(
        "create_completion", None, _completion_response(None)
    )
    raw_request = _raw_request("openai_serving_completion", handler)

    result = await create_completion(
        CompletionRequest(model="test-model", prompt="Hi"),
        raw_request,
    )

    assert "system_fingerprint" not in _body(result)


@pytest.mark.asyncio
async def test_completion_full_mode_keeps_system_fingerprint():
    fingerprint = "vllm-0.10.1-deadbeef"
    handler = _make_handler(
        "create_completion", fingerprint, _completion_response(fingerprint)
    )
    raw_request = _raw_request("openai_serving_completion", handler)

    result = await create_completion(
        CompletionRequest(model="test-model", prompt="Hi"),
        raw_request,
    )

    body = _body(result)
    assert body["system_fingerprint"] == fingerprint
    # Other unset fields keep their explicit ``null`` outside ``none`` mode.
    assert body["metrics"] is None
