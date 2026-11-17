# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
from starlette.requests import Request

from vllm.entrypoints.generate.base.serving import GenerateBaseServing
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.entrypoints.openai.completion.protocol import CompletionRequest
from vllm.entrypoints.openai.responses.protocol import ResponsesRequest


def _raw_request(headers: dict[str, str]) -> Request:
    return Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/v1/chat/completions",
            "headers": [
                (key.lower().encode("latin-1"), value.encode("latin-1"))
                for key, value in headers.items()
            ],
        }
    )


@pytest.mark.parametrize(
    "openai_request",
    [
        ChatCompletionRequest(
            model="test-model",
            messages=[{"role": "user", "content": "hi"}],
        ),
        CompletionRequest(model="test-model", prompt="hi"),
        ResponsesRequest(model="test-model", input="hi"),
    ],
)
def test_get_session_id_accepts_body_field(openai_request):
    openai_request.session_id = "body-session"

    session_id = GenerateBaseServing._get_session_id(
        openai_request,
        _raw_request({"X-Session-ID": "header-session"}),
    )

    assert session_id == "body-session"


def test_get_session_id_accepts_session_header():
    request = CompletionRequest(model="test-model", prompt="hi")

    session_id = GenerateBaseServing._get_session_id(
        request,
        _raw_request({"X-Session-ID": "header-session"}),
    )

    assert session_id == "header-session"


def test_get_session_id_ignores_correlation_header():
    request = CompletionRequest(
        model="test-model",
        prompt="hi",
        vllm_xargs={"session_id": "xargs-session"},
    )

    session_id = GenerateBaseServing._get_session_id(
        request,
        _raw_request({"X-Correlation-ID": "correlation-session"}),
    )

    assert session_id == "xargs-session"


def test_get_session_id_keeps_vllm_xargs_as_compatibility_fallback():
    request = CompletionRequest(
        model="test-model",
        prompt="hi",
        vllm_xargs={"session_id": "xargs-session"},
    )

    session_id = GenerateBaseServing._get_session_id(request, None)

    assert session_id == "xargs-session"


def test_get_session_id_ignores_empty_and_non_string_values():
    request = CompletionRequest(
        model="test-model",
        prompt="hi",
        session_id="",
        vllm_xargs={"session_id": 7},
    )

    session_id = GenerateBaseServing._get_session_id(request, None)

    assert session_id is None
