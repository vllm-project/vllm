# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Contract tests for the shared ``X-Cache-Salt`` header resolver.

End-to-end coverage for the generate endpoints lives in
``tests/entrypoints/openai/chat_completion/test_serving_chat.py``; these cover
the resolver itself across the request families that accept ``cache_salt``.
"""

from unittest.mock import MagicMock

import pytest

from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.entrypoints.openai.completion.protocol import CompletionRequest
from vllm.entrypoints.pooling.embed.protocol import (
    CohereEmbedRequest,
    EmbeddingCompletionRequest,
)
from vllm.entrypoints.serve.engine.serving import (
    CACHE_SALT_HEADER,
    resolve_cache_salt_header,
)
from vllm.exceptions import VLLMValidationError

MODEL_NAME = "test-model"


def _raw_request(headers: dict[str, str] | None):
    raw_request = MagicMock()
    raw_request.headers = headers or {}
    return raw_request


def _requests():
    return [
        ChatCompletionRequest(
            model=MODEL_NAME, messages=[{"role": "user", "content": "hi"}]
        ),
        CompletionRequest(model=MODEL_NAME, prompt="hi"),
        EmbeddingCompletionRequest(model=MODEL_NAME, input="hi"),
    ]


@pytest.mark.parametrize("request_obj", _requests())
def test_header_sets_cache_salt(request_obj):
    resolve_cache_salt_header(
        request_obj, _raw_request({CACHE_SALT_HEADER: "tenant-42"})
    )
    assert request_obj.cache_salt == "tenant-42"


@pytest.mark.parametrize("request_obj", _requests())
def test_header_overrides_body(request_obj):
    request_obj.cache_salt = "from-body"
    resolve_cache_salt_header(
        request_obj, _raw_request({CACHE_SALT_HEADER: "from-header"})
    )
    assert request_obj.cache_salt == "from-header"


@pytest.mark.parametrize("request_obj", _requests())
def test_body_salt_survives_without_header(request_obj):
    request_obj.cache_salt = "from-body"
    resolve_cache_salt_header(request_obj, _raw_request({}))
    resolve_cache_salt_header(request_obj, None)
    assert request_obj.cache_salt == "from-body"


@pytest.mark.parametrize("bad_salt", ["", "with/slash", "with@at", "x" * 129])
def test_invalid_header_salt_is_rejected(bad_salt):
    request_obj = ChatCompletionRequest(
        model=MODEL_NAME, messages=[{"role": "user", "content": "hi"}]
    )
    with pytest.raises(VLLMValidationError, match="cache_salt"):
        resolve_cache_salt_header(
            request_obj, _raw_request({CACHE_SALT_HEADER: bad_salt})
        )
    assert request_obj.cache_salt is None


def test_header_on_unsupported_request_type_fails_closed():
    """CohereEmbedRequest has no cache_salt field. Honoring the header is
    impossible there, so the request is rejected rather than served from the
    unsalted cache."""
    request_obj = CohereEmbedRequest(model=MODEL_NAME, texts=["hi"])
    with pytest.raises(VLLMValidationError, match="does not support"):
        resolve_cache_salt_header(
            request_obj, _raw_request({CACHE_SALT_HEADER: "tenant-42"})
        )


def test_unsupported_request_type_unaffected_without_header():
    request_obj = CohereEmbedRequest(model=MODEL_NAME, texts=["hi"])
    resolve_cache_salt_header(request_obj, _raw_request({}))
    assert not hasattr(request_obj, "cache_salt")
