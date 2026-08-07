# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.entrypoints.openai.completion.protocol import CompletionRequest
from vllm.entrypoints.openai.responses.protocol import ResponsesRequest
from vllm.entrypoints.scale_out.token_in_token_out.protocol import GenerateRequest

pytestmark = pytest.mark.skip_global_cleanup


@pytest.mark.parametrize(
    ("request_type", "payload"),
    [
        (CompletionRequest, {"prompt": "hello"}),
        (
            ChatCompletionRequest,
            {"messages": [{"role": "user", "content": "hello"}]},
        ),
        (ResponsesRequest, {"input": "hello"}),
        (GenerateRequest, {"token_ids": [1], "sampling_params": {}}),
    ],
)
def test_public_requests_reject_lmcache_forbidden_cache_salt(
    request_type, payload: dict
) -> None:
    with pytest.raises(ValueError, match="cache_salt"):
        request_type.model_validate({**payload, "cache_salt": "/"})


def test_public_requests_reject_overlong_cache_salt() -> None:
    with pytest.raises(ValueError, match="cache_salt"):
        CompletionRequest.model_validate({"prompt": "hello", "cache_salt": "a" * 129})


def test_public_requests_accept_lmcache_safe_cache_salt() -> None:
    request = CompletionRequest.model_validate(
        {"prompt": "hello", "cache_salt": "safe-cache_salt-123"}
    )
    assert request.cache_salt == "safe-cache_salt-123"
