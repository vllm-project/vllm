# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
)
from vllm.entrypoints.openai.completion.protocol import CompletionRequest
from vllm.entrypoints.openai.responses.protocol import ResponsesRequest


@pytest.mark.parametrize(
    "api_request",
    [
        ChatCompletionRequest(
            model="test-model",
            messages=[{"role": "user", "content": "hello"}],
            skip_writing_prefix_cache=True,
        ),
        CompletionRequest(
            model="test-model",
            prompt="hello",
            skip_writing_prefix_cache=True,
        ),
        ResponsesRequest(
            model="test-model",
            input="hello",
            skip_writing_prefix_cache=True,
        ),
    ],
)
def test_skip_writing_prefix_cache_reaches_sampling_params(api_request):
    if isinstance(api_request, (ChatCompletionRequest, CompletionRequest)):
        sampling_params = api_request.to_sampling_params(16, {})
    else:
        sampling_params = api_request.to_sampling_params(16)

    assert sampling_params.extra_args is not None
    assert sampling_params.extra_args["skip_writing_prefix_cache"] is True


@pytest.mark.parametrize(
    "api_request",
    [
        ChatCompletionRequest(
            model="test-model",
            messages=[{"role": "user", "content": "hello"}],
            use_beam_search=True,
            skip_writing_prefix_cache=True,
        ),
        CompletionRequest(
            model="test-model",
            prompt="hello",
            use_beam_search=True,
            skip_writing_prefix_cache=True,
        ),
    ],
)
def test_skip_writing_prefix_cache_reaches_beam_search_params(api_request):
    beam_search_params = api_request.to_beam_search_params(16, {})
    assert beam_search_params.skip_writing_prefix_cache is True
