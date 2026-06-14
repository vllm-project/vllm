# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.entrypoints.openai.completion.protocol import CompletionRequest

pytestmark = pytest.mark.skip_global_cleanup


def _make_chat_request(stop_token_ids: list[int] | None = None):
    return ChatCompletionRequest(
        model="test-model",
        messages=[{"role": "user", "content": "Hello"}],
        stop_token_ids=stop_token_ids,
    )


def _make_completion_request(stop_token_ids: list[int] | None = None):
    return CompletionRequest(
        model="test-model",
        prompt="Hello",
        stop_token_ids=stop_token_ids,
    )


@pytest.mark.parametrize(
    "request_factory",
    [_make_chat_request, _make_completion_request],
)
def test_default_stop_token_ids_are_preserved(request_factory):
    request = request_factory()

    sampling_params = request.to_sampling_params(
        max_tokens=16,
        default_sampling_params={"stop_token_ids": [101, 102]},
    )

    assert sampling_params.stop_token_ids == [101, 102]


@pytest.mark.parametrize(
    "request_factory",
    [_make_chat_request, _make_completion_request],
)
def test_request_stop_token_ids_are_preserved_without_defaults(request_factory):
    request = request_factory(stop_token_ids=[201, 202])

    sampling_params = request.to_sampling_params(
        max_tokens=16,
        default_sampling_params={},
    )

    assert sampling_params.stop_token_ids == [201, 202]


@pytest.mark.parametrize(
    "request_factory",
    [_make_chat_request, _make_completion_request],
)
def test_default_and_request_stop_token_ids_are_merged(request_factory):
    request = request_factory(stop_token_ids=[202, 301])
    default_sampling_params = {"stop_token_ids": [101, 202]}

    sampling_params = request.to_sampling_params(
        max_tokens=16,
        default_sampling_params=default_sampling_params,
    )

    assert sampling_params.stop_token_ids == [101, 202, 301]
    assert default_sampling_params == {"stop_token_ids": [101, 202]}
