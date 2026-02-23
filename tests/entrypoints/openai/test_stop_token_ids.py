# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Unit tests for stop_token_ids propagation from default_sampling_params
to SamplingParams in ChatCompletionRequest and CompletionRequest.

Regression test for https://github.com/vllm-project/vllm/issues/22519
where gpt-oss model stop tokens (e.g., </call> = 200012) were loaded into
default_sampling_params at server startup but silently discarded on every
request because to_sampling_params() never fell back to defaults.
"""

import pytest

from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
)
from vllm.entrypoints.openai.completion.protocol import (
    CompletionRequest,
)


class TestChatCompletionStopTokenIds:
    """Test stop_token_ids merging in ChatCompletionRequest.to_sampling_params()."""

    @pytest.fixture
    def minimal_chat_request(self):
        return ChatCompletionRequest(
            model="test-model",
            messages=[{"role": "user", "content": "hello"}],
        )

    def test_default_stop_token_ids_applied(self, minimal_chat_request):
        """Server-default stop_token_ids are applied when client sends none."""
        default_sampling_params = {
            "stop_token_ids": [200012, 200002],
        }

        sampling_params = minimal_chat_request.to_sampling_params(
            max_tokens=100,
            default_sampling_params=default_sampling_params,
        )

        assert set(sampling_params.stop_token_ids) == {200012, 200002}

    def test_client_stop_token_ids_merged_with_defaults(self):
        """Client-specified stop_token_ids are merged with server defaults."""
        request = ChatCompletionRequest(
            model="test-model",
            messages=[{"role": "user", "content": "hello"}],
            stop_token_ids=[99999],
        )
        default_sampling_params = {
            "stop_token_ids": [200012, 200002],
        }

        sampling_params = request.to_sampling_params(
            max_tokens=100,
            default_sampling_params=default_sampling_params,
        )

        assert set(sampling_params.stop_token_ids) == {200012, 200002, 99999}

    def test_no_stop_token_ids_anywhere(self, minimal_chat_request):
        """When neither client nor server specifies stop_token_ids, result is empty."""
        sampling_params = minimal_chat_request.to_sampling_params(
            max_tokens=100,
            default_sampling_params={},
        )

        assert not sampling_params.stop_token_ids

    def test_only_client_stop_token_ids(self):
        """Client stop_token_ids work when no server defaults exist."""
        request = ChatCompletionRequest(
            model="test-model",
            messages=[{"role": "user", "content": "hello"}],
            stop_token_ids=[42, 43],
        )

        sampling_params = request.to_sampling_params(
            max_tokens=100,
            default_sampling_params={},
        )

        assert set(sampling_params.stop_token_ids) == {42, 43}

    def test_duplicate_stop_token_ids_deduplicated(self):
        """Overlapping stop_token_ids between client and server are deduplicated."""
        request = ChatCompletionRequest(
            model="test-model",
            messages=[{"role": "user", "content": "hello"}],
            stop_token_ids=[200012, 55555],
        )
        default_sampling_params = {
            "stop_token_ids": [200012, 200002],
        }

        sampling_params = request.to_sampling_params(
            max_tokens=100,
            default_sampling_params=default_sampling_params,
        )

        assert set(sampling_params.stop_token_ids) == {200012, 200002, 55555}
        assert len(sampling_params.stop_token_ids) == 3


class TestCompletionStopTokenIds:
    """Test stop_token_ids merging in CompletionRequest.to_sampling_params()."""

    @pytest.fixture
    def minimal_completion_request(self):
        return CompletionRequest(
            model="test-model",
            prompt="hello",
        )

    def test_default_stop_token_ids_applied(self, minimal_completion_request):
        """Server-default stop_token_ids are applied when client sends none."""
        default_sampling_params = {
            "stop_token_ids": [200012, 200002],
        }

        sampling_params = minimal_completion_request.to_sampling_params(
            max_tokens=100,
            default_sampling_params=default_sampling_params,
        )

        assert set(sampling_params.stop_token_ids) == {200012, 200002}

    def test_client_stop_token_ids_merged_with_defaults(self):
        """Client-specified stop_token_ids are merged with server defaults."""
        request = CompletionRequest(
            model="test-model",
            prompt="hello",
            stop_token_ids=[99999],
        )
        default_sampling_params = {
            "stop_token_ids": [200012, 200002],
        }

        sampling_params = request.to_sampling_params(
            max_tokens=100,
            default_sampling_params=default_sampling_params,
        )

        assert set(sampling_params.stop_token_ids) == {200012, 200002, 99999}

    def test_no_stop_token_ids_anywhere(self, minimal_completion_request):
        """When neither client nor server specifies stop_token_ids, result is empty."""
        sampling_params = minimal_completion_request.to_sampling_params(
            max_tokens=100,
            default_sampling_params={},
        )

        assert not sampling_params.stop_token_ids
