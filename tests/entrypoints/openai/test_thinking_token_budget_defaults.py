# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Unit tests for thinking_token_budget propagation from default_sampling_params
to SamplingParams in ChatCompletionRequest and CompletionRequest.

Same bug class as https://github.com/vllm-project/vllm/issues/22519 (fixed for
stop_token_ids in tests/entrypoints/openai/test_stop_token_ids.py): a value set
at server startup via --override-generation-config landed in
default_sampling_params but was silently discarded on every request, because
to_sampling_params() passed self.thinking_token_budget straight through instead
of falling back to the defaults.
"""

import pytest

from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
)
from vllm.entrypoints.openai.completion.protocol import (
    CompletionRequest,
)


class TestChatCompletionThinkingTokenBudget:
    """thinking_token_budget defaulting in ChatCompletionRequest."""

    @pytest.fixture
    def minimal_chat_request(self):
        return ChatCompletionRequest(
            model="test-model",
            messages=[{"role": "user", "content": "hello"}],
        )

    def test_default_thinking_token_budget_applied(self, minimal_chat_request):
        """Server-default thinking_token_budget is applied when client sends none."""
        sampling_params = minimal_chat_request.to_sampling_params(
            max_tokens=100,
            default_sampling_params={"thinking_token_budget": 128},
        )

        assert sampling_params.thinking_token_budget == 128

    def test_client_thinking_token_budget_overrides_default(self):
        """An explicit per-request value wins over the server default."""
        request = ChatCompletionRequest(
            model="test-model",
            messages=[{"role": "user", "content": "hello"}],
            thinking_token_budget=1024,
        )

        sampling_params = request.to_sampling_params(
            max_tokens=100,
            default_sampling_params={"thinking_token_budget": 128},
        )

        assert sampling_params.thinking_token_budget == 1024

    def test_no_default_and_no_request_value_stays_none(self, minimal_chat_request):
        """Absent both, the field stays None (no budget enforced)."""
        sampling_params = minimal_chat_request.to_sampling_params(
            max_tokens=100,
            default_sampling_params={},
        )

        assert sampling_params.thinking_token_budget is None

    def test_client_zero_budget_not_swallowed_by_default(self):
        """0 is a meaningful budget and must not fall through to the server default."""
        request = ChatCompletionRequest(
            model="test-model",
            messages=[{"role": "user", "content": "hello"}],
            thinking_token_budget=0,
        )

        sampling_params = request.to_sampling_params(
            max_tokens=100,
            default_sampling_params={"thinking_token_budget": 128},
        )

        assert sampling_params.thinking_token_budget == 0


class TestCompletionThinkingTokenBudget:
    """thinking_token_budget defaulting in CompletionRequest."""

    @pytest.fixture
    def minimal_completion_request(self):
        return CompletionRequest(
            model="test-model",
            prompt="hello",
        )

    def test_default_thinking_token_budget_applied(self, minimal_completion_request):
        """Server-default thinking_token_budget is applied when client sends none."""
        sampling_params = minimal_completion_request.to_sampling_params(
            max_tokens=100,
            default_sampling_params={"thinking_token_budget": 128},
        )

        assert sampling_params.thinking_token_budget == 128

    def test_client_thinking_token_budget_overrides_default(self):
        """An explicit per-request value wins over the server default."""
        request = CompletionRequest(
            model="test-model",
            prompt="hello",
            thinking_token_budget=1024,
        )

        sampling_params = request.to_sampling_params(
            max_tokens=100,
            default_sampling_params={"thinking_token_budget": 128},
        )

        assert sampling_params.thinking_token_budget == 1024

    def test_no_default_and_no_request_value_stays_none(
        self, minimal_completion_request
    ):
        """Absent both, the field stays None (no budget enforced)."""
        sampling_params = minimal_completion_request.to_sampling_params(
            max_tokens=100,
            default_sampling_params={},
        )

        assert sampling_params.thinking_token_budget is None

    def test_client_zero_budget_not_swallowed_by_default(self):
        """0 is a meaningful budget and must not fall through to the server default."""
        request = CompletionRequest(
            model="test-model",
            prompt="hello",
            thinking_token_budget=0,
        )

        sampling_params = request.to_sampling_params(
            max_tokens=100,
            default_sampling_params={"thinking_token_budget": 128},
        )

        assert sampling_params.thinking_token_budget == 0
