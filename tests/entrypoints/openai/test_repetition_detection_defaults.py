# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Unit tests for repetition_detection propagation from default_sampling_params
to SamplingParams in ChatCompletionRequest and CompletionRequest.

Operators can set repetition_detection server-side via the model generation
config or --override-generation-config to guard against degenerate repetition
loops; the serving layer must apply it when a request does not set its own.
"""

from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
)
from vllm.entrypoints.openai.completion.protocol import (
    CompletionRequest,
)
from vllm.sampling_params import RepetitionDetectionParams

SERVER_DEFAULT = RepetitionDetectionParams(max_pattern_size=8, min_count=4)
REQUEST_OVERRIDE = RepetitionDetectionParams(max_pattern_size=1, min_count=10)


class TestCompletionRepetitionDetection:
    """repetition_detection fallback in CompletionRequest.to_sampling_params()."""

    def test_server_default_applied(self):
        """Server-default repetition_detection applies when request sends none."""
        request = CompletionRequest(model="test-model", prompt="hello")

        sampling_params = request.to_sampling_params(
            max_tokens=100,
            default_sampling_params={"repetition_detection": SERVER_DEFAULT},
        )

        assert sampling_params.repetition_detection == SERVER_DEFAULT

    def test_request_value_wins(self):
        """Request-specified repetition_detection overrides the server default."""
        request = CompletionRequest(
            model="test-model",
            prompt="hello",
            repetition_detection=REQUEST_OVERRIDE,
        )

        sampling_params = request.to_sampling_params(
            max_tokens=100,
            default_sampling_params={"repetition_detection": SERVER_DEFAULT},
        )

        assert sampling_params.repetition_detection == REQUEST_OVERRIDE

    def test_no_repetition_detection_anywhere(self):
        """With no server default and no request value, it stays disabled."""
        request = CompletionRequest(model="test-model", prompt="hello")

        sampling_params = request.to_sampling_params(
            max_tokens=100,
            default_sampling_params={},
        )

        assert sampling_params.repetition_detection is None


class TestChatCompletionRepetitionDetection:
    """repetition_detection fallback in ChatCompletionRequest.to_sampling_params()."""

    def test_server_default_applied(self):
        """Server-default repetition_detection applies when request sends none."""
        request = ChatCompletionRequest(
            model="test-model",
            messages=[{"role": "user", "content": "hello"}],
        )

        sampling_params = request.to_sampling_params(
            max_tokens=100,
            default_sampling_params={"repetition_detection": SERVER_DEFAULT},
        )

        assert sampling_params.repetition_detection == SERVER_DEFAULT

    def test_request_value_wins(self):
        """Request-specified repetition_detection overrides the server default."""
        request = ChatCompletionRequest(
            model="test-model",
            messages=[{"role": "user", "content": "hello"}],
            repetition_detection=REQUEST_OVERRIDE,
        )

        sampling_params = request.to_sampling_params(
            max_tokens=100,
            default_sampling_params={"repetition_detection": SERVER_DEFAULT},
        )

        assert sampling_params.repetition_detection == REQUEST_OVERRIDE

    def test_no_repetition_detection_anywhere(self):
        """With no server default and no request value, it stays disabled."""
        request = ChatCompletionRequest(
            model="test-model",
            messages=[{"role": "user", "content": "hello"}],
        )

        sampling_params = request.to_sampling_params(
            max_tokens=100,
            default_sampling_params={},
        )

        assert sampling_params.repetition_detection is None
