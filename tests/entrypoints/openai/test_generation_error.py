# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.entrypoints.openai.engine.serving import GenerationError, OpenAIServing


def test_raise_if_error_uses_string_stop_reason():
    serving = OpenAIServing.__new__(OpenAIServing)

    with pytest.raises(GenerationError, match="KV cache load failed"):
        serving._raise_if_error(
            "error",
            "test-request",
            "KV cache load failed for one or more remote blocks. "
            "The request can be retried.",
        )


def test_raise_if_error_falls_back_for_missing_stop_reason():
    serving = OpenAIServing.__new__(OpenAIServing)

    with pytest.raises(GenerationError, match="Internal server error"):
        serving._raise_if_error("error", "test-request")


def test_raise_if_error_falls_back_for_blank_stop_reason():
    serving = OpenAIServing.__new__(OpenAIServing)

    with pytest.raises(GenerationError, match="Internal server error"):
        serving._raise_if_error("error", "test-request", "   ")
