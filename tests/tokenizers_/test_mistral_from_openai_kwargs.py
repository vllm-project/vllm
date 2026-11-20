# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the mistral-common kwargs filter in vllm.tokenizers.mistral."""

from vllm.tokenizers.mistral import _filter_from_openai_kwargs


def test_filter_from_openai_kwargs_drops_unknown():
    kwargs = {
        "cohere_format": "cmd4",
        "enable_thinking": True,
        "documents": [{"title": "doc"}],
        "temperature": 0.5,
        "max_tokens": 128,
        "seed": 42,
        "reasoning_effort": "low",
    }
    filtered = _filter_from_openai_kwargs(kwargs)
    # Not fields of mistral-common's ChatCompletionRequest: dropped.
    assert "cohere_format" not in filtered
    assert "enable_thinking" not in filtered
    assert "documents" not in filtered
    # Accepted fields (and the OpenAI seed alias): kept.
    assert filtered == {
        "temperature": 0.5,
        "max_tokens": 128,
        "seed": 42,
        "reasoning_effort": "low",
    }


def test_filter_from_openai_kwargs_empty():
    assert _filter_from_openai_kwargs({}) == {}
