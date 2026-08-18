# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.exceptions import VLLMValidationError
from vllm.sampling_params import (
    PostThinkingParams,
    SamplingParams,
    tokens_in_reasoning,
)


def test_resolve_sampling_knobs_inherits_unset_overlay_fields():
    params = SamplingParams(
        temperature=1.0,
        top_p=1.0,
        top_k=0,
        min_p=0.0,
        post_thinking=PostThinkingParams(temperature=0.4),
    )
    thinking = params.resolve_sampling_knobs(in_reasoning=True)
    assert thinking.temperature == 1.0
    assert thinking.top_p == 1.0
    assert thinking.top_k == 0

    answer = params.resolve_sampling_knobs(in_reasoning=False)
    assert answer.temperature == 0.4
    assert answer.top_p == 1.0
    assert answer.top_k == 0


def test_resolve_sampling_knobs_without_overlay_is_primary():
    params = SamplingParams(temperature=0.7, top_p=0.9, top_k=20)
    knobs = params.resolve_sampling_knobs(in_reasoning=False)
    assert knobs.temperature == 0.7
    assert knobs.top_p == 0.9
    assert knobs.top_k == 20


def test_post_thinking_from_optional_dict():
    params = SamplingParams.from_optional(
        temperature=1.0, post_thinking={"temperature": 0.4, "top_k": 20}
    )
    assert params.post_thinking is not None
    assert params.post_thinking.temperature == 0.4
    assert params.post_thinking.top_k == 20


def test_post_thinking_rejects_invalid_temperature():
    with pytest.raises(VLLMValidationError, match="temperature"):
        PostThinkingParams(temperature=-0.1)


def test_tokens_in_reasoning_uses_last_start_vs_last_end():
    start, end = [10], [11]
    assert tokens_in_reasoning([1, 10, 2, 3], start, end)
    assert not tokens_in_reasoning([1, 10, 2, 11, 3], start, end)
    assert tokens_in_reasoning([1, 10, 2, 11, 10, 4], start, end)
    assert not tokens_in_reasoning([1, 2, 3], start, end)
    assert not tokens_in_reasoning([], start, end)
