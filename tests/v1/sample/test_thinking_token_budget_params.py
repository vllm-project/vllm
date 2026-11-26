# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.exceptions import VLLMValidationError
from vllm.sampling_params import SamplingParams, validate_thinking_token_budget


@pytest.mark.parametrize(
    ("raw_value", "expected"),
    [
        (None, None),
        (-1, None),
        (10, 10),
        (0, 0),
    ],
)
def test_validate_thinking_token_budget(raw_value, expected):
    assert validate_thinking_token_budget(raw_value) == expected


def test_sampling_params_minus_one_normalizes_to_none():
    params = SamplingParams(thinking_token_budget=-1)
    assert params.thinking_token_budget is None


@pytest.mark.parametrize("invalid_budget", [-2, 0.6, 10.5, True])
def test_validate_thinking_token_budget_rejects_invalid(invalid_budget):
    with pytest.raises(VLLMValidationError, match="thinking_token_budget"):
        validate_thinking_token_budget(invalid_budget)


@pytest.mark.parametrize("invalid_budget", [-2, 0.6, 10.5])
def test_thinking_budget_invalid_budget_rejected(invalid_budget):
    with pytest.raises(VLLMValidationError, match="thinking_token_budget"):
        SamplingParams(thinking_token_budget=invalid_budget)
