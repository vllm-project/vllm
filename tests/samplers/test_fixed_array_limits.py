# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm import SamplingParams
from vllm.exceptions import VLLMValidationError


@pytest.mark.parametrize(
    ("kwargs", "parameter"),
    [
        ({"allowed_token_ids": list(range(1025))}, "allowed_token_ids"),
        (
            {"logit_bias": {str(token_id): 1.0 for token_id in range(1025)}},
            "logit_bias",
        ),
        (
            {"min_tokens": 1, "stop_token_ids": list(range(129))},
            "stop_token_ids",
        ),
    ],
)
def test_rejects_sampler_inputs_wider_than_fixed_rows(
    kwargs: dict[str, object],
    parameter: str,
) -> None:
    with pytest.raises(VLLMValidationError, match=parameter):
        SamplingParams.from_optional(**kwargs)


def test_rejects_stop_tokens_wider_than_fixed_row_after_eos_merge() -> None:
    params = SamplingParams.from_optional(
        min_tokens=1,
        stop_token_ids=list(range(128)),
    )

    with pytest.raises(VLLMValidationError, match="stop_token_ids"):
        params.update_from_generation_config({}, eos_token_id=128)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"allowed_token_ids": list(range(1024))},
        {"logit_bias": {str(token_id): 1.0 for token_id in range(1024)}},
        {"min_tokens": 1, "stop_token_ids": list(range(128))},
    ],
)
def test_accepts_sampler_inputs_at_fixed_row_limits(
    kwargs: dict[str, object],
) -> None:
    SamplingParams.from_optional(**kwargs)


def test_accepts_duplicate_stop_tokens_within_effective_fixed_row() -> None:
    params = SamplingParams.from_optional(
        min_tokens=1,
        stop_token_ids=[7] * 129,
    )

    assert params.all_stop_token_ids == {7}
