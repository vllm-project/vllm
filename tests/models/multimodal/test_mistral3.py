# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest

from vllm.model_executor.models.mistral3 import Mistral3ForConditionalGeneration


def _make_mistral3_model(
    spatial_merge_size: int,
) -> Mistral3ForConditionalGeneration:
    model = object.__new__(Mistral3ForConditionalGeneration)
    object.__setattr__(
        model, "config", SimpleNamespace(spatial_merge_size=spatial_merge_size)
    )
    return model


@pytest.mark.parametrize(
    ("spatial_merge_size", "num_image_tokens"),
    [
        (1, 16),
        (2, 16),
        (4, 8),
    ],
)
def test_mistral3_mm_lora_token_counts(
    spatial_merge_size: int, num_image_tokens: int
) -> None:
    model = _make_mistral3_model(spatial_merge_size)

    num_vision_tokens = num_image_tokens * (spatial_merge_size**2)

    assert model.get_num_mm_encoder_tokens(num_image_tokens) == num_vision_tokens
    assert model.get_num_mm_connector_tokens(num_vision_tokens) == num_image_tokens
