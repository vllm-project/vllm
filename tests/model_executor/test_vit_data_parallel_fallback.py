# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import patch

import pytest

from vllm.model_executor.models.vision import is_vit_use_data_parallel


@pytest.mark.parametrize(
    "tp_size,num_heads,expected",
    [
        # Non-divisible head counts should fall back to data parallelism.
        (3, 16, True),
        (5, 16, True),
        (7, 16, True),
        # Divisible head counts should keep tensor parallelism.
        (1, 16, False),
        (2, 16, False),
        (4, 16, False),
        (8, 16, False),
        (3, 12, False),
        (3, 15, False),
    ],
)
def test_is_vit_use_data_parallel_chooses_fallback(
    tp_size: int, num_heads: int, expected: bool
) -> None:
    with patch(
        "vllm.model_executor.models.vision.get_tensor_model_parallel_world_size",
        return_value=tp_size,
    ):
        assert is_vit_use_data_parallel(num_heads) is expected


def test_is_vit_use_data_parallel_no_heads_uses_config() -> None:
    # When num_heads is not provided, the helper should fall back to the
    # mm_encoder_tp_mode config value. With the default config this is False.
    with patch(
        "vllm.model_executor.models.vision.get_tensor_model_parallel_world_size",
        return_value=3,
    ):
        assert is_vit_use_data_parallel() is False
