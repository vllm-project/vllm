# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.model_executor.models.gemma4_mm import (
    get_gemma4_pooled_token_counts,
)


def _rect_positions(width: int, height: int) -> list[list[int]]:
    return [[x, y] for y in range(height) for x in range(width)]


def test_get_pooled_token_counts_without_pooling() -> None:
    pixel_position_ids = torch.tensor(
        [
            _rect_positions(2, 2) + [[-1, -1], [-1, -1]],
            _rect_positions(3, 1) + [[-1, -1], [-1, -1], [-1, -1]],
        ],
        dtype=torch.long,
    )

    counts = get_gemma4_pooled_token_counts(pixel_position_ids, output_length=6)

    assert torch.equal(counts, torch.tensor([4, 3], dtype=torch.long))


def test_get_pooled_token_counts_with_pooling() -> None:
    first_image = _rect_positions(6, 6)
    second_image = _rect_positions(3, 6) + [[-1, -1]] * 18
    pixel_position_ids = torch.tensor(
        [first_image, second_image],
        dtype=torch.long,
    )

    counts = get_gemma4_pooled_token_counts(pixel_position_ids, output_length=4)

    assert torch.equal(counts, torch.tensor([4, 2], dtype=torch.long))


def test_get_pooled_token_counts_rejects_non_square_pooling_ratio() -> None:
    pixel_position_ids = torch.tensor([_rect_positions(2, 3)], dtype=torch.long)

    with pytest.raises(ValueError, match="expected input_seq_len"):
        get_gemma4_pooled_token_counts(pixel_position_ids, output_length=4)


def test_get_pooled_token_counts_rejects_non_divisible_valid_grid() -> None:
    pixel_position_ids = torch.tensor(
        [_rect_positions(3, 2) + [[-1, -1], [-1, -1]]],
        dtype=torch.long,
    )

    with pytest.raises(ValueError, match="must be divisible"):
        get_gemma4_pooled_token_counts(pixel_position_ids, output_length=2)
