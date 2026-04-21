# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Utilities shared by Gemma4 multimodal model code and tests."""

import math

import torch


def get_gemma4_pooled_token_counts(
    pixel_position_ids: torch.Tensor,
    output_length: int,
) -> torch.Tensor:
    """Compute the number of non-padding pooled tokens per image.

    Gemma4's image processor produces a dense rectangle of valid patch
    coordinates and pads the remaining patch slots with ``(-1, -1)``.
    After pooling by ``k x k``, the valid soft-token count is simply the
    pooled grid area for each image.
    """
    if pixel_position_ids.ndim != 3 or pixel_position_ids.shape[-1] != 2:
        raise ValueError(
            "pixel_position_ids must have shape "
            f"(batch, max_patches, 2), got {tuple(pixel_position_ids.shape)}"
        )
    if output_length <= 0:
        raise ValueError(f"output_length must be positive, got {output_length}.")

    input_seq_len = pixel_position_ids.shape[1]
    quotient, remainder = divmod(input_seq_len, output_length)
    if remainder:
        raise ValueError(
            f"Cannot map {input_seq_len=} to {output_length=}: "
            "expected input_seq_len == output_length * k^2."
        )

    pooling_factor = math.isqrt(quotient)
    if pooling_factor**2 != quotient:
        raise ValueError(
            f"Cannot map {input_seq_len=} to {output_length=}: "
            "expected input_seq_len == output_length * k^2."
        )

    padding_positions = (pixel_position_ids == -1).all(dim=-1)
    clamped_positions = pixel_position_ids.clamp(min=0)
    valid_x = clamped_positions[..., 0].masked_fill(padding_positions, -1)
    valid_y = clamped_positions[..., 1].masked_fill(padding_positions, -1)

    width_patches = valid_x.amax(dim=-1) + 1
    height_patches = valid_y.amax(dim=-1) + 1
    if ((width_patches % pooling_factor) != 0).any() or (
        (height_patches % pooling_factor) != 0
    ).any():
        raise ValueError(
            "Gemma4 valid patch grid dimensions must be divisible by the "
            f"pooling factor {pooling_factor}."
        )

    return (width_patches // pooling_factor) * (height_patches // pooling_factor)
