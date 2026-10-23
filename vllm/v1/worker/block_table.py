# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math

from vllm.utils.math_utils import cdiv


def get_block_table_width(
    max_num_blocks: int,
    block_size: int,
    *,
    token_alignment: int | None = 128,
) -> int:
    """Return the width after optional alignment."""
    if token_alignment is not None:
        if token_alignment <= 0:
            raise ValueError("token_alignment must be positive")
        block_alignment = token_alignment // math.gcd(token_alignment, block_size)
        max_num_blocks = cdiv(max_num_blocks, block_alignment) * block_alignment
    return max_num_blocks
