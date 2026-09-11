# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MiniMax-M3 FlyDSL MoE decode helpers."""

from __future__ import annotations

MAX_DECODE_TOKENS = 256


def supports_shapes(hidden_size: int, intermediate_size: int) -> bool:
    """Dimensions must fit the 256-wide K tiles and the three-way split-K."""
    return hidden_size % 256 == 0 and intermediate_size % (256 * 3) == 0
