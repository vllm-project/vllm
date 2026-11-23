# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from typing import Literal, cast


def suffix_token_counts(
    num_tokens: int,
    block_ids: list[list[int]],
    block_sizes: dict[int, int],
) -> dict[int, int]:
    """Valid tokens in the transferred suffix, after D's block-aligned prefix hit."""
    result = {}
    for group, block_size in block_sizes.items():
        full_blocks = (num_tokens + block_size - 1) // block_size
        # A full prefix hit (or a release-only notification) has no block table.
        count = len(block_ids[group]) if block_ids else 0
        result[group] = num_tokens - (full_blocks - count) * block_size if count else 0
    return result


def chunk_token_counts(
    valid_tokens_by_group: dict[int, int],
    block_sizes: dict[int, int],
    logical_start: int,
    logical_count: int,
) -> dict[int, int]:
    """Slice a session's valid suffix into a physical transfer chunk."""
    result = {}
    for group, block_size in block_sizes.items():
        remaining = valid_tokens_by_group[group] - logical_start * block_size
        count = min(remaining, logical_count * block_size)
        result[group] = count
    return result


def chunk_window(
    chunk_id: int, logical_block_count: int, logical_blocks_per_chunk: int
) -> tuple[int, int, Literal[0, 1]]:
    logical_start = chunk_id // 2 * logical_blocks_per_chunk
    count = min(logical_blocks_per_chunk, logical_block_count - logical_start)
    return logical_start, count, cast(Literal[0, 1], chunk_id % 2)
