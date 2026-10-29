# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Token tiles of whole requests, each with its own mailbox storage, for the
launches past one tile's rows."""

from dataclasses import replace

MAX_TOKENS = 32  # a launch's rows
TILE_TOKENS = 8  # rows past which K1 tiles its x rows; a request's tokens at most


def token_tiles(
    tokens: int, sequence_length: int = 1, *, tile_tokens: int = TILE_TOKENS
) -> tuple[tuple[int, int], ...]:
    """(first row, row count) tiles of ``tokens`` rows, no request split."""
    if not 1 <= tokens <= MAX_TOKENS:
        raise ValueError(f"mono needs 1..{MAX_TOKENS} rows, got {tokens}")
    if not 1 <= sequence_length <= TILE_TOKENS or tokens % sequence_length:
        raise ValueError(
            f"mono tiles need whole requests of at most {TILE_TOKENS} rows"
        )
    if tile_tokens not in (8, 16):
        raise ValueError("mono tile size must be 8 or 16")
    rows = tile_tokens // sequence_length * sequence_length
    return tuple((first, min(rows, tokens - first)) for first in range(0, tokens, rows))


def tile_layout(
    key, layout, sequence_length: int = 1, *, tile_tokens: int = TILE_TOKENS
) -> dict:
    """Each tile's ``layout`` regions, 256 B aligned one after another, as
    ``tile{i}/{name}``."""
    result = {}
    offset = 0
    for index, (_, rows) in enumerate(
        token_tiles(key.tokens, sequence_length, tile_tokens=tile_tokens)
    ):
        regions = layout(replace(key, tokens=rows))
        offset = (offset + 255) // 256 * 256
        for name, (start, size) in regions.items():
            result[f"tile{index}/{name}"] = (offset + start, size)
        offset += max(start + size for start, size in regions.values())
    return result
