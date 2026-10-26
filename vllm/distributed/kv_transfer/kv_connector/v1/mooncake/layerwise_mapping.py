# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Device-independent logical block mapping for layerwise transfers."""

from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class BlockSlice:
    source: tuple[int, ...]
    destination: tuple[int, ...]
    end_tokens: int


def plan_append_blocks(
    source: Sequence[int],
    destination: Sequence[int],
    *,
    token_end: int,
    available: int,
    sent: int | None,
    block_size: int,
    final: bool,
) -> BlockSlice | None:
    """Map an uncached D suffix without discarding P logical positions."""
    if not destination:
        return None
    target = (token_end + block_size - 1) // block_size
    offset = target - len(destination)
    if offset < 0:
        raise ValueError("D requests more blocks than the handoff contains")
    end = min(target, (available + (block_size - 1 if final else 0)) // block_size)
    start = offset if sent is None else sent // block_size
    if end <= start:
        return None
    src = tuple(source[start:end])
    dst = tuple(destination[start - offset : end - offset])
    if len(src) != end - start or len(dst) != len(src):
        raise ValueError("Incomplete source/destination block mapping")
    return BlockSlice(src, dst, end * block_size)


def plan_final_window(
    source: Sequence[int],
    destination: Sequence[int],
    *,
    token_end: int,
    block_size: int,
    null_block: int,
) -> BlockSlice:
    """Align a clipped destination suffix without compacting logical null slots."""
    target = (token_end + block_size - 1) // block_size
    offset = target - len(destination)
    if offset < 0:
        raise ValueError("Window exceeds the handoff block table")
    src, dst = [], []
    for index, block in enumerate(destination, offset):
        if block == null_block:
            continue
        if index >= len(source) or source[index] == null_block:
            raise ValueError("Required window block was retired on producer")
        src.append(source[index])
        dst.append(block)
    return BlockSlice(tuple(src), tuple(dst), target * block_size)


def plan_final_state(
    source: Sequence[int],
    destination: Sequence[int],
    *,
    token_end: int,
    null_block: int,
) -> BlockSlice:
    """Pair resident state slots after removing align-mode placeholders."""
    src = tuple(block for block in source if block != null_block)
    dst = tuple(block for block in destination if block != null_block)
    if len(src) < len(dst):
        raise ValueError("Producer is missing required recurrent state slots")
    return BlockSlice(src[-len(dst) :] if dst else (), dst, token_end)
