# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Prefix-cache keys of frontier checkpoints."""

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from vllm.multimodal.utils import get_mm_features_in_window
from vllm.v1.core.kv_cache_utils import (
    BlockHash,
    generate_block_hash_extra_keys,
    hash_block_tokens,
)

if TYPE_CHECKING:
    from vllm.v1.request import Request

# Separates frontier keys from block hashes of the same prefix.
FRONTIER_KEY_MARKER = "mamba-frontier-checkpoint"


def frontier_key(
    hash_fn: Callable[[Any], bytes],
    request: "Request",
    num_tokens: int,
    hash_block_size: int,
    last_token: int | None = None,
) -> BlockHash:
    """Key of the state after `request`'s first `num_tokens` tokens, or
    after those tokens with the last one replaced by `last_token`.

    It chains the request's block hash at `h * hash_block_size` (`h =
    num_tokens // hash_block_size`) with the tokens after it, their extra
    keys (multimodal inputs, LoRA, cache salt) and `num_tokens`, so equal
    keys mean equal prefixes and lengths.
    """
    h = num_tokens // hash_block_size
    start = h * hash_block_size
    parent = request.block_hashes[h - 1] if h else None
    extra = _extra_keys(request, start, num_tokens)
    tokens = list(request.all_token_ids[start:num_tokens])
    if last_token is not None:
        tokens[-1] = last_token
    return hash_block_tokens(
        hash_fn,
        parent,
        tokens,
        (*(extra or ()), (FRONTIER_KEY_MARKER, num_tokens)),
    )


def _extra_keys(request: "Request", start: int, end: int) -> tuple[Any, ...] | None:
    """Prefix-cache extra keys (multimodal inputs, LoRA, cache salt) of
    tokens `[start, end)`."""
    mm_idx = 0
    mm_features = request.mm_features
    if start > 0 and mm_features:
        last = mm_features[-1].mm_position
        if last.offset + last.length > start:
            mm_idx, _ = get_mm_features_in_window(mm_features, start, end)
    extra, _ = generate_block_hash_extra_keys(request, start, end, mm_idx)
    return extra
