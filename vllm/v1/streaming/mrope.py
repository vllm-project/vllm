# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""mRoPE position helpers for streaming sessions.

A streaming session is one long-lived request that grows by one chunk per
frame. For each new chunk we compute mRoPE positions for the new tokens only,
starting just past the highest position already in the KV cache, and append
them to the session's position tensor. Positions for cached tokens are never
recomputed, so RoPE on surviving tokens stays consistent with their cached K/V
even after eviction leaves gaps in the position space.
"""

from dataclasses import replace
from typing import TYPE_CHECKING

import torch

from vllm.model_executor.models.interfaces import SupportsMRoPE

if TYPE_CHECKING:
    from vllm.multimodal.inputs import MultiModalFeatureSpec


def compute_chunk_mrope_positions(
    model: SupportsMRoPE,
    chunk_tokens: list[int],
    chunk_mm_features: list["MultiModalFeatureSpec"],
    base_position: int,
) -> tuple[torch.Tensor, int]:
    """Compute mRoPE positions for one chunk's new tokens.

    `chunk_tokens` and `chunk_mm_features` must cover the new chunk only, with
    feature offsets relative to the chunk start (offset 0 == first chunk token).
    Positions are shifted to begin at `base_position`; pass
    `request.max_cached_position + 1` so the chunk's first position is strictly
    greater than any position already in the KV cache.

    Returns the (3, len(chunk_tokens)) position tensor and the largest position
    assigned (use it to update `max_cached_position`).
    """
    if not chunk_tokens:
        return torch.empty((3, 0), dtype=torch.long), base_position - 1

    chunk_positions, _ = model.get_mrope_input_positions(
        chunk_tokens, chunk_mm_features
    )
    # The model numbers positions from 0 within the slice it sees; shift them
    # uniformly so the lowest equals `base_position`.
    chunk_positions = chunk_positions + base_position
    new_max_position = int(chunk_positions.max().item())
    return chunk_positions, new_max_position


def make_chunk_relative_mm_features(
    cumulative_mm_features: list["MultiModalFeatureSpec"],
    prev_num_tokens: int,
) -> list["MultiModalFeatureSpec"]:
    """Extract the latest chunk's mm-features with chunk-relative offsets.

    `cumulative_mm_features` is `req_state.mm_features` after the scheduler has
    shifted offsets to be session-absolute. Features at offset >=
    `prev_num_tokens` belong to the new chunk; they are copied with offset
    reduced by `prev_num_tokens` so the model's mRoPE function sees chunk-local
    indices.
    """
    chunk_features: list[MultiModalFeatureSpec] = []
    for feature in cumulative_mm_features:
        if feature.mm_position.offset >= prev_num_tokens:
            chunk_features.append(
                replace(
                    feature,
                    mm_position=replace(
                        feature.mm_position,
                        offset=feature.mm_position.offset - prev_num_tokens,
                    ),
                )
            )
    return chunk_features
