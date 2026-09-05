# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Paged-cache helpers for KSA attention."""

from __future__ import annotations

import torch


def build_ksa_summary_slot_mapping(
    *,
    token_positions: torch.Tensor,
    token_to_request: torch.Tensor,
    boundary_mask: torch.Tensor,
    block_table: torch.Tensor,
    manager_block_size: int,
    states_per_block: int,
    summary_chunk_size: int,
) -> torch.Tensor:
    """Map chunk-boundary text rows to compressed Summary cache slots."""
    if manager_block_size % summary_chunk_size != 0:
        raise ValueError("KSA cache block size must be divisible by summary_chunk_size")
    if states_per_block != manager_block_size // summary_chunk_size:
        raise ValueError("invalid compressed Summary cache geometry")
    if token_positions.shape != token_to_request.shape:
        raise ValueError("token position and request mappings must have equal shape")
    if boundary_mask.shape != token_positions.shape:
        raise ValueError("boundary mask must match token positions")

    slots = torch.full_like(token_positions, -1, dtype=torch.int64)
    if token_positions.numel() == 0:
        return slots

    summary_positions = torch.div(
        token_positions, summary_chunk_size, rounding_mode="floor"
    )
    logical_block_indices = torch.div(
        summary_positions, states_per_block, rounding_mode="floor"
    )
    safe_block_indices = torch.where(
        boundary_mask,
        logical_block_indices,
        torch.zeros_like(logical_block_indices),
    )
    request_indices = token_to_request.to(dtype=torch.int64)
    physical_blocks = block_table[request_indices, safe_block_indices]
    compressed_offsets = summary_positions.remainder(states_per_block)
    valid_slots = (
        physical_blocks.to(torch.int64) * states_per_block + compressed_offsets
    )
    slots.copy_(torch.where(boundary_mask, valid_slots, slots))
    return slots


def split_ksa_kv_cache(
    kv_cache: torch.Tensor,
    *,
    head_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return K and V views with shape ``[blocks, states, heads, dim]``."""
    if kv_cache.ndim != 4:
        raise ValueError("KSA KV cache must have shape [blocks, heads, states, K+V]")
    if kv_cache.shape[-1] != 2 * head_dim:
        raise ValueError("KSA KV cache content width does not match head_dim")
    return kv_cache.transpose(1, 2).split(head_dim, dim=-1)


__all__ = [
    "build_ksa_summary_slot_mapping",
    "split_ksa_kv_cache",
]
