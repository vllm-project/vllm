# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared helpers for DeepSeek-V4's CPU-ported kernels."""

import torch


def map_local_to_global_slots_cpu(
    local_indices: torch.Tensor,
    req_idx: torch.Tensor,
    block_table: torch.Tensor,
    block_size: int,
) -> torch.Tensor:
    """Map per-request-local (token or compressed-token) indices to global
    paged-cache slot ids via a block table, mirroring
    ``compute_global_topk_indices_and_lens``.

    Args:
        local_indices: [N, K], -1 sentinel for invalid entries.
        req_idx: [N], row into ``block_table`` for each of the N rows.
        block_table: [num_reqs, max_blocks_per_seq].
        block_size: tokens (or compressed tokens) per physical block.

    Returns:
        [N, K] int64 global slot ids, -1 where ``local_indices`` was invalid.
    """
    valid = local_indices >= 0
    safe_local = local_indices.clamp(min=0).to(torch.int64)
    block_pos = torch.div(safe_local, block_size, rounding_mode="floor")
    offset = safe_local % block_size
    req_expand = req_idx.to(torch.int64).unsqueeze(-1).expand_as(local_indices)
    block_num = block_table[req_expand, block_pos].to(torch.int64)
    slot = block_num * block_size + offset
    return torch.where(valid, slot, torch.full_like(slot, -1))
