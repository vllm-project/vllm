# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.triton_utils import tl, triton


def cp_local_to_global_indices(
    local_indices: torch.Tensor,
    cp_world_size: int,
    cp_rank: int,
    cp_kv_cache_interleave_size: int,
) -> torch.Tensor:
    safe_indices = torch.clamp(local_indices, min=0)
    interleave_blocks = safe_indices // cp_kv_cache_interleave_size
    interleave_offsets = safe_indices % cp_kv_cache_interleave_size
    global_indices = (
        interleave_blocks * (cp_kv_cache_interleave_size * cp_world_size)
        + cp_rank * cp_kv_cache_interleave_size
    )
    global_indices += interleave_offsets
    return torch.where(local_indices >= 0, global_indices, -1)


def get_cp_local_seq_lens(
    seq_lens: torch.Tensor,
    cp_world_size: int = 1,
    cp_rank: int | None = None,
    cp_kv_cache_interleave_size: int = 1,
) -> torch.Tensor:
    num_requests = seq_lens.size(0)
    if cp_rank is None:
        rank_offsets = (
            torch.arange(cp_world_size, dtype=torch.int32, device=seq_lens.device)
            .unsqueeze(0)
            .expand(num_requests, -1)
        )
    else:
        rank_offsets = torch.tensor(
            [[cp_rank]], dtype=torch.int32, device=seq_lens.device
        )
    seq_lens_tiled = seq_lens.to(torch.int32).unsqueeze(-1)
    seq_lens_tiled = seq_lens_tiled.expand(-1, rank_offsets.shape[1])
    rank_stride = cp_world_size * cp_kv_cache_interleave_size
    base = seq_lens_tiled // rank_stride * cp_kv_cache_interleave_size
    remainder = seq_lens_tiled - base * cp_world_size
    extra = torch.clip(
        remainder - rank_offsets * cp_kv_cache_interleave_size,
        0,
        cp_kv_cache_interleave_size,
    )
    return (base + extra).squeeze(1)


@triton.jit
def cp_global_to_local_pos(
    pos,
    CP_WORLD_SIZE: tl.constexpr,
    CP_RANK: tl.constexpr,
    CP_KV_CACHE_INTERLEAVE_SIZE: tl.constexpr,
):
    rank_stride = CP_WORLD_SIZE * CP_KV_CACHE_INTERLEAVE_SIZE
    base = pos // rank_stride * CP_KV_CACHE_INTERLEAVE_SIZE
    remainder = pos - base * CP_WORLD_SIZE
    extra = tl.minimum(
        tl.maximum(remainder - CP_RANK * CP_KV_CACHE_INTERLEAVE_SIZE, 0),
        CP_KV_CACHE_INTERLEAVE_SIZE,
    )
    return base + extra


@triton.jit
def cp_is_local_pos(
    pos,
    CP_WORLD_SIZE: tl.constexpr,
    CP_RANK: tl.constexpr,
    CP_KV_CACHE_INTERLEAVE_SIZE: tl.constexpr,
):
    return (pos // CP_KV_CACHE_INTERLEAVE_SIZE) % CP_WORLD_SIZE == CP_RANK
