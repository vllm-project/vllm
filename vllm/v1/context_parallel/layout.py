# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from vllm.triton_utils import tl, triton

if TYPE_CHECKING:
    from vllm.config import VllmConfig


@dataclass(frozen=True, slots=True)
class ContextParallelLayout:
    """Logical interleaved context-parallel token layout."""

    world_size: int = 1
    rank: int = 0
    interleave_size: int = 1

    def __post_init__(self) -> None:
        if self.world_size < 1:
            raise ValueError("world_size must be positive")
        if not 0 <= self.rank < self.world_size:
            raise ValueError("rank must be in [0, world_size)")
        if self.interleave_size < 1:
            raise ValueError("interleave_size must be positive")

    @property
    def enabled(self) -> bool:
        return self.world_size > 1

    @classmethod
    def from_config(cls, vllm_config: "VllmConfig") -> "ContextParallelLayout":
        parallel_config = vllm_config.parallel_config
        world_size = parallel_config.decode_context_parallel_size
        rank = 0
        if world_size > 1:
            from vllm.distributed import get_dcp_group

            rank = get_dcp_group().rank_in_group
        return cls(
            world_size=world_size,
            rank=rank,
            interleave_size=parallel_config.cp_kv_cache_interleave_size,
        )

    def local_to_global(self, local_indices: torch.Tensor) -> torch.Tensor:
        safe_indices = torch.clamp(local_indices, min=0)
        interleave_blocks = safe_indices // self.interleave_size
        interleave_offsets = safe_indices % self.interleave_size
        global_indices = (
            interleave_blocks * (self.interleave_size * self.world_size)
            + self.rank * self.interleave_size
            + interleave_offsets
        )
        return torch.where(local_indices >= 0, global_indices, -1)

    def owns(self, global_indices: torch.Tensor) -> torch.Tensor:
        safe_indices = torch.clamp(global_indices, min=0)
        owner = (safe_indices // self.interleave_size) % self.world_size
        return (global_indices >= 0) & (owner == self.rank)

    def global_to_local(self, global_indices: torch.Tensor) -> torch.Tensor:
        safe_indices = torch.clamp(global_indices, min=0)
        rank_stride = self.world_size * self.interleave_size
        base = safe_indices // rank_stride * self.interleave_size
        remainder = safe_indices - base * self.world_size
        extra = torch.clip(
            remainder - self.rank * self.interleave_size,
            0,
            self.interleave_size,
        )
        return torch.where(global_indices >= 0, base + extra, -1)

    def local_seq_lens(self, seq_lens: torch.Tensor) -> torch.Tensor:
        return self.global_to_local(seq_lens.to(torch.int32))

    def all_local_seq_lens(self, seq_lens: torch.Tensor) -> torch.Tensor:
        num_requests = seq_lens.size(0)
        rank_offsets = (
            torch.arange(
                self.world_size,
                dtype=torch.int32,
                device=seq_lens.device,
            )
            .unsqueeze(0)
            .expand(num_requests, -1)
        )
        seq_lens_tiled = seq_lens.to(torch.int32).unsqueeze(-1)
        rank_stride = self.world_size * self.interleave_size
        base = seq_lens_tiled // rank_stride * self.interleave_size
        remainder = seq_lens_tiled - base * self.world_size
        extra = torch.clip(
            remainder - rank_offsets * self.interleave_size,
            0,
            self.interleave_size,
        )
        return (base + extra).squeeze(1)

    def triton_kwargs(
        self,
        world_key: str = "DCP_WORLD_SIZE",
        rank_key: str = "DCP_RANK",
    ) -> dict[str, int]:
        return {
            world_key: self.world_size,
            rank_key: self.rank,
            "CP_KV_CACHE_INTERLEAVE_SIZE": self.interleave_size,
        }


DEFAULT_CP_LAYOUT = ContextParallelLayout()
InterleavedCPLayout = ContextParallelLayout


def get_dcp_local_seq_lens(
    seq_lens: torch.Tensor,
    dcp_size: int = 1,
    dcp_rank: int | None = None,
    cp_kv_cache_interleave_size: int = 1,
) -> torch.Tensor:
    """Return per-rank DCP local sequence lengths."""
    layout = ContextParallelLayout(
        world_size=dcp_size,
        rank=0 if dcp_rank is None else dcp_rank,
        interleave_size=cp_kv_cache_interleave_size,
    )
    return (
        layout.all_local_seq_lens(seq_lens)
        if dcp_rank is None
        else layout.local_seq_lens(seq_lens)
    )


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


@triton.jit
def cp_global_to_local_block(
    pos,
    block_size,
    CP_WORLD_SIZE: tl.constexpr,
    CP_RANK: tl.constexpr,
    CP_KV_CACHE_INTERLEAVE_SIZE: tl.constexpr,
):
    """Map a global logical position to a rank-local physical block address."""
    virtual_block_size = block_size * CP_WORLD_SIZE
    block_idx = pos // virtual_block_size
    virtual_offset = pos - block_idx * virtual_block_size
    block_offset = (
        virtual_offset // (CP_WORLD_SIZE * CP_KV_CACHE_INTERLEAVE_SIZE)
    ) * CP_KV_CACHE_INTERLEAVE_SIZE + (virtual_offset % CP_KV_CACHE_INTERLEAVE_SIZE)
    is_local = cp_is_local_pos(
        virtual_offset,
        CP_WORLD_SIZE,
        CP_RANK,
        CP_KV_CACHE_INTERLEAVE_SIZE,
    )
    return block_idx, block_offset, is_local
