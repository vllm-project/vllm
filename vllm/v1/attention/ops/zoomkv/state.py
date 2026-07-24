# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Physical-block ZoomKV block-summary state (min/max/centroid/KIVI packed)."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from vllm.v1.attention.ops.zoomkv.quant_pack import pack_kcache_4bit


@dataclass
class ZoomKVBlockSummaryBuffers:
    """Per-layer block-summary tensors indexed by physical block id."""

    chunk_min: torch.Tensor  # [num_slots, kv_heads, head_dim]
    chunk_max: torch.Tensor
    centroid: torch.Tensor
    packed: torch.Tensor  # [num_slots, kv_heads, n_pack, block_size]
    valid: torch.Tensor  # [num_slots] bool
    parent_min: torch.Tensor | None = None
    parent_max: torch.Tensor | None = None
    parent_valid: torch.Tensor | None = None
    blocks_per_parent: int = 16


class ZoomKVBlockSummary:
    """Maintain ZoomKV summaries keyed by physical paged-block id.

    One vLLM physical block (== block_size tokens) maps to one Quest child
    chunk.
    """

    def __init__(
        self,
        num_blocks: int,
        num_kv_heads: int,
        head_dim: int,
        block_size: int,
        device: torch.device,
        dtype: torch.dtype = torch.bfloat16,
        blocks_per_parent: int = 16,
    ) -> None:
        if block_size != 16:
            raise ValueError(
                f"ZoomKV first release requires block_size=16, got {block_size}"
            )
        if head_dim % 8 != 0:
            raise ValueError(f"head_dim must be divisible by 8, got {head_dim}")
        self.num_blocks = num_blocks
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.block_size = block_size
        self.n_pack = head_dim // 8
        self.blocks_per_parent = blocks_per_parent
        self.device = device
        self.dtype = dtype

        self.chunk_min = torch.zeros(
            num_blocks, num_kv_heads, head_dim, device=device, dtype=dtype
        )
        self.chunk_max = torch.zeros(
            num_blocks, num_kv_heads, head_dim, device=device, dtype=dtype
        )
        self.centroid = torch.zeros(
            num_blocks, num_kv_heads, head_dim, device=device, dtype=dtype
        )
        self.packed = torch.zeros(
            num_blocks,
            num_kv_heads,
            self.n_pack,
            block_size,
            device=device,
            dtype=torch.int32,
        )
        self.valid = torch.zeros(num_blocks, device=device, dtype=torch.bool)
        self._request_block_summary_cache: dict[tuple, tuple[torch.Tensor, ...]] = {}

    def invalidate_blocks(self, block_ids: list[int] | torch.Tensor) -> None:
        """Mark physical blocks invalid (reuse / zero / free)."""
        self._request_block_summary_cache.clear()
        if isinstance(block_ids, torch.Tensor):
            ids = block_ids.to(device=self.device, dtype=torch.int64).reshape(-1)
        else:
            if not block_ids:
                return
            ids = torch.tensor(block_ids, device=self.device, dtype=torch.int64)
        ids = ids[(ids >= 0) & (ids < self.num_blocks)]
        if ids.numel() == 0:
            return
        self.valid.index_fill_(0, ids, False)
        self.chunk_min.index_fill_(0, ids, 0)
        self.chunk_max.index_fill_(0, ids, 0)
        self.centroid.index_fill_(0, ids, 0)
        self.packed.index_fill_(0, ids, 0)

    def copy_blocks(
        self,
        src_dst_pairs: list[tuple[int, int]] | torch.Tensor,
    ) -> None:
        """Copy block-summary state for CoW block remaps (src -> dst)."""
        self._request_block_summary_cache.clear()
        if isinstance(src_dst_pairs, torch.Tensor):
            if src_dst_pairs.numel() == 0:
                return
            pairs = src_dst_pairs.to(device="cpu", dtype=torch.int64).tolist()
        else:
            pairs = src_dst_pairs
        for src, dst in pairs:
            if src < 0 or dst < 0 or src >= self.num_blocks or dst >= self.num_blocks:
                continue
            if src == dst:
                continue
            self.chunk_min[dst].copy_(self.chunk_min[src])
            self.chunk_max[dst].copy_(self.chunk_max[src])
            self.centroid[dst].copy_(self.centroid[src])
            self.packed[dst].copy_(self.packed[src])
            self.valid[dst] = self.valid[src]

    def update_blocks_from_key_cache(
        self,
        key_cache: torch.Tensor,
        block_ids: torch.Tensor,
    ) -> None:
        """Recompute block summaries for complete physical blocks.

        Args:
            key_cache: [num_blocks, block_size, num_kv_heads, head_dim]
            block_ids: 1-D int tensor of physical block ids to refresh
        """
        self._request_block_summary_cache.clear()
        if block_ids.numel() == 0:
            return
        uniq = torch.unique(block_ids.to(torch.int64))
        uniq = uniq[(uniq >= 0) & (uniq < self.num_blocks)]
        if uniq.numel() == 0:
            return
        # Quantize all completed blocks in one batched GPU pipeline. The
        # original per-block Python loop launched ~10 tiny kernels per block
        # and made long-prefill block-summary construction slower than attention.
        keys = (
            key_cache.index_select(0, uniq).permute(0, 2, 1, 3).contiguous()
        )  # [n, H, block_size, D]
        cmin = keys.amin(dim=2)
        cmax = keys.amax(dim=2)
        cent = keys.mean(dim=2)
        packed_tok, _ = pack_kcache_4bit(
            keys,
            cmin.unsqueeze(2),
            cmax.unsqueeze(2),
            group_size=self.block_size,
            bits=4,
        )  # [n, H, block_size, n_pack]
        packed = packed_tok.permute(0, 1, 3, 2).contiguous()
        self.packed.index_copy_(0, uniq, packed)
        self.chunk_min.index_copy_(0, uniq, cmin)
        self.chunk_max.index_copy_(0, uniq, cmax)
        self.centroid.index_copy_(0, uniq, cent)
        self.valid.index_fill_(0, uniq, True)

    def update_completed_slots(
        self,
        key_cache: torch.Tensor,
        slots: torch.Tensor,
    ) -> None:
        """Finalize decode blocks without a GPU→CPU predicate synchronization."""
        if slots.numel() == 0:
            return
        if slots.is_cuda:
            from vllm.v1.attention.ops.zoomkv.block_summary_triton import (
                compact_completed_slots,
                finalize_completed_slots,
            )

            finalize_slots = slots
            if slots.numel() > 256:
                # Prefill: compact to block-ending slots before launching the
                # D=256 finalizer. This avoids 64 masked programs per ordinary
                # token while retaining the sync-free one-token decode path.
                finalize_slots = compact_completed_slots(slots, self.block_size)
                self._request_block_summary_cache.clear()
            finalize_completed_slots(key_cache, finalize_slots, self)
            return
        # A large update is prefill/admission for a new batch; previously
        # cached logical request views can no longer be reused.
        self._request_block_summary_cache.clear()
        valid_slots = slots[slots >= 0]
        block_ids = torch.div(valid_slots, self.block_size, rounding_mode="floor")
        offsets = torch.remainder(valid_slots, self.block_size)
        complete_blocks = block_ids[offsets == (self.block_size - 1)]
        self.update_blocks_from_key_cache(key_cache, complete_blocks)

    def cached_request_block_summaries(
        self,
        physical_block_ids: torch.Tensor,
        cache_key: tuple,
    ) -> tuple[torch.Tensor, ...]:
        """Reuse gathered child/parent block_summaries across 16 decode steps."""
        cached = self._request_block_summary_cache.get(cache_key)
        if cached is not None:
            return cached
        packed, cmin, cmax, centroid, valid = self.gather_request_block_summaries(
            physical_block_ids
        )
        pmin, pmax, pvalid = self.build_parent_minmax(
            physical_block_ids, cmin, cmax, valid
        )
        result = (packed, cmin, cmax, centroid, valid, pmin, pmax, pvalid)
        if len(self._request_block_summary_cache) >= 4:
            self._request_block_summary_cache.pop(
                next(iter(self._request_block_summary_cache))
            )
        self._request_block_summary_cache[cache_key] = result
        return result

    def gather_request_block_summaries(
        self,
        physical_block_ids: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Gather child-chunk block_summaries for a request's retrieval blocks.

        Args:
            physical_block_ids: [n_chunks] physical block ids in logical order
        Returns:
            packed: [1, kv, n_chunks, n_pack, g]
            chunk_min/max/centroid: [1, kv, n_chunks, D]
            valid: [1, kv, n_chunks] bool
        """
        ids = physical_block_ids.to(torch.int64).clamp(0, self.num_blocks - 1)
        n = ids.numel()
        packed = (
            self.packed.index_select(0, ids)
            .permute(1, 0, 2, 3)
            .unsqueeze(0)
            .contiguous()
        )
        chunk_min = (
            self.chunk_min.index_select(0, ids)
            .permute(1, 0, 2)
            .unsqueeze(0)
            .contiguous()
        )
        chunk_max = (
            self.chunk_max.index_select(0, ids)
            .permute(1, 0, 2)
            .unsqueeze(0)
            .contiguous()
        )
        centroid = (
            self.centroid.index_select(0, ids)
            .permute(1, 0, 2)
            .unsqueeze(0)
            .contiguous()
        )
        valid = self.valid.index_select(0, ids)
        valid = valid.view(1, 1, n).expand(1, self.num_kv_heads, n).contiguous()
        return packed, chunk_min, chunk_max, centroid, valid

    def build_parent_minmax(
        self,
        physical_block_ids: torch.Tensor,
        chunk_min: torch.Tensor,
        chunk_max: torch.Tensor,
        valid: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Aggregate child chunks into parent chunks for hierarchical Quest."""
        del physical_block_ids
        factor = self.blocks_per_parent
        n_chunks = chunk_min.shape[2]
        n_parent = n_chunks // factor
        if n_parent <= 0:
            empty = chunk_min[:, :, :0, :]
            return empty, empty.clone(), valid[:, :, :0]
        usable = n_parent * factor
        cmin = chunk_min[:, :, :usable, :].reshape(
            1, self.num_kv_heads, n_parent, factor, self.head_dim
        )
        cmax = chunk_max[:, :, :usable, :].reshape(
            1, self.num_kv_heads, n_parent, factor, self.head_dim
        )
        v = valid[:, :, :usable].reshape(1, self.num_kv_heads, n_parent, factor)
        neg = torch.full_like(cmin, float("-inf"))
        pos = torch.full_like(cmax, float("inf"))
        cmin_m = torch.where(v.unsqueeze(-1), cmin, pos)
        cmax_m = torch.where(v.unsqueeze(-1), cmax, neg)
        parent_min = cmin_m.amin(dim=3)
        parent_max = cmax_m.amax(dim=3)
        parent_valid = v.any(dim=3)
        parent_min = torch.where(
            parent_valid.unsqueeze(-1), parent_min, torch.zeros_like(parent_min)
        )
        parent_max = torch.where(
            parent_valid.unsqueeze(-1), parent_max, torch.zeros_like(parent_max)
        )
        return parent_min, parent_max, parent_valid


# Global registry: layer_name -> ZoomKVBlockSummary
_LAYER_BLOCK_SUMMARIES: dict[str, ZoomKVBlockSummary] = {}


def get_or_create_block_summary(
    layer_name: str,
    num_blocks: int,
    num_kv_heads: int,
    head_dim: int,
    block_size: int,
    device: torch.device,
    dtype: torch.dtype,
    blocks_per_parent: int = 16,
) -> ZoomKVBlockSummary:
    sc = _LAYER_BLOCK_SUMMARIES.get(layer_name)
    if (
        sc is None
        or sc.num_blocks != num_blocks
        or sc.num_kv_heads != num_kv_heads
        or sc.head_dim != head_dim
        or sc.block_size != block_size
        or sc.device != device
    ):
        sc = ZoomKVBlockSummary(
            num_blocks=num_blocks,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            block_size=block_size,
            device=device,
            dtype=dtype,
            blocks_per_parent=blocks_per_parent,
        )
        _LAYER_BLOCK_SUMMARIES[layer_name] = sc
    return sc


def iter_block_summaries() -> list[ZoomKVBlockSummary]:
    return list(_LAYER_BLOCK_SUMMARIES.values())


def invalidate_block_summaries_for_blocks(
    block_ids: list[int],
    allocation_num_blocks: int | None = None,
) -> None:
    if not block_ids:
        return
    for sc in _LAYER_BLOCK_SUMMARIES.values():
        factor = (
            sc.num_blocks // allocation_num_blocks
            if allocation_num_blocks and sc.num_blocks % allocation_num_blocks == 0
            else 1
        )
        expanded = [
            block_id * factor + offset
            for block_id in block_ids
            for offset in range(factor)
        ]
        sc.invalidate_blocks(expanded)


def copy_block_summaries_for_block_pairs(
    pairs: list[tuple[int, int]],
    allocation_num_blocks: int | None = None,
) -> None:
    if not pairs:
        return
    for sc in _LAYER_BLOCK_SUMMARIES.values():
        factor = (
            sc.num_blocks // allocation_num_blocks
            if allocation_num_blocks and sc.num_blocks % allocation_num_blocks == 0
            else 1
        )
        expanded = [
            (src * factor + offset, dst * factor + offset)
            for src, dst in pairs
            for offset in range(factor)
        ]
        sc.copy_blocks(expanded)


def clear_block_summaries() -> None:
    _LAYER_BLOCK_SUMMARIES.clear()
