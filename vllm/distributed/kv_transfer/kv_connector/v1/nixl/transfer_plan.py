# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Plan-based transfer design for NIXL connector.

Data structures, plan generators, and local descriptor builders
for NIXL KV cache transfers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from vllm.distributed.kv_transfer.kv_connector.utils import (
    BlockIds,
    EngineTransferInfo,
    TransferTopology,
)
from vllm.distributed.kv_transfer.kv_connector.v1.ssm_conv_transfer_utils import (
    MambaConvSplitInfo,
)
from vllm.v1.kv_cache_interface import AttentionSpec, KVCacheSpec, MambaSpec

if TYPE_CHECKING:
    from vllm.distributed.kv_transfer.kv_connector.v1.nixl.metadata import (
        NixlAgentMetadata,
    )


# ======================================================================
# 1. Data structures
# ======================================================================


@dataclass(frozen=True)
class ReadSpec:
    """Specification for a single remote block read operation."""

    remote_rank: int
    local_block_ids: BlockIds
    remote_block_ids: BlockIds


def _is_attention_spec(spec_type: type[KVCacheSpec]) -> bool:
    return issubclass(spec_type, AttentionSpec)


def _is_ssm_spec(spec_type: type[KVCacheSpec]) -> bool:
    return issubclass(spec_type, MambaSpec)


@dataclass(frozen=True)
class RegionPlan:
    """Pre-computed plan for one descriptor region.

    Everything needed to build NIXL descriptors and compute descriptor
    IDs is baked in.  The executor plugs in per-rank ``base_addr`` and
    ``device_id`` from NixlAgentMetadata.
    """

    layer_idx: int

    # Descriptor geometry
    descriptor_bytes: int
    offset_in_page: int
    page_stride: int
    num_blocks: int


@dataclass(frozen=True)
class EngineTransferPlan:
    """Complete transfer plan for one remote engine.

    Generated once during handshake.  Regions are split into
    ``fa_regions`` and ``ssm_regions`` matching the descriptor
    handle layout.
    """

    # Regions in descriptor handle order
    fa_regions: tuple[RegionPlan, ...]
    ssm_regions: tuple[RegionPlan, ...]

    # Per-group KVCacheSpec type — used for descriptor indexing.
    group_spec_types: tuple[type[KVCacheSpec], ...]

    # Per-group ordered source ranks. Position = local piece index.
    source_ranks_per_group: tuple[tuple[int, ...], ...]

    # Superset of all source ranks (union of all groups).
    all_source_ranks: tuple[int, ...]

    # Maps each source rank to its FA head slot index.
    rank_to_attention_slot: dict[int, int]

    # Stride for expanding remote logical block IDs to kernel block IDs.
    # Dense: local_physical_blocks_per_logical.
    # Mamba: remote_physical_blocks_per_logical.
    remote_expansion_stride: int

    @property
    def all_regions(self) -> tuple[RegionPlan, ...]:
        return self.fa_regions + self.ssm_regions


# ======================================================================
# 2. Internal helpers
# ======================================================================


def _get_kv_block_len(
    layer_idx: int,
    block_len_per_layer: list[int],
    is_blocks_first: bool,
) -> int:
    if is_blocks_first:
        return block_len_per_layer[layer_idx] // 2
    return block_len_per_layer[layer_idx]


@dataclass(frozen=True)
class TPMapping:
    """Complete local-to-remote TP mapping for one remote engine."""

    source_ranks_per_group: tuple[tuple[int, ...], ...]
    all_source_ranks: tuple[int, ...]
    rank_to_attention_slot: dict[int, int]
    rank_offset_factor: int


def _compute_tp_mapping(
    tp_rank: int,
    tp_size: int,
    remote_tp_size: int,
    is_mla: bool,
    total_num_kv_heads: int,
    group_spec_types: tuple[type[KVCacheSpec], ...],
) -> TPMapping:
    """Build the complete local-to-remote TP mapping.

    Computes source ranks, head slot assignments, and the rank offset
    factor in a single pass.  Both generators call this and unpack.
    """
    # --- Attention source ranks ---
    if is_mla:
        # All heads replicated across all ranks.
        attn_ranks = [0]
    elif tp_size >= remote_tp_size:
        attn_ranks = [tp_rank * remote_tp_size // tp_size]
    else:
        # P (remote TP) > D (local TP): one local rank
        # reads from multiple remote ranks.
        # GQA dedup: when K < remote_tp_size, several remote ranks
        # hold the same KV head.  np.unique keeps only the first
        # rank per unique head so we don't issue redundant reads.
        abs_tp = remote_tp_size // tp_size
        start = tp_rank * abs_tp
        heads = np.arange(start, start + abs_tp) * total_num_kv_heads // remote_tp_size
        _, unique_idx = np.unique(heads, return_index=True)
        attn_ranks = (start + np.sort(unique_idx)).tolist()

    # --- SSM source ranks ---
    has_ssm = any(_is_ssm_spec(t) for t in group_spec_types)
    if has_ssm:
        if tp_size < remote_tp_size:
            abs_tp = remote_tp_size // tp_size
            ssm_ranks = list(range(tp_rank * abs_tp, (tp_rank + 1) * abs_tp))
        else:
            ssm_ranks = list(attn_ranks)
    else:
        ssm_ranks = []

    all_ranks = sorted(set(attn_ranks) | set(ssm_ranks))

    # --- Per-group ordered source ranks ---
    source_ranks_per_group = tuple(
        tuple(ssm_ranks) if _is_ssm_spec(t) else tuple(attn_ranks)
        for t in group_spec_types
    )

    # --- Attention head slots ---
    head_to_slot: dict[int, int] = {}
    for i, r in enumerate(attn_ranks):
        head_to_slot[r * total_num_kv_heads // remote_tp_size] = i
    rank_to_attention_slot = {
        r: head_to_slot.get(r * total_num_kv_heads // remote_tp_size, 0)
        for r in all_ranks
    }

    # --- Rank offset factor ---
    if is_mla or tp_size <= remote_tp_size:
        rank_offset_factor = 0
    elif tp_size > total_num_kv_heads:
        local_head = tp_rank * total_num_kv_heads // tp_size
        p_start = attn_ranks[0] * total_num_kv_heads // remote_tp_size
        rank_offset_factor = local_head - p_start
    else:
        rank_offset_factor = tp_rank % (tp_size // remote_tp_size)

    return TPMapping(
        source_ranks_per_group=source_ranks_per_group,
        all_source_ranks=tuple(all_ranks),
        rank_to_attention_slot=rank_to_attention_slot,
        rank_offset_factor=rank_offset_factor,
    )


def _build_fa_regions(
    *,
    block_len_per_layer: list[int],
    remote_block_lens: list[int],
    is_blocks_first: bool,
    block_size_ratio: int,
    num_attn_reads: int,
    rank_offset_factor: int,
    remote_num_blocks: int,
) -> list[RegionPlan]:
    """Build FA (attention) regions for the transfer plan.

    K bytes = remote_kv_block_len / num_attn_reads.
    V bytes = local_block_len / num_attn_reads (no block_size_ratio).
    Offset = rank_offset_factor * remote_kv_block_len per layer.
    """
    fa_regions: list[RegionPlan] = []
    for i in range(len(remote_block_lens)):
        local_block_len = _get_kv_block_len(i, block_len_per_layer, is_blocks_first)
        remote_kv_block_len = local_block_len // block_size_ratio
        k_desc_bytes = remote_kv_block_len // num_attn_reads
        rank_offset = rank_offset_factor * remote_kv_block_len
        page_stride = remote_block_lens[i]

        fa_regions.append(
            RegionPlan(
                layer_idx=i,
                descriptor_bytes=k_desc_bytes,
                offset_in_page=rank_offset,
                page_stride=page_stride,
                num_blocks=remote_num_blocks,
            )
        )

        if is_blocks_first:
            v_desc_bytes = local_block_len // num_attn_reads
            fa_regions.append(
                RegionPlan(
                    layer_idx=i,
                    descriptor_bytes=v_desc_bytes,
                    offset_in_page=rank_offset + page_stride // 2,
                    page_stride=page_stride,
                    num_blocks=remote_num_blocks,
                )
            )

    return fa_regions


# ======================================================================
# 3. Plan generators
# ======================================================================


def generate_dense_plan(
    *,
    transfer_topo: TransferTopology,
    block_len_per_layer: list[int],
    remote_info: EngineTransferInfo,
    remote_meta: NixlAgentMetadata,
    group_spec_types: tuple[type[KVCacheSpec], ...],
    local_physical_blocks_per_logical: int,
) -> EngineTransferPlan:
    """Generate transfer plan for dense (attention-only) models."""
    block_size_ratio = transfer_topo.block_size // remote_info.remote_block_size

    m = _compute_tp_mapping(
        transfer_topo.tp_rank,
        transfer_topo.tp_size,
        remote_info.remote_tp_size,
        transfer_topo.is_mla,
        transfer_topo.total_num_kv_heads,
        group_spec_types=group_spec_types,
    )

    fa_regions = _build_fa_regions(
        block_len_per_layer=block_len_per_layer,
        remote_block_lens=remote_meta.block_lens,
        is_blocks_first=transfer_topo.is_kv_layout_blocks_first,
        block_size_ratio=block_size_ratio,
        num_attn_reads=len(m.source_ranks_per_group[0]),
        rank_offset_factor=m.rank_offset_factor,
        remote_num_blocks=remote_meta.num_blocks,
    )

    return EngineTransferPlan(
        fa_regions=tuple(fa_regions),
        ssm_regions=(),
        group_spec_types=group_spec_types,
        source_ranks_per_group=m.source_ranks_per_group,
        all_source_ranks=m.all_source_ranks,
        rank_to_attention_slot=m.rank_to_attention_slot,
        remote_expansion_stride=local_physical_blocks_per_logical,
    )


def generate_mamba_plan(
    *,
    transfer_topo: TransferTopology,
    block_len_per_layer: list[int],
    remote_info: EngineTransferInfo,
    remote_meta: NixlAgentMetadata,
    group_spec_types: tuple[type[KVCacheSpec], ...],
    conv_decomp: MambaConvSplitInfo,
    ssm_sizes: tuple[int, int],
) -> EngineTransferPlan:
    """Generate transfer plan for hybrid Mamba (SSM + FA) models."""
    assert _is_attention_spec(group_spec_types[0]), (
        f"First group must be an attention spec, got {group_spec_types[0]}"
    )
    tp_rank = transfer_topo.tp_rank
    tp_size = transfer_topo.tp_size
    remote_tp_size = remote_info.remote_tp_size
    remote_phys_ratio = remote_info.remote_physical_blocks_per_logical
    remote_block_lens = remote_meta.block_lens
    remote_ssm_sizes = remote_meta.ssm_sizes

    block_size_ratio = transfer_topo.block_size // remote_info.remote_block_size
    assert block_size_ratio == 1, (
        "Mamba 3-read transfer with block_size_ratio != 1 "
        f"is not tested. Got {block_size_ratio=}."
    )

    m = _compute_tp_mapping(
        tp_rank,
        tp_size,
        remote_tp_size,
        transfer_topo.is_mla,
        transfer_topo.total_num_kv_heads,
        group_spec_types,
    )

    # ---- FA regions ----
    fa_regions = _build_fa_regions(
        block_len_per_layer=block_len_per_layer,
        remote_block_lens=remote_block_lens,
        is_blocks_first=transfer_topo.is_kv_layout_blocks_first,
        block_size_ratio=block_size_ratio,
        num_attn_reads=len(m.source_ranks_per_group[0]),
        rank_offset_factor=m.rank_offset_factor,
        remote_num_blocks=remote_meta.num_blocks,
    )

    # ---- SSM regions ----
    effective_ratio = tp_size // remote_tp_size if tp_size >= remote_tp_size else 1
    local_offset = tp_rank % max(effective_ratio, 1)
    conv_size_remote = remote_ssm_sizes[0]
    ssm_num_blocks = remote_meta.num_blocks // remote_phys_ratio

    if tp_size >= remote_tp_size:
        conv_offsets = conv_decomp.remote_conv_offsets(
            local_offset,
            effective_ratio,
        )
        ssm_read_size = ssm_sizes[1]
    else:
        abs_ratio = remote_tp_size // tp_size
        xb_p = conv_decomp.x_bytes // abs_ratio
        bb_p = conv_decomp.b_bytes // abs_ratio
        conv_offsets = [
            (0, xb_p),
            (xb_p, bb_p),
            (xb_p + bb_p, bb_p),
        ]
        ssm_read_size = remote_ssm_sizes[1]

    ssm_regions: list[RegionPlan] = []
    for i in range(len(remote_block_lens)):
        page_stride = remote_block_lens[i] * remote_phys_ratio

        for off, sz in conv_offsets:
            ssm_regions.append(
                RegionPlan(
                    layer_idx=i,
                    descriptor_bytes=sz,
                    offset_in_page=off,
                    page_stride=page_stride,
                    num_blocks=ssm_num_blocks,
                )
            )

        ssm_regions.append(
            RegionPlan(
                layer_idx=i,
                descriptor_bytes=ssm_read_size,
                offset_in_page=conv_size_remote + local_offset * ssm_read_size,
                page_stride=page_stride,
                num_blocks=ssm_num_blocks,
            )
        )

    return EngineTransferPlan(
        fa_regions=tuple(fa_regions),
        ssm_regions=tuple(ssm_regions),
        group_spec_types=group_spec_types,
        source_ranks_per_group=m.source_ranks_per_group,
        all_source_ranks=m.all_source_ranks,
        rank_to_attention_slot=m.rank_to_attention_slot,
        remote_expansion_stride=remote_phys_ratio,
    )


# ======================================================================
# 4. Local descriptor building
# ======================================================================


def build_fa_local_descs(
    base_addresses: list[int],
    device_id: int,
    num_blocks: int,
    block_size_ratio: int,
    block_len_per_layer: list[int],
    is_blocks_first: bool,
) -> list[tuple[int, int, int]]:
    """Build FA local descriptors for NIXL registration."""
    result: list[tuple[int, int, int]] = []
    n_blocks = num_blocks * block_size_ratio
    for i, base_addr in enumerate(base_addresses):
        kv_block_len = (
            _get_kv_block_len(i, block_len_per_layer, is_blocks_first)
            // block_size_ratio
        )
        page_stride = block_len_per_layer[i] // block_size_ratio
        for block_id in range(n_blocks):
            result.append(
                (
                    base_addr + block_id * page_stride,
                    kv_block_len,
                    device_id,
                )
            )
        if is_blocks_first:
            second_split = _get_kv_block_len(
                i,
                block_len_per_layer,
                is_blocks_first,
            )
            for block_id in range(n_blocks):
                v_addr = base_addr + block_id * page_stride + kv_block_len
                result.append((v_addr, second_split, device_id))
    return result


def build_mamba_local_descs(
    base_addresses: list[int],
    block_len_per_layer: list[int],
    logical_num_blocks: int,
    block_size_ratio: int,
    device_id: int,
    conv_decomp: MambaConvSplitInfo,
    ssm_sizes: tuple[int, int],
    physical_blocks_per_logical: int,
) -> list[tuple[int, int, int]]:
    """Build 4 SSM descriptor regions (x, B, C, ssm) per layer."""
    assert block_size_ratio == 1, (
        "Mamba 3-read transfer with block_size_ratio != 1 "
        f"is not tested. Got {block_size_ratio=}."
    )
    conv_offsets = conv_decomp.local_conv_offsets
    conv_size, ssm_size = ssm_sizes
    n_blocks = logical_num_blocks * block_size_ratio
    phys_ratio = physical_blocks_per_logical

    result: list[tuple[int, int, int]] = []
    for i, base_addr in enumerate(base_addresses):
        page_stride = block_len_per_layer[i] // block_size_ratio * phys_ratio
        for off, sz in conv_offsets:
            for blk in range(n_blocks):
                result.append(
                    (
                        base_addr + blk * page_stride + off,
                        sz,
                        device_id,
                    )
                )
        for blk in range(n_blocks):
            result.append(
                (
                    base_addr + blk * page_stride + conv_size,
                    ssm_size,
                    device_id,
                )
            )
    return result
