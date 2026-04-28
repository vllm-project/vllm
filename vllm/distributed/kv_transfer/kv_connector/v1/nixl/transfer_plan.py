# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Plan-based transfer design for NIXL connector.

Data structures, plan generators, and local descriptor builders
for NIXL KV cache transfers.
"""

from __future__ import annotations

import logging
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

logger = logging.getLogger(__name__)

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
    """Geometry for one descriptor region.

    Everything needed to build NIXL descriptors and compute descriptor
    IDs is baked in.  The caller plugs in ``base_addr`` and
    ``device_id`` when constructing the final descriptor tuples.

    When ``descs_per_block > 1``, each physical block produces multiple
    NIXL descriptors.  This happens when the remote page is larger than
    the local page (e.g. Gemma4 2p4d where P page = 65536 bytes,
    D page = 32768 bytes → ``descs_per_block = 2``).  Each descriptor
    covers one local-page-sized chunk of the remote block.
    """

    layer_idx: int

    # Descriptor geometry
    descriptor_bytes: int
    offset_in_page: int
    page_stride: int
    num_blocks: int

    # How many NIXL descriptors to register per physical block.
    # Default 1 (one desc per block).  When the remote page is N times
    # larger than local, set to N so each block produces N descriptors.
    descs_per_block: int = 1
    # Byte offset between consecutive descriptors within the same block.
    desc_stride_bytes: int = 0


@dataclass(frozen=True)
class EngineTransferPlan:
    """Complete transfer plan for one remote engine.

    Generated once during handshake.  Regions are split into
    ``fa_regions`` and ``ssm_regions`` matching the descriptor
    handle layout.

    Per-group HeteroTP fields enable models where different attention
    groups have different transfer behaviors (e.g. Gemma4 SWA + FA).
    """

    # --- Core regions (descriptor handle order) ---
    fa_regions: tuple[RegionPlan, ...]
    ssm_regions: tuple[RegionPlan, ...]

    # Per-group KVCacheSpec type — used for descriptor indexing.
    group_spec_types: tuple[type[KVCacheSpec], ...]

    # Per-group ordered source ranks. Position = local piece index.
    source_ranks_per_group: tuple[tuple[int, ...], ...]

    # Superset of all source ranks (union of all groups).
    all_source_ranks: tuple[int, ...]

    # Per-group head slot mapping.  Each dict maps source rank → slot.
    # Per-group because different groups can have different num_kv_heads,
    # leading to different head-to-slot assignments.
    # Example: Gemma4 has SWA K=8 and FA K=2; at 4p2d these would
    # produce genuinely different slot mappings.
    rank_to_attention_slot: tuple[dict[int, int], ...]

    # Stride for expanding remote logical block IDs to kernel block IDs.
    remote_expansion_stride: int

    # --- HeteroTP per-group fields (e.g. Gemma4 SWA + FA) ---
    # Active only when remote_to_local_page_ratio > 1.
    # For Dense/Mamba (ratio=1), these are unused and default to empty.

    # remote_page_size_bytes / local_page_size_bytes.
    # Gemma4 2p4d: 65536 / 32768 = 2.
    remote_to_local_page_ratio: int = 1

    # Per-group: how many local (D) blocks correspond to one remote (P)
    # block. Computed as remote_block_size / local_block_size per group.
    # Gemma4 2p4d: SWA = 16/16 = 1, FA = 32/16 = 2.
    local_blocks_per_remote_block: tuple[int, ...] = ()

    # Per-group: which descriptor offset to read from a multi-descriptor
    # remote block (for head-split groups where local reads a portion).
    # Gemma4 2p4d rank 0: SWA = 0 (first half), FA = 0 (unused, reads all).
    # Gemma4 2p4d rank 1: SWA = 1 (second half), FA = 0.
    remote_desc_offset_per_group: tuple[int, ...] = ()

    # --- Gather-read fields (local page > remote page, e.g. 4p2d FA) ---
    # When D pages are larger than P pages, each local block is registered
    # as multiple NIXL descriptors matching the remote block size.

    # local_page_size_bytes / remote_page_size_bytes.
    # 4p2d Gemma4: 65536 / 32768 = 2.  1 when no gather-read.
    local_to_remote_page_ratio: int = 1

    # Per-group: how many remote blocks fill one local block.
    # FA in 4p2d: D_tpb / P_tpb = 32 / 16 = 2.
    remote_blocks_per_local_block: tuple[int, ...] = ()

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
    assert len(remote_block_lens) == len(block_len_per_layer), (
        f"Layer count mismatch: remote has {len(remote_block_lens)} layers "
        f"but local has {len(block_len_per_layer)}"
    )
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

    tp_mapping = _compute_tp_mapping(
        transfer_topo.tp_rank,
        transfer_topo.tp_size,
        remote_info.remote_tp_size,
        transfer_topo.is_mla,
        transfer_topo.total_num_kv_heads,
        group_spec_types=group_spec_types,
    )

    num_attn_reads = next(
        len(ranks)
        for t, ranks in zip(group_spec_types, tp_mapping.source_ranks_per_group)
        if _is_attention_spec(t)
    )
    fa_regions = _build_fa_regions(
        block_len_per_layer=block_len_per_layer,
        remote_block_lens=remote_meta.block_lens,
        is_blocks_first=transfer_topo.is_kv_layout_blocks_first,
        block_size_ratio=block_size_ratio,
        num_attn_reads=num_attn_reads,
        rank_offset_factor=tp_mapping.rank_offset_factor,
        remote_num_blocks=remote_meta.num_blocks,
    )

    return EngineTransferPlan(
        fa_regions=tuple(fa_regions),
        ssm_regions=(),
        group_spec_types=group_spec_types,
        source_ranks_per_group=tp_mapping.source_ranks_per_group,
        all_source_ranks=tp_mapping.all_source_ranks,
        rank_to_attention_slot=(tp_mapping.rank_to_attention_slot,),
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

    tp_mapping = _compute_tp_mapping(
        tp_rank,
        tp_size,
        remote_tp_size,
        transfer_topo.is_mla,
        transfer_topo.total_num_kv_heads,
        group_spec_types,
    )

    # ---- FA regions ----
    num_attn_reads = next(
        len(ranks)
        for t, ranks in zip(group_spec_types, tp_mapping.source_ranks_per_group)
        if _is_attention_spec(t)
    )
    fa_regions = _build_fa_regions(
        block_len_per_layer=block_len_per_layer,
        remote_block_lens=remote_block_lens,
        is_blocks_first=transfer_topo.is_kv_layout_blocks_first,
        block_size_ratio=block_size_ratio,
        num_attn_reads=num_attn_reads,
        rank_offset_factor=tp_mapping.rank_offset_factor,
        remote_num_blocks=remote_meta.num_blocks,
    )

    # ---- SSM regions ----
    effective_ratio = tp_size // remote_tp_size if tp_size >= remote_tp_size else 1
    local_offset = tp_rank % max(effective_ratio, 1)
    conv_size_remote = remote_ssm_sizes[0]
    ssm_num_blocks = remote_meta.num_blocks // remote_phys_ratio

    # Mamba conv state is always TP-sharded, even when attention KV
    # is replicated (num_kv_heads < tp_size).
    if tp_size >= remote_tp_size:
        # D_TP >= P_TP: P page is larger, D reads its slice.
        conv_offsets = conv_decomp.remote_conv_offsets(
            local_offset,
            effective_ratio,
        )
        ssm_read_size = ssm_sizes[1]
    else:
        # NOTE (ZhanqiuHu): P_TP > D_TP, so P pages are smaller
        # than D's.  conv_decomp has D-sized dimensions, but we
        # need P-sized offsets.  Scale down by abs_ratio.
        abs_ratio = remote_tp_size // tp_size
        xb_p = conv_decomp.x_bytes // abs_ratio
        bb_p = conv_decomp.b_bytes // abs_ratio
        conv_offsets = [
            (0, xb_p),
            (xb_p, bb_p),
            (xb_p + bb_p, bb_p),
        ]
        ssm_read_size = remote_ssm_sizes[1]

    # NOTE (ZhanqiuHu): use per-layer block_lens[i], not [0],
    # in case block lengths vary across layers (e.g. MLA).
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

    n_groups = len(group_spec_types)
    return EngineTransferPlan(
        fa_regions=tuple(fa_regions),
        ssm_regions=tuple(ssm_regions),
        group_spec_types=group_spec_types,
        source_ranks_per_group=tp_mapping.source_ranks_per_group,
        all_source_ranks=tp_mapping.all_source_ranks,
        rank_to_attention_slot=(tp_mapping.rank_to_attention_slot,) * n_groups,
        remote_expansion_stride=remote_phys_ratio,
    )


def generate_gemma4_plan(
    *,
    transfer_topo: TransferTopology,
    block_len_per_layer: list[int],
    remote_info: EngineTransferInfo,
    remote_meta: NixlAgentMetadata,
    group_spec_types: tuple[type[KVCacheSpec], ...],
    total_num_kv_heads_per_group: tuple[int, ...],
    local_tokens_per_block: tuple[int, ...],
    remote_tokens_per_block: tuple[int, ...],
) -> EngineTransferPlan:
    """Generate transfer plan for Gemma4-style heterogeneous attention.

    Gemma4 has multiple attention groups (SWA, FA) with different
    ``total_num_kv_heads`` and ``head_dim``.  With page unification and
    HMA, all groups share physical memory pools.  This generator:

    1. Calls ``_compute_tp_mapping`` per group with group-specific K.
    2. Handles both **split-read** (remote page > local page, e.g. 2p4d)
       and **gather-read** (local page > remote page, e.g. 4p2d).
    3. Encodes per-group transfer behavior via
       ``local_blocks_per_remote_block`` / ``remote_blocks_per_local_block``
       and ``remote_desc_offset_per_group``.

    Split-read (P_page > D_page): each remote block → multiple descriptors.
    Gather-read (D_page > P_page): each local block → multiple descriptors.
    """
    tp_rank = transfer_topo.tp_rank
    tp_size = transfer_topo.tp_size
    remote_tp_size = remote_info.remote_tp_size
    is_mla = transfer_topo.is_mla
    is_blocks_first = transfer_topo.is_kv_layout_blocks_first
    n_groups = len(group_spec_types)

    local_page = block_len_per_layer[0]
    remote_page = remote_meta.block_lens[0]

    if remote_page >= local_page:
        split_page_ratio = remote_page // local_page
        gather_page_ratio = 1
    else:
        split_page_ratio = 1
        gather_page_ratio = local_page // remote_page

    blocks_per_remote: list[int] = []
    remote_blocks_per_local: list[int] = []
    remote_desc_offset: list[int] = []

    source_ranks_all: list[tuple[int, ...]] = []
    rank_to_slot_all: list[dict[int, int]] = []

    for g in range(n_groups):
        r_tpb = remote_tokens_per_block[g]
        l_tpb = local_tokens_per_block[g]

        if r_tpb >= l_tpb:
            blocks_per_remote.append(r_tpb // l_tpb)
            remote_blocks_per_local.append(1)
        else:
            blocks_per_remote.append(1)
            remote_blocks_per_local.append(l_tpb // r_tpb)

        K_g = total_num_kv_heads_per_group[g]
        m_g = _compute_tp_mapping(
            tp_rank,
            tp_size,
            remote_tp_size,
            is_mla,
            K_g,
            (group_spec_types[g],),
        )
        source_ranks_all.append(m_g.source_ranks_per_group[0])
        rank_to_slot_all.append(m_g.rank_to_attention_slot)

        # Head-split groups (split-read only): rank_offset selects descriptor.
        if blocks_per_remote[-1] == 1 and split_page_ratio > 1:
            remote_desc_offset.append(m_g.rank_offset_factor)
        else:
            remote_desc_offset.append(0)

    all_ranks: set[int] = set()
    for ranks in source_ranks_all:
        all_ranks.update(ranks)
    all_source_ranks = tuple(sorted(all_ranks))

    # --- Diagnostic logging for HeteroTP plan ---
    logger.info(
        "[HeteroTP Plan] tp_rank=%d, tp_size=%d, remote_tp_size=%d, "
        "local_page=%d, remote_page=%d, "
        "split_page_ratio=%d, gather_page_ratio=%d",
        tp_rank,
        tp_size,
        remote_tp_size,
        local_page,
        remote_page,
        split_page_ratio,
        gather_page_ratio,
    )
    for g in range(n_groups):
        logger.info(
            "[HeteroTP Plan] group=%d kind=%s: K=%d, "
            "local_tpb=%d, remote_tpb=%d, "
            "blocks_per_remote=%d, remote_blocks_per_local=%d, "
            "desc_offset=%d, source_ranks=%s, slot_map=%s",
            g,
            group_kinds[g].value,
            total_num_kv_heads_per_group[g],
            local_tokens_per_block[g],
            remote_tokens_per_block[g],
            blocks_per_remote[g],
            remote_blocks_per_local[g],
            remote_desc_offset[g],
            source_ranks_all[g],
            rank_to_slot_all[g],
        )

    # HMA: one K pool (+ optional V pool) shared by all groups.
    fa_regions: list[RegionPlan] = []
    for i in range(len(remote_meta.block_lens)):
        local_block_len = _get_kv_block_len(
            i,
            block_len_per_layer,
            is_blocks_first,
        )
        page_stride = remote_meta.block_lens[i]

        if split_page_ratio > 1:
            # Split-read: remote blocks produce descriptors of local page size
            desc_bytes = local_block_len
            descs_per_block = split_page_ratio
            desc_stride = local_block_len
        elif gather_page_ratio > 1:
            # Gather-read: standard remote descs at remote page size
            remote_block_len = _get_kv_block_len(
                i, remote_meta.block_lens, is_blocks_first
            )
            desc_bytes = remote_block_len
            descs_per_block = 1
            desc_stride = 0
        else:
            desc_bytes = local_block_len
            descs_per_block = 1
            desc_stride = 0

        fa_regions.append(
            RegionPlan(
                kind=RegionKind.FA_K,
                layer_idx=i,
                descriptor_bytes=desc_bytes,
                offset_in_page=0,
                page_stride=page_stride,
                num_blocks=remote_meta.num_blocks,
                descs_per_block=descs_per_block,
                desc_stride_bytes=desc_stride,
            )
        )

        if is_blocks_first:
            fa_regions.append(
                RegionPlan(
                    kind=RegionKind.FA_V,
                    layer_idx=i,
                    descriptor_bytes=desc_bytes,
                    offset_in_page=page_stride // 2,
                    page_stride=page_stride,
                    num_blocks=remote_meta.num_blocks,
                    descs_per_block=descs_per_block,
                    desc_stride_bytes=desc_stride,
                )
            )

    return EngineTransferPlan(
        fa_regions=tuple(fa_regions),
        ssm_regions=(),
        group_spec_types=group_spec_types,
        source_ranks_per_group=tuple(source_ranks_all),
        all_source_ranks=all_source_ranks,
        rank_to_attention_slot=tuple(rank_to_slot_all),
        remote_expansion_stride=1,
        remote_to_local_page_ratio=split_page_ratio,
        local_blocks_per_remote_block=tuple(blocks_per_remote),
        remote_desc_offset_per_group=tuple(remote_desc_offset),
        local_to_remote_page_ratio=gather_page_ratio,
        remote_blocks_per_local_block=tuple(remote_blocks_per_local),
    )


# ======================================================================
# 4. Local descriptor building
# ======================================================================


def _remap_remote_blocks_to_desc_ids(
    plan: EngineTransferPlan,
    remote_block_ids: BlockIds,
    local_block_ids: BlockIds,
) -> tuple[BlockIds, BlockIds]:
    """Convert remote block IDs into descriptor-level indices.

    When ``remote_to_local_page_ratio > 1``, each remote physical block
    is registered as multiple descriptors (one per local-page-sized
    chunk).  This function converts remote block IDs into the
    descriptor index space so that ``_compute_desc_ids_from_plan`` can
    look up the correct descriptors.

    Two per-group cases:

    * **Multi-block** (``local_blocks_per_remote_block > 1``, e.g. FA):
      One remote block covers multiple local blocks.
      Remote block ``b`` → descriptor indices
      ``[b*ratio, b*ratio+1, ..., b*ratio+(n-1)]``.
      Example: FA block 10, ratio=2 → desc indices [20, 21].

    * **Head-split** (``local_blocks_per_remote_block == 1``, e.g. SWA):
      Local reads one specific chunk of the remote block.
      Remote block ``b`` → descriptor index
      ``b*ratio + remote_desc_offset_per_group[g]``.
      Example: SWA block 10, ratio=2, offset=1 → desc index 21.

    Local block IDs are returned unchanged.
    """
    if plan.remote_to_local_page_ratio <= 1:
        return remote_block_ids, local_block_ids

    ratio = plan.remote_to_local_page_ratio
    num_groups = len(remote_block_ids)
    new_remote: list[list[int]] = []
    new_local: list[list[int]] = []

    for g in range(num_groups):
        n_local = plan.local_blocks_per_remote_block[g]
        r_ids = list(remote_block_ids[g])
        l_ids = list(local_block_ids[g])

        if n_local > 1:
            remapped: list[int] = []
            for b in r_ids:
                remapped.extend(b * ratio + s for s in range(n_local))
            new_remote.append(remapped)
        else:
            idx = plan.remote_desc_offset_per_group[g]
            new_remote.append([b * ratio + idx for b in r_ids])

        new_local.append(l_ids)

    return new_remote, new_local


def _build_gather_read_specs(
    plan: EngineTransferPlan,
    local_block_ids: BlockIds,
    remote_block_ids: BlockIds,
) -> list[ReadSpec]:
    """Build read specs for gather-read (local page > remote page).

    In gather-read, each local block is registered as multiple NIXL
    descriptors (``descs_per_block`` in ``RegionPlan``), each matching
    the remote block byte size.  Each rank's read targets specific
    local descriptor IDs:

    * **Gather groups** (``remote_blocks_per_local_block > 1``, e.g. FA):
      N remote blocks fill one local block.
      Local block ``b`` → descriptor IDs
      ``[b*gather_ratio, b*gather_ratio+1, ..., b*gather_ratio+(N-1)]``.
      Remote block IDs are kept as-is (one remote block = one
      remote descriptor).  The matched-length invariant
      ``len(local_desc_ids) == len(remote_block_ids)`` must hold;
      it is enforced by an assertion after construction.

    * **Concat groups** (``remote_blocks_per_local_block == 1``, e.g. SWA):
      Each rank writes to a specific descriptor within the local block.
      Local block ``b`` → descriptor ID
      ``b*gather_ratio + rank_slot``.
    """
    gather_ratio = plan.local_to_remote_page_ratio
    num_groups = len(local_block_ids)

    def _pair_gather_group(
        g_local_block_ids: list[int],
        g_remote_block_ids: list[int],
        remote_blocks_per_local: int,
    ) -> tuple[list[int], list[int]]:
        """Pair local descriptor IDs with remote block IDs for a gather group.

        With HMA, all groups receive the same block ID list.  For gather
        groups (``remote_blocks_per_local > 1``), every
        ``remote_blocks_per_local`` consecutive remote blocks map to
        descriptors of a single local block:

            local block b, remote blocks [r0, r1]  →
                local desc  b*gather_ratio + 0  paired with  r0
                local desc  b*gather_ratio + 1  paired with  r1

        When the remote block count is not a multiple of
        ``remote_blocks_per_local``, the remainder fills the first
        descriptors of the next local block (partial fill).

        Returns matched-length lists:
            (local_desc_ids, paired_remote_block_ids)
        """
        n_local = len(g_local_block_ids)
        n_remote = len(g_remote_block_ids)
        n_full = min(n_remote // remote_blocks_per_local, n_local)
        remainder_remote = n_remote - n_full * remote_blocks_per_local

        local_desc_ids: list[int] = []
        paired_remote: list[int] = []

        for i in range(n_full):
            b = g_local_block_ids[i]
            for s in range(remote_blocks_per_local):
                local_desc_ids.append(b * gather_ratio + s)
                paired_remote.append(
                    g_remote_block_ids[i * remote_blocks_per_local + s]
                )

        if remainder_remote > 0 and n_full < n_local:
            b = g_local_block_ids[n_full]
            base = n_full * remote_blocks_per_local
            for s in range(remainder_remote):
                local_desc_ids.append(b * gather_ratio + s)
                paired_remote.append(g_remote_block_ids[base + s])

        return local_desc_ids, paired_remote

    specs: list[ReadSpec] = []

    for rank in plan.all_source_ranks:
        rank_local: list[list[int]] = []
        rank_remote: list[list[int]] = []

        for g in range(num_groups):
            if rank not in plan.source_ranks_per_group[g]:
                rank_local.append([])
                rank_remote.append([])
                continue

            n_remote_per_local = plan.remote_blocks_per_local_block[g]

            if n_remote_per_local > 1:
                g_local, g_remote = _pair_gather_group(
                    local_block_ids[g],
                    remote_block_ids[g],
                    n_remote_per_local,
                )
                rank_local.append(g_local)
                rank_remote.append(g_remote)
            else:
                slot = plan.rank_to_attention_slot[g].get(rank, 0)
                l_ids = local_block_ids[g]
                r_ids = remote_block_ids[g]
                n = min(len(l_ids), len(r_ids))
                rank_local.append(
                    [
                        l_ids[i] * gather_ratio + slot
                        for i in range(len(l_ids) - n, len(l_ids))
                    ]
                )
                rank_remote.append(list(r_ids[len(r_ids) - n :]))

        for g in range(num_groups):
            assert len(rank_local[g]) == len(rank_remote[g]), (
                f"Gather-read length mismatch: group={g}, rank={rank}, "
                f"n_local_descs={len(rank_local[g])}, "
                f"n_remote_blocks={len(rank_remote[g])}. "
                f"Each local descriptor must pair with exactly one "
                f"remote block ID."
            )

        specs.append(
            ReadSpec(
                remote_rank=rank,
                local_block_ids=rank_local,
                remote_block_ids=rank_remote,
            )
        )

    return specs


logger = logging.getLogger(__name__)



# ======================================================================
# 4. Local descriptor building
# ======================================================================


def build_fa_local_regions(
    num_blocks: int,
    block_size_ratio: int,
    block_len_per_layer: list[int],
    is_blocks_first: bool,
) -> list[RegionPlan]:
    """Build FA local region specs for NIXL registration."""
    regions: list[RegionPlan] = []
    n_blocks = num_blocks * block_size_ratio
    for i in range(len(block_len_per_layer)):
        kv_block_len = (
            _get_kv_block_len(i, block_len_per_layer, is_blocks_first)
            // block_size_ratio
        )
        page_stride = block_len_per_layer[i] // block_size_ratio
        regions.append(
            RegionPlan(
                layer_idx=i,
                descriptor_bytes=kv_block_len,
                offset_in_page=0,
                page_stride=page_stride,
                num_blocks=n_blocks,
            )
        )
        if is_blocks_first:
            second_split = _get_kv_block_len(
                i,
                block_len_per_layer,
                is_blocks_first,
            )
            regions.append(
                RegionPlan(
                    layer_idx=i,
                    descriptor_bytes=second_split,
                    offset_in_page=kv_block_len,
                    page_stride=page_stride,
                    num_blocks=n_blocks,
                )
            )
    return regions


def build_fa_local_descs_for_gather_read(
    base_addresses: list[int],
    device_id: int,
    num_blocks: int,
    block_len_per_layer: list[int],
    is_blocks_first: bool,
    gather_page_ratio: int,
) -> list[tuple[int, int, int]]:
    """Build FA local descriptors for gather-read.

    Each local block produces ``gather_page_ratio`` descriptors, each
    covering ``kv_block_len // gather_page_ratio`` bytes.  This allows
    NIXL to pair each local descriptor with a remote descriptor of
    matching size (the remote's natural page size).
    """
    result: list[tuple[int, int, int]] = []
    for i, base_addr in enumerate(base_addresses):
        kv_block_len = _get_kv_block_len(i, block_len_per_layer, is_blocks_first)
        page_stride = block_len_per_layer[i]
        desc_bytes = kv_block_len // gather_page_ratio

        for block_id in range(num_blocks):
            blk_addr = base_addr + block_id * page_stride
            for s in range(gather_page_ratio):
                result.append((blk_addr + s * desc_bytes, desc_bytes, device_id))

        if is_blocks_first:
            v_desc_bytes = kv_block_len // gather_page_ratio
            for block_id in range(num_blocks):
                v_blk_addr = base_addr + block_id * page_stride + kv_block_len
                for s in range(gather_page_ratio):
                    result.append(
                        (v_blk_addr + s * v_desc_bytes, v_desc_bytes, device_id)
                    )

    return result


def build_mamba_local_regions(
    block_len_per_layer: list[int],
    logical_num_blocks: int,
    block_size_ratio: int,
    conv_decomp: MambaConvSplitInfo,
    ssm_sizes: tuple[int, int],
    physical_blocks_per_logical: int,
) -> list[RegionPlan]:
    """Build 4 SSM region specs (x, B, C, ssm) per layer."""
    assert block_size_ratio == 1, (
        "Mamba 3-read transfer with block_size_ratio != 1 "
        f"is not tested. Got {block_size_ratio=}."
    )
    conv_offsets = conv_decomp.local_conv_offsets
    conv_size, ssm_size = ssm_sizes
    n_blocks = logical_num_blocks * block_size_ratio
    phys_ratio = physical_blocks_per_logical

    regions: list[RegionPlan] = []
    for i in range(len(block_len_per_layer)):
        page_stride = block_len_per_layer[i] // block_size_ratio * phys_ratio
        for off, sz in conv_offsets:
            regions.append(
                RegionPlan(
                    layer_idx=i,
                    descriptor_bytes=sz,
                    offset_in_page=off,
                    page_stride=page_stride,
                    num_blocks=n_blocks,
                )
            )
        # SSM temporal state follows the conv state.
        regions.append(
            RegionPlan(
                layer_idx=i,
                descriptor_bytes=ssm_size,
                offset_in_page=conv_size,
                page_stride=page_stride,
                num_blocks=n_blocks,
            )
        )
    return regions
