# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Plan-based transfer design for NIXL connector.

Instead of an ABC hierarchy with duplicated Dense/Mamba implementations,
we pre-generate a flat transfer plan per remote engine during handshake.
All downstream operations become generic plan executors with zero model
branching.

Architecture:
    1. Plan generators (generate_dense_plan, generate_mamba_plan)
       — the ONLY model-specific code.
    2. Generic executors (build_remote_descs_from_plan, etc.)
       — consume plans without model branching.
    3. Visualization (visualize_plan).
"""

from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from vllm.distributed.kv_transfer.kv_connector.utils import BlockIds
from vllm.distributed.kv_transfer.kv_connector.v1.ssm_conv_transfer_utils import (
    MambaConvSplitInfo,
)

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


class GroupKind(enum.Enum):
    """KV cache group type for transfer purposes.

    Used by ``EngineTransferPlan`` and block expansion functions to
    dispatch per-group behavior without model-specific branching.
    """

    FA = "fa"
    SWA = "swa"
    MAMBA = "mamba"
    GDN = "gdn"

    @property
    def is_attention(self) -> bool:
        """FA and SWA both need block expansion and standard descriptors."""
        return self in (GroupKind.FA, GroupKind.SWA)

    @property
    def is_ssm(self) -> bool:
        """MAMBA and GDN have state descriptors instead of KV pages."""
        return self in (GroupKind.MAMBA, GroupKind.GDN)


class RegionKind(enum.Enum):
    """Descriptor region type.  Used for visualization/debugging only;
    executors never branch on this value."""

    FA_K = "fa_k"
    FA_V = "fa_v"
    SSM_CONV_X = "ssm_conv_x"
    SSM_CONV_B = "ssm_conv_b"
    SSM_CONV_C = "ssm_conv_c"
    SSM_STATE = "ssm_state"


@dataclass(frozen=True)
class RegionPlan:
    """Pre-computed plan for one descriptor region.

    Everything needed to build NIXL descriptors and compute descriptor
    IDs is baked in — no runtime model branching.  The executor plugs
    in per-rank ``base_addr`` and ``device_id`` from NixlAgentMetadata.
    """

    kind: RegionKind
    layer_idx: int

    # Descriptor geometry
    descriptor_bytes: int
    offset_in_page: int
    page_stride: int
    num_blocks: int

    # Block ID expansion (HMA / kernel block mismatch)
    physical_per_logical: int


@dataclass(frozen=True)
class EngineTransferPlan:
    """Complete transfer plan for one remote engine.

    Generated once during handshake.  Stored alongside (or replacing)
    ``EngineTransferInfo`` on ``TransferTopology``.

    Regions are split into ``fa_regions`` and ``ssm_regions`` matching
    the descriptor handle layout: [FA descriptors | SSM descriptors].
    ``group_kinds`` maps each kv_cache_group to its type for descriptor
    indexing.  ``source_ranks_per_group`` encodes which ranks read each
    group — executors use this instead of group_kinds for rank routing.
    """

    # Regions in descriptor handle order
    fa_regions: tuple[RegionPlan, ...]
    ssm_regions: tuple[RegionPlan, ...]

    # Per-group geometric properties (worker-facing, model-agnostic)
    physical_per_logical: tuple[int, ...]

    # Per-group type — used only for descriptor indexing (save path).
    group_kinds: tuple[GroupKind, ...]

    # Per-group ordered source ranks. Position = local piece index.
    source_ranks_per_group: tuple[tuple[int, ...], ...]

    # Superset of all source ranks (union of all groups).
    all_source_ranks: tuple[int, ...]

    # Maps each source rank to its FA head slot index.
    rank_to_attention_slot: dict[int, int]

    # Remote engine facts (needed by worker at read time)
    remote_tp_size: int
    remote_block_size: int
    remote_block_len: int
    remote_physical_blocks_per_logical: int

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
    group_kinds: tuple[GroupKind, ...],
) -> TPMapping:
    """Build the complete local-to-remote TP mapping.

    Computes source ranks, head slot assignments, and the rank offset
    factor in a single pass.  Both generators call this and unpack.
    """
    K = total_num_kv_heads

    # --- Attention source ranks ---
    if is_mla:
        attn_ranks = [0]
    elif tp_size >= remote_tp_size:
        attn_ranks = [tp_rank * remote_tp_size // tp_size]
    else:
        # P > D: one local rank reads from multiple remote ranks.
        # GQA dedup: when K < remote_tp_size, several remote ranks
        # hold the same KV head.  np.unique keeps only the first
        # rank per unique head so we don't issue redundant reads.
        abs_tp = remote_tp_size // tp_size
        start = tp_rank * abs_tp
        heads = np.arange(start, start + abs_tp) * K // remote_tp_size
        _, unique_idx = np.unique(heads, return_index=True)
        attn_ranks = (start + np.sort(unique_idx)).tolist()

    # --- All source ranks (expand for SSM if needed) ---
    has_ssm = any(k.is_ssm for k in group_kinds)
    if not has_ssm or tp_size >= remote_tp_size:
        all_ranks = list(attn_ranks)
    else:
        abs_tp = remote_tp_size // tp_size
        if abs_tp > len(attn_ranks):
            all_ranks = list(
                range(
                    tp_rank * abs_tp,
                    (tp_rank + 1) * abs_tp,
                )
            )
        else:
            all_ranks = list(attn_ranks)

    # --- Per-group ordered source ranks ---
    attn_tuple = tuple(attn_ranks)
    all_tuple = tuple(all_ranks)
    source_ranks_per_group = tuple(
        all_tuple if k.is_ssm else attn_tuple for k in group_kinds
    )

    # --- Attention head slots ---
    head_to_slot: dict[int, int] = {}
    for i, r in enumerate(attn_ranks):
        head_to_slot[r * K // remote_tp_size] = i
    rank_to_attention_slot = {
        r: head_to_slot.get(r * K // remote_tp_size, 0) for r in all_ranks
    }

    # --- Rank offset factor ---
    if is_mla or tp_size <= remote_tp_size:
        rank_offset_factor = 0
    elif tp_size > K:
        local_head = tp_rank * K // tp_size
        p_start = attn_ranks[0] * K // remote_tp_size
        rank_offset_factor = local_head - p_start
    else:
        rank_offset_factor = tp_rank % (tp_size // remote_tp_size)

    return TPMapping(
        source_ranks_per_group=source_ranks_per_group,
        all_source_ranks=all_tuple,
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
    remote_physical_blocks_per_logical: int,
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
                kind=RegionKind.FA_K,
                layer_idx=i,
                descriptor_bytes=k_desc_bytes,
                offset_in_page=rank_offset,
                page_stride=page_stride,
                num_blocks=remote_num_blocks,
                physical_per_logical=remote_physical_blocks_per_logical,
            )
        )

        if is_blocks_first:
            v_desc_bytes = local_block_len // num_attn_reads
            fa_regions.append(
                RegionPlan(
                    kind=RegionKind.FA_V,
                    layer_idx=i,
                    descriptor_bytes=v_desc_bytes,
                    offset_in_page=rank_offset + page_stride // 2,
                    page_stride=page_stride,
                    num_blocks=remote_num_blocks,
                    physical_per_logical=remote_physical_blocks_per_logical,
                )
            )

    return fa_regions


# ======================================================================
# 3. Plan generators — the ONLY model-specific code
# ======================================================================


def generate_dense_plan(
    *,
    tp_rank: int,
    tp_size: int,
    is_mla: bool,
    total_num_kv_heads: int,
    is_blocks_first: bool,
    block_len_per_layer: list[int],
    block_size: int,
    remote_tp_size: int,
    remote_block_size: int,
    remote_num_blocks: int,
    remote_block_lens: list[int],
    remote_physical_blocks_per_logical: int,
    local_physical_blocks_per_logical: int,
) -> EngineTransferPlan:
    """Generate transfer plan for dense (FA-only) models."""
    block_size_ratio = block_size // remote_block_size

    m = _compute_tp_mapping(
        tp_rank,
        tp_size,
        remote_tp_size,
        is_mla,
        total_num_kv_heads,
        group_kinds=(GroupKind.FA,),
    )

    fa_regions = _build_fa_regions(
        block_len_per_layer=block_len_per_layer,
        remote_block_lens=remote_block_lens,
        is_blocks_first=is_blocks_first,
        block_size_ratio=block_size_ratio,
        num_attn_reads=len(m.source_ranks_per_group[0]),
        rank_offset_factor=m.rank_offset_factor,
        remote_num_blocks=remote_num_blocks,
        remote_physical_blocks_per_logical=remote_physical_blocks_per_logical,
    )

    return EngineTransferPlan(
        fa_regions=tuple(fa_regions),
        ssm_regions=(),
        physical_per_logical=(remote_physical_blocks_per_logical,),
        group_kinds=(GroupKind.FA,),
        source_ranks_per_group=m.source_ranks_per_group,
        all_source_ranks=m.all_source_ranks,
        rank_to_attention_slot=m.rank_to_attention_slot,
        remote_tp_size=remote_tp_size,
        remote_block_size=remote_block_size,
        remote_block_len=remote_block_lens[0],
        remote_physical_blocks_per_logical=remote_physical_blocks_per_logical,
        remote_expansion_stride=local_physical_blocks_per_logical,
    )


def generate_mamba_plan(
    *,
    tp_rank: int,
    tp_size: int,
    is_mla: bool,
    total_num_kv_heads: int,
    is_blocks_first: bool,
    block_len_per_layer: list[int],
    block_size: int,
    remote_tp_size: int,
    remote_block_size: int,
    remote_num_blocks: int,
    remote_block_lens: list[int],
    remote_physical_blocks_per_logical: int,
    group_kinds: tuple[GroupKind, ...],
    conv_decomp: MambaConvSplitInfo,
    ssm_sizes: tuple[int, int],
    remote_ssm_sizes: tuple[int, int],
) -> EngineTransferPlan:
    """Generate transfer plan for hybrid Mamba (SSM + FA) models."""
    block_size_ratio = block_size // remote_block_size
    assert block_size_ratio == 1, (
        "Mamba 3-read transfer with block_size_ratio != 1 "
        f"is not tested. Got {block_size_ratio=}."
    )

    m = _compute_tp_mapping(
        tp_rank,
        tp_size,
        remote_tp_size,
        is_mla,
        total_num_kv_heads,
        group_kinds,
    )

    # ---- FA regions ----
    fa_regions = _build_fa_regions(
        block_len_per_layer=block_len_per_layer,
        remote_block_lens=remote_block_lens,
        is_blocks_first=is_blocks_first,
        block_size_ratio=block_size_ratio,
        num_attn_reads=len(m.source_ranks_per_group[0]),
        rank_offset_factor=m.rank_offset_factor,
        remote_num_blocks=remote_num_blocks,
        remote_physical_blocks_per_logical=remote_physical_blocks_per_logical,
    )

    # ---- SSM regions ----
    effective_ratio = tp_size // remote_tp_size if tp_size >= remote_tp_size else 1
    local_offset = tp_rank % max(effective_ratio, 1)
    conv_size_remote = remote_ssm_sizes[0]
    remote_ratio = remote_physical_blocks_per_logical
    ssm_num_blocks = remote_num_blocks // remote_ratio

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

    conv_kinds = [
        RegionKind.SSM_CONV_X,
        RegionKind.SSM_CONV_B,
        RegionKind.SSM_CONV_C,
    ]
    ssm_regions: list[RegionPlan] = []
    for i in range(len(remote_block_lens)):
        page_stride = remote_block_lens[i] * remote_ratio

        for kind, (off, sz) in zip(conv_kinds, conv_offsets):
            ssm_regions.append(
                RegionPlan(
                    kind=kind,
                    layer_idx=i,
                    descriptor_bytes=sz,
                    offset_in_page=off,
                    page_stride=page_stride,
                    num_blocks=ssm_num_blocks,
                    physical_per_logical=1,
                )
            )

        ssm_regions.append(
            RegionPlan(
                kind=RegionKind.SSM_STATE,
                layer_idx=i,
                descriptor_bytes=ssm_read_size,
                offset_in_page=conv_size_remote + local_offset * ssm_read_size,
                page_stride=page_stride,
                num_blocks=ssm_num_blocks,
                physical_per_logical=1,
            )
        )

    physical_per_logical_per_group = tuple(
        1 if k.is_ssm else remote_physical_blocks_per_logical for k in group_kinds
    )
    return EngineTransferPlan(
        fa_regions=tuple(fa_regions),
        ssm_regions=tuple(ssm_regions),
        physical_per_logical=physical_per_logical_per_group,
        group_kinds=group_kinds,
        source_ranks_per_group=m.source_ranks_per_group,
        all_source_ranks=m.all_source_ranks,
        rank_to_attention_slot=m.rank_to_attention_slot,
        remote_tp_size=remote_tp_size,
        remote_block_size=remote_block_size,
        remote_block_len=remote_block_lens[0],
        remote_physical_blocks_per_logical=remote_physical_blocks_per_logical,
        remote_expansion_stride=remote_physical_blocks_per_logical,
    )


# ======================================================================
# 4. Generic executors — identical for ALL models
# ======================================================================


def logical_to_kernel_block_ids(
    block_ids: BlockIds,
    physical_per_logical: tuple[int, ...],
) -> BlockIds:
    """Convert logical block IDs to kernel-level physical block IDs.

    Each group has its own ratio in ``physical_per_logical``.
    Groups with ratio == 1 are passed through unchanged.
    """
    if all(r == 1 for r in physical_per_logical):
        return block_ids
    result: list[list[int]] = []
    for i, group in enumerate(block_ids):
        ratio = physical_per_logical[i]
        if ratio == 1:
            result.append(group)
        else:
            arr = np.array(group).reshape(-1, 1)
            arange = np.arange(ratio).reshape(1, -1)
            result.append((arr * ratio + arange).flatten().tolist())
    return result


def build_remote_descs_from_plan(
    plan: EngineTransferPlan,
    nixl_agent_meta: NixlAgentMetadata,
) -> list[tuple[int, int, int]]:
    """Build (addr, len, dev_id) descriptor tuples from plan.

    Builds remote descriptors from a pre-computed plan.
    """
    result: list[tuple[int, int, int]] = []
    dev_id = nixl_agent_meta.device_id

    for region in plan.all_regions:
        base_addr = nixl_agent_meta.kv_caches_base_addr[region.layer_idx]
        for blk in range(region.num_blocks):
            addr = base_addr + blk * region.page_stride + region.offset_in_page
            result.append((addr, region.descriptor_bytes, dev_id))

    return result


def compute_desc_ids_from_plan(
    plan: EngineTransferPlan,
    block_ids: BlockIds,
    dst_num_blocks: int,
    block_size_ratio: float | None = None,
    physical_blocks_per_logical: int = 1,
) -> np.ndarray:
    """Compute NIXL descriptor IDs for given block IDs.

    Computes descriptor indices from a pre-computed plan.
    """
    num_fa_regions = len(plan.fa_regions)
    num_ssm_regions = len(plan.ssm_regions)

    num_blocks = dst_num_blocks
    if block_size_ratio is not None:
        num_blocks = int(num_blocks * block_size_ratio)
    ratio = physical_blocks_per_logical
    logical_blocks = num_blocks // ratio

    num_fa_descs = num_fa_regions * num_blocks

    all_descs: list[np.ndarray] = []
    for i, group in enumerate(block_ids):
        group_arr = np.asarray(group)
        if plan.group_kinds[i].is_attention:
            fa_region_ids = np.arange(num_fa_regions)[:, None]
            all_descs.append(
                (fa_region_ids * num_blocks + group_arr[None, :]).flatten()
            )
        elif plan.group_kinds[i].is_ssm:
            ssm_region_ids = np.arange(num_ssm_regions)[:, None]
            all_descs.append(
                (
                    ssm_region_ids * logical_blocks + group_arr[None, :] + num_fa_descs
                ).flatten()
            )
        else:
            raise ValueError(f"Unknown group kind {plan.group_kinds[i]} at index {i}")

    return np.concatenate(all_descs)


def compute_read_specs_from_plan(
    plan: EngineTransferPlan,
    local_block_ids: BlockIds,
    remote_block_ids: BlockIds,
) -> list[ReadSpec]:
    """Compute read specs from plan.

    For each source rank, includes only the groups whose
    source_ranks_per_group contains that rank.
    """
    num_groups = len(local_block_ids)
    return [
        ReadSpec(
            remote_rank=rank,
            local_block_ids=[
                list(local_block_ids[g])
                if rank in plan.source_ranks_per_group[g]
                else []
                for g in range(num_groups)
            ],
            remote_block_ids=[
                list(remote_block_ids[g])
                if rank in plan.source_ranks_per_group[g]
                else []
                for g in range(num_groups)
            ],
        )
        for rank in plan.all_source_ranks
    ]


def build_local_splits_from_plan(
    plan: EngineTransferPlan,
    src_blocks_data: list[tuple[int, int, int]],
    num_fa_descs: int,
) -> list[list[tuple[int, int, int]]]:
    """Build split handle data for P_TP > D_TP scenario.

    num_fa_descs is the boundary between FA and SSM descriptors.
    Split counts are derived from source_ranks_per_group lengths.
    FA uses rank_to_attention_slot for the slot offset;
    SSM uses the rank's positional index.
    """
    fa_num_splits = len(plan.source_ranks_per_group[0])

    has_ssm_descs = num_fa_descs < len(src_blocks_data)
    ssm_num_splits = len(plan.source_ranks_per_group[-1]) if has_ssm_descs else 0

    result: list[list[tuple[int, int, int]]] = []

    for p_idx, p_rank in enumerate(plan.all_source_ranks):
        fa_slot = plan.rank_to_attention_slot.get(p_rank, 0)

        handle: list[tuple[int, int, int]] = []
        for j, (addr, local_len, dev) in enumerate(src_blocks_data):
            if j < num_fa_descs:
                chunk = local_len // fa_num_splits
                handle.append((addr + fa_slot * chunk, chunk, dev))
            else:
                chunk = local_len // ssm_num_splits
                handle.append((addr + p_idx * chunk, chunk, dev))
        result.append(handle)

    return result


# ======================================================================
# 5. Local descriptor building (no plan needed — purely local geometry)
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


def build_local_descs(
    *,
    has_mamba: bool,
    conv_decomp: MambaConvSplitInfo | None,
    ssm_sizes: tuple[int, int],
    base_addresses: list[int],
    device_id: int,
    num_blocks: int,
    logical_num_blocks: int,
    block_size_ratio: int,
    block_len_per_layer: list[int],
    is_blocks_first: bool,
    physical_blocks_per_logical: int = 1,
) -> list[tuple[int, int, int]]:
    """Build local (src) descriptor tuples for NIXL registration."""
    fa_descs = build_fa_local_descs(
        base_addresses,
        device_id,
        num_blocks,
        block_size_ratio,
        block_len_per_layer,
        is_blocks_first,
    )
    if not has_mamba:
        return fa_descs
    assert conv_decomp is not None
    mamba_descs = build_mamba_local_descs(
        base_addresses,
        block_len_per_layer,
        logical_num_blocks,
        block_size_ratio,
        device_id,
        conv_decomp,
        ssm_sizes,
        physical_blocks_per_logical,
    )
    return fa_descs + mamba_descs


# ======================================================================
# 6. Visualization
# ======================================================================


def visualize_plan(plan: EngineTransferPlan) -> str:
    """Human-readable transfer plan for logging and debugging."""
    lines = [
        f"EngineTransferPlan(remote_tp={plan.remote_tp_size}, "
        f"remote_bs={plan.remote_block_size}):",
        f"  Source ranks: all={list(plan.all_source_ranks)}",
    ]
    total_descs = 0

    if plan.fa_regions:
        lines.append(f"  FA regions ({len(plan.fa_regions)}):")
        for idx, r in enumerate(plan.fa_regions):
            ratio_str = (
                f", p/l={r.physical_per_logical}" if r.physical_per_logical > 1 else ""
            )
            lines.append(
                f"    [{idx}] {r.kind.value:12s} L{r.layer_idx}  "
                f"{r.descriptor_bytes:6d}B x {r.num_blocks:4d} blks  "
                f"stride={r.page_stride:6d}  "
                f"off={r.offset_in_page:6d}"
                f"{ratio_str}"
            )
            total_descs += r.num_blocks

    if plan.ssm_regions:
        lines.append(f"  SSM regions ({len(plan.ssm_regions)}):")
        for idx, r in enumerate(plan.ssm_regions):
            lines.append(
                f"    [{idx}] {r.kind.value:12s} L{r.layer_idx}  "
                f"{r.descriptor_bytes:6d}B x {r.num_blocks:4d} blks  "
                f"stride={r.page_stride:6d}  "
                f"off={r.offset_in_page:6d}"
            )
            total_descs += r.num_blocks

    lines.append(f"  Groups: {[k.value for k in plan.group_kinds]}")
    lines.append(f"  Total descriptors: {total_descs}")
    return "\n".join(lines)
