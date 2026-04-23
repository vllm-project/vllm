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
    ``group_kinds`` maps each kv_cache_group to its type.
    """

    # Regions in descriptor handle order
    fa_regions: tuple[RegionPlan, ...]
    ssm_regions: tuple[RegionPlan, ...]

    # Per-group geometric properties (worker-facing, model-agnostic)
    physical_per_logical: tuple[int, ...]

    # Per-group type (FA, SWA, MAMBA, GDN).
    group_kinds: tuple[GroupKind, ...]

    # Source rank routing
    all_source_ranks: tuple[int, ...]
    fa_source_ranks: tuple[int, ...]
    fa_source_set: frozenset[int]

    # Split handle parameters
    num_fa_reads: int
    num_mamba_reads: int
    fa_head_slots: dict[int, int]

    # Remote engine facts (needed by worker at read time)
    remote_tp_size: int
    remote_block_size: int
    remote_block_len: int
    remote_physical_blocks_per_logical: int

    # Stride for expanding remote logical block IDs to kernel block IDs.
    # Dense: equals local physical_blocks_per_logical (stride == count).
    # Mamba: equals remote_physical_blocks_per_logical (stride != count).
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


def _physical_head_range(tp_size: int, num_heads: int, rank: int) -> range:
    if tp_size <= num_heads:
        assert num_heads % tp_size == 0
        per_rank = num_heads // tp_size
        return range(rank * per_rank, (rank + 1) * per_rank)
    else:
        h = rank * num_heads // tp_size
        return range(h, h + 1)


def _range_overlap(a: range, b: range) -> range:
    start = max(a.start, b.start)
    stop = min(a.stop, b.stop)
    return range(start, max(start, stop))


def _compute_tp_ratio(tp_size: int, remote_tp_size: int) -> int:
    if tp_size >= remote_tp_size:
        assert tp_size % remote_tp_size == 0
        return tp_size // remote_tp_size
    assert remote_tp_size % tp_size == 0
    return -(remote_tp_size // tp_size)


def _compute_fa_source_ranks(
    tp_rank: int,
    tp_size: int,
    remote_tp_size: int,
    is_mla: bool,
    total_num_kv_heads: int,
) -> tuple[list[int], list[int], int, int]:
    """Compute FA and all source ranks for Mamba models.

    Returns (fa_source_ranks, all_source_ranks, num_fa_reads, num_mamba_reads).
    Mirrors the logic in MambaModelBlockTransferPolicy.build_engine_transfer_info.
    """
    K = total_num_kv_heads
    tp_ratio = _compute_tp_ratio(tp_size, remote_tp_size)
    abs_tp = -tp_ratio if tp_ratio < 0 else 1
    mamba_range: range | None = None
    if tp_ratio < 0:
        mamba_range = range(tp_rank * abs_tp, (tp_rank + 1) * abs_tp)

    fa_source_ranks: list[int]
    if is_mla or tp_ratio >= 0:
        num_fa_reads = 1
        if is_mla:
            fa_source_ranks = [0]
        elif tp_ratio > 0:
            fa_source_ranks = [tp_rank // tp_ratio]
        else:
            fa_source_ranks = [tp_rank]
    else:
        local_needs = _physical_head_range(tp_size, K, tp_rank)
        search_range = mamba_range if mamba_range is not None else range(remote_tp_size)
        seen: set[tuple[int, int]] = set()
        fa_source_ranks = []
        for p in search_range:
            p_has = _physical_head_range(remote_tp_size, K, p)
            ov = _range_overlap(local_needs, p_has)
            if len(ov) > 0:
                key = (ov.start, ov.stop)
                if key not in seen:
                    seen.add(key)
                    fa_source_ranks.append(p)
        if not fa_source_ranks:
            for p in range(remote_tp_size):
                p_has = _physical_head_range(remote_tp_size, K, p)
                ov = _range_overlap(local_needs, p_has)
                if len(ov) > 0:
                    key = (ov.start, ov.stop)
                    if key not in seen:
                        seen.add(key)
                        fa_source_ranks.append(p)
        num_fa_reads = len(fa_source_ranks)

    if mamba_range is not None and abs_tp > num_fa_reads:
        num_mamba_reads = abs_tp
        all_source_ranks = list(mamba_range)
    else:
        num_mamba_reads = num_fa_reads
        all_source_ranks = list(fa_source_ranks)

    return fa_source_ranks, all_source_ranks, num_fa_reads, num_mamba_reads


def _compute_fa_head_slots(
    fa_source_ranks: list[int],
    all_source_ranks: list[int],
    remote_tp_size: int,
    total_num_kv_heads: int,
) -> dict[int, int]:
    """Pre-compute the FA head slot for each source rank.

    Mirrors _fa_head_slot from block_transfer_policy.py but pre-computes
    all values at plan generation time.
    """
    fa_index = {r: i for i, r in enumerate(fa_source_ranks)}
    K = total_num_kv_heads
    result: dict[int, int] = {}
    for rank in all_source_ranks:
        if rank in fa_index:
            result[rank] = fa_index[rank]
        else:
            r_head = _physical_head_range(remote_tp_size, K, rank)
            for target in fa_source_ranks:
                t_head = _physical_head_range(remote_tp_size, K, target)
                if _range_overlap(r_head, t_head):
                    result[rank] = fa_index[target]
                    break
            else:
                result[rank] = 0
    return result


def _compute_fa_rank_offset(
    tp_rank: int,
    tp_size: int,
    tp_ratio: int,
    is_mla: bool,
    total_num_kv_heads: int,
    remote_tp_size: int,
    fa_source_ranks: list[int],
    remote_kv_block_len: int,
) -> int:
    """Byte offset into remote FA block for this local rank.

    Mirrors _fa_rank_offset from block_transfer_policy.py, but takes
    raw parameters instead of MambaEngineTransferInfo.
    """
    if is_mla or tp_ratio <= 0:
        return 0
    K = total_num_kv_heads
    is_local_replicated = tp_size > K
    if is_local_replicated:
        local_head = tp_rank * K // tp_size
        p_rank = fa_source_ranks[0]
        p_start = p_rank * K // remote_tp_size
        return (local_head - p_start) * remote_kv_block_len
    return tp_rank % tp_ratio * remote_kv_block_len


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
    """Generate transfer plan for dense (FA-only) models.

    Mirrors the combined logic of:
      - DenseModelBlockTransferPolicy.build_engine_transfer_info()
      - DenseModelBlockTransferPolicy.build_remote_descs()
    """
    tp_ratio = _compute_tp_ratio(tp_size, remote_tp_size)
    block_size_ratio = block_size // remote_block_size
    indexes_into_remote = (
        not (is_mla or remote_tp_size > total_num_kv_heads) and tp_ratio > 0
    )

    # Source ranks — mirrors TransferTopology.target_remote_ranks for dense
    if tp_ratio > 0:
        all_source_ranks: tuple[int, ...] = (tp_rank // tp_ratio,)
    else:
        abs_ratio = -tp_ratio
        all_source_ranks = tuple(tp_rank * abs_ratio + i for i in range(abs_ratio))

    # Build FA regions — one (K, optionally V) per layer
    fa_regions: list[RegionPlan] = []
    for i in range(len(remote_block_lens)):
        local_block_len = _get_kv_block_len(i, block_len_per_layer, is_blocks_first)
        remote_kv_block_len = local_block_len // block_size_ratio

        k_desc_bytes = local_block_len
        if block_size_ratio > 1:
            k_desc_bytes = remote_kv_block_len
        if tp_ratio < 0 and not is_mla:
            k_desc_bytes = k_desc_bytes // (-tp_ratio)

        rank_offset = (
            tp_rank % tp_ratio * remote_kv_block_len if indexes_into_remote else 0
        )
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
            v_desc_bytes = _get_kv_block_len(i, block_len_per_layer, is_blocks_first)
            if tp_ratio < 0 and not is_mla:
                v_desc_bytes = v_desc_bytes // (-tp_ratio)

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

    # For dense split handles: fa_head_slots maps rank → index,
    # so the executor uniformly splits all descs by abs_tp.
    fa_head_slots = {r: i for i, r in enumerate(all_source_ranks)}

    return EngineTransferPlan(
        fa_regions=tuple(fa_regions),
        ssm_regions=(),
        physical_per_logical=(remote_physical_blocks_per_logical,),
        group_kinds=(GroupKind.FA,),
        all_source_ranks=all_source_ranks,
        fa_source_ranks=all_source_ranks,
        fa_source_set=frozenset(all_source_ranks),
        num_fa_reads=len(all_source_ranks),
        num_mamba_reads=0,
        fa_head_slots=fa_head_slots,
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
    """Generate transfer plan for hybrid Mamba (SSM + FA) models.

    Mirrors the combined logic of:
      - MambaModelBlockTransferPolicy.build_engine_transfer_info()
      - MambaModelBlockTransferPolicy._build_fa_remote_descs()
      - MambaModelBlockTransferPolicy._build_mamba_remote_descs()
    """
    tp_ratio = _compute_tp_ratio(tp_size, remote_tp_size)
    block_size_ratio = block_size // remote_block_size
    assert block_size_ratio == 1, (
        "Mamba 3-read transfer with block_size_ratio != 1 "
        f"is not tested. Got {block_size_ratio=}."
    )

    # ---- Source rank computation ----
    (
        fa_source_ranks,
        all_source_ranks,
        num_fa_reads,
        num_mamba_reads,
    ) = _compute_fa_source_ranks(
        tp_rank,
        tp_size,
        remote_tp_size,
        is_mla,
        total_num_kv_heads,
    )

    # ---- FA head slots (for split handles) ----
    fa_head_slots = _compute_fa_head_slots(
        fa_source_ranks,
        all_source_ranks,
        remote_tp_size,
        total_num_kv_heads,
    )

    # ---- FA regions ----
    fa_regions: list[RegionPlan] = []
    for i in range(len(remote_block_lens)):
        local_block_len = _get_kv_block_len(
            i,
            block_len_per_layer,
            is_blocks_first,
        )
        remote_kv_block_len = local_block_len // block_size_ratio

        k_desc_bytes = local_block_len
        if block_size_ratio > 1:
            k_desc_bytes = remote_kv_block_len
        if tp_ratio < 0 and not is_mla:
            k_desc_bytes = k_desc_bytes // num_fa_reads

        rank_offset = _compute_fa_rank_offset(
            tp_rank,
            tp_size,
            tp_ratio,
            is_mla,
            total_num_kv_heads,
            remote_tp_size,
            fa_source_ranks,
            remote_kv_block_len,
        )

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
            v_desc_bytes = _get_kv_block_len(
                i,
                block_len_per_layer,
                is_blocks_first,
            )
            if tp_ratio < 0 and not is_mla:
                v_desc_bytes = v_desc_bytes // num_fa_reads

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

    # ---- SSM regions ----
    effective_ratio = max(tp_ratio, 1)
    local_offset = tp_rank % effective_ratio
    conv_size_remote = remote_ssm_sizes[0]
    remote_ratio = remote_physical_blocks_per_logical
    ssm_num_blocks = remote_num_blocks // remote_ratio

    if tp_ratio >= 1:
        conv_offsets = conv_decomp.remote_conv_offsets(
            local_offset,
            effective_ratio,
        )
        ssm_read_size = ssm_sizes[1]
    else:
        abs_ratio = -tp_ratio
        xb_p = conv_decomp.x_bytes // abs_ratio
        bb_p = conv_decomp.b_bytes // abs_ratio
        conv_offsets = [
            (0, xb_p),
            (xb_p, bb_p),
            (xb_p + bb_p, bb_p),
        ]
        ssm_read_size = remote_ssm_sizes[1]

    conv_kinds = [RegionKind.SSM_CONV_X, RegionKind.SSM_CONV_B, RegionKind.SSM_CONV_C]
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
        all_source_ranks=tuple(all_source_ranks),
        fa_source_ranks=tuple(fa_source_ranks),
        fa_source_set=frozenset(fa_source_ranks),
        num_fa_reads=num_fa_reads,
        num_mamba_reads=num_mamba_reads,
        fa_head_slots=fa_head_slots,
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

    Replaces DenseModelBlockTransferPolicy.build_remote_descs() and
    MambaModelBlockTransferPolicy.build_remote_descs().
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

    Replaces DenseModelBlockTransferPolicy.get_block_descs_ids() and
    MambaModelBlockTransferPolicy.get_block_descs_ids().
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
        if plan.group_kinds[i].is_ssm:
            ssm_region_ids = np.arange(num_ssm_regions)[:, None]
            all_descs.append(
                (
                    ssm_region_ids * logical_blocks + group_arr[None, :] + num_fa_descs
                ).flatten()
            )
        else:
            fa_region_ids = np.arange(num_fa_regions)[:, None]
            all_descs.append(
                (fa_region_ids * num_blocks + group_arr[None, :]).flatten()
            )

    return np.concatenate(all_descs)


def compute_read_specs_from_plan(
    plan: EngineTransferPlan,
    local_block_ids: BlockIds,
    remote_block_ids: BlockIds,
) -> list[ReadSpec]:
    """Compute read specs from plan.

    Replaces compute_read_specs() + filter_block_ids_for_rank().
    No _should_skip_fa — the plan structurally encodes which ranks
    serve which groups via fa_source_set.
    """
    specs: list[ReadSpec] = []
    for rank in plan.all_source_ranks:
        skip_fa = rank not in plan.fa_source_set
        if not skip_fa:
            specs.append(
                ReadSpec(
                    remote_rank=rank,
                    local_block_ids=local_block_ids,
                    remote_block_ids=remote_block_ids,
                )
            )
        else:
            num_groups = len(local_block_ids)
            filtered_local: list[list[int]] = [
                list(local_block_ids[g]) if plan.group_kinds[g].is_ssm else []
                for g in range(num_groups)
            ]
            filtered_remote: list[list[int]] = [
                list(remote_block_ids[g]) if plan.group_kinds[g].is_ssm else []
                for g in range(num_groups)
            ]
            specs.append(
                ReadSpec(
                    remote_rank=rank,
                    local_block_ids=filtered_local,
                    remote_block_ids=filtered_remote,
                )
            )
    return specs


def build_local_splits_from_plan(
    plan: EngineTransferPlan,
    src_blocks_data: list[tuple[int, int, int]],
    num_fa_descs: int,
) -> list[list[tuple[int, int, int]]]:
    """Build split handle data for P_TP > D_TP scenario.

    Replaces DenseModelBlockTransferPolicy.build_src_split_handles() and
    MambaModelBlockTransferPolicy.build_src_split_handles() +
    compute_split_handle_data().

    When num_ssm_regions == 0 (dense), all descs are FA and the split
    is uniform.  When SSM regions exist, FA and SSM descs get different
    split factors.
    """
    abs_tp = len(plan.all_source_ranks)
    result: list[list[tuple[int, int, int]]] = []

    for p_idx, p_rank in enumerate(plan.all_source_ranks):
        skip_fa = p_rank not in plan.fa_source_set
        fa_slot = plan.fa_head_slots.get(p_rank, 0) if not skip_fa else 0

        handle: list[tuple[int, int, int]] = []
        for j, (addr, local_len, dev) in enumerate(src_blocks_data):
            if j < num_fa_descs:
                assert plan.num_fa_reads >= 1
                fa_chunk = local_len // plan.num_fa_reads
                handle.append((addr + fa_slot * fa_chunk, fa_chunk, dev))
            else:
                mamba_chunk = local_len // abs_tp
                handle.append((addr + p_idx * mamba_chunk, mamba_chunk, dev))
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
        f"  Source ranks: all={list(plan.all_source_ranks)}, "
        f"fa={list(plan.fa_source_ranks)}",
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
