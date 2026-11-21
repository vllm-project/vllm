# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Distribute compression work while keeping a full KV-cache replica per rank.

Ranks exchange boundary states before compression, then exchange packed output
rows after compression. Interior states stay on the rank that computes them.
"""

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import torch

from vllm.distributed import get_pcp_group
from vllm.models.deepseek_v4.common.ops.save_partial_states import (
    _SAVE_PARTIAL_STATES_KERNEL,
)
from vllm.models.deepseek_v4.compressor import CompressorMetadata, DeepseekCompressor
from vllm.models.deepseek_v41.common.ops.fused_compress_quant_cache import (
    _copy_pipeline_cache_rows,
)
from vllm.triton_utils import triton

if TYPE_CHECKING:
    from vllm.models.deepseek_v4.sparse_mla import DeepseekV4FlashMLAMetadata
    from vllm.v1.attention.backends.mla.indexer import DeepseekV32IndexerMetadata


@dataclass
class PCPTokenSelection:
    """Selected token rows, padded to the same capacity on every PCP rank.

    Attributes:
        local_indices: Input row indices per rank, shaped `[pcp_size, capacity]`.
        global_indices: Indices into the concatenation of all ranks' input rows.
        positions: Original request positions for the selected tokens.
        slot_mapping: Compressor-state cache slots; `-1` marks padding.
        token_to_req_indices: Row indices into the gathered request block table.

    Fields other than `local_indices` are flattened in rank order. Padded
    entries have safe input indices but must be ignored using `slot_mapping`.

    """

    local_indices: torch.Tensor
    global_indices: torch.Tensor
    positions: torch.Tensor
    slot_mapping: torch.Tensor
    token_to_req_indices: torch.Tensor


@dataclass
class PCPCompressorPlan:
    """Row selections for the two stages of sharded compression.

    Attributes:
        boundary: State rows to replicate across ranks before compression.
        output: Group-ending rows whose ranks compute the compressed outputs.

    """

    boundary: PCPTokenSelection
    output: PCPTokenSelection


def build_pcp_compressor_plan(
    metadata: CompressorMetadata,
    positions: torch.Tensor,
    pcp_size: int,
    max_reqs_per_rank: int,
    compress_ratio: int,
) -> PCPCompressorPlan:
    """Choose boundary states to exchange and assign compressed outputs to ranks.

    The rank holding a compression group's last token computes its output.
    Groups split across rank-local segments need boundary states from other
    ranks. Ratio 4 also retains the previous complete group because the next
    output reads it. Selected rows are padded uniformly for all-gather.

    Args:
        metadata: State-cache slots and request-segment indices for all ranks.
            Request indices address the concatenated, padded block tables.
        positions: Original request positions of all input rows, in rank order.
        pcp_size: Number of PCP ranks.
        max_reqs_per_rank: Reserved block-table rows per rank, including padding.
        compress_ratio: Number of new tokens per output, either 4 or 128.

    Returns:
        Index lists for boundary-state exchange and owned compressed outputs.
        No KV data is exchanged or compressed by this function.

    """
    req_indices = metadata.token_to_req_indices
    assert req_indices is not None
    slots = metadata.slot_mapping
    valid = slots >= 0
    starts = torch.full(
        (pcp_size * max_reqs_per_rank,),
        torch.iinfo(positions.dtype).max,
        dtype=positions.dtype,
        device=positions.device,
    )
    ends = torch.full_like(starts, -1)
    starts.scatter_reduce_(
        0,
        req_indices.long(),
        torch.where(valid, positions, torch.iinfo(positions.dtype).max),
        reduce="amin",
    )
    ends.scatter_reduce_(
        0, req_indices.long(), torch.where(valid, positions, -1), reduce="amax"
    )
    first = starts[req_indices.long()]
    last = ends[req_indices.long()]
    assert compress_ratio in (4, 128)
    overlap = int(compress_ratio == 4)
    window = positions // compress_ratio
    # C4's next output also reads the previous complete four-token block.
    # Preserve that block even when a segment ends on a compression boundary.
    tail_window = (last + 1) // compress_ratio - overlap
    boundary = valid & (
        ((first % compress_ratio != 0) & (window == first // compress_ratio))
        | (
            ((overlap != 0) | ((last + 1) % compress_ratio != 0))
            & (window >= tail_window)
        )
    )
    output = valid & ((positions + 1) % compress_ratio == 0)
    masks = (boundary.view(pcp_size, -1), output.view(pcp_size, -1))
    # One metadata-time sync, not one sync per layer. All ranks see the same
    # gathered metadata, including empty ranks, and choose identical capacities.
    capacities = torch.stack([mask.sum(1).max() for mask in masks]).tolist()
    num_tokens = slots.numel() // pcp_size
    indices = torch.arange(num_tokens, device=slots.device).expand(pcp_size, -1)
    rank_offsets = torch.arange(pcp_size, device=slots.device)[:, None] * num_tokens

    def select(mask: torch.Tensor, capacity: int) -> PCPTokenSelection:
        selected = (
            torch.where(mask, indices, num_tokens).sort(dim=1).values[:, :capacity]
        )
        is_valid = (selected < num_tokens).flatten()
        local_indices = selected.clamp_max(num_tokens - 1).contiguous()
        global_indices = (local_indices + rank_offsets).flatten()
        return PCPTokenSelection(
            local_indices=local_indices,
            global_indices=global_indices,
            positions=positions[global_indices],
            slot_mapping=torch.where(is_valid, slots[global_indices], -1),
            token_to_req_indices=req_indices[global_indices],
        )

    return PCPCompressorPlan(
        *(select(mask, size) for mask, size in zip(masks, capacities))
    )


def forward_sharded_compressor(
    compressor: DeepseekCompressor,
    kv_score: torch.Tensor,
    positions: torch.Tensor,
    rotary_emb,
    metadata: CompressorMetadata,
    k_cache_metadata: "DeepseekV4FlashMLAMetadata | DeepseekV32IndexerMetadata",
    plan: PCPCompressorPlan,
) -> None:
    """Execute the plan: save states, exchange boundaries, and compress locally.

    First save local KV/score rows to the compressor's intermediate state cache,
    then all-gather selected boundary rows and save those too. Each rank now has
    the states needed for its assigned outputs.

    The existing compressor writes normalized, RoPE-applied, quantized results
    directly to the final KV cache. Pack those local results into communication
    rows, all-gather them, and scatter them back to each rank's final KV cache.
    Only boundary states and finished outputs are exchanged, not all raw states.

    Args:
        compressor: Main-attention or indexer compressor providing the kernels.
        kv_score: Local projected KV and score, packed along the last dimension.
        positions: Original request positions for the local input rows.
        rotary_emb: Rotary embedding used for compressed outputs.
        metadata: Gathered intermediate state-cache slots and request tables.
        k_cache_metadata: Final compressed-cache slots for all input rows.
        plan: Boundary and output selections from `build_pcp_compressor_plan`.

    """
    group = get_pcp_group()
    rank = group.rank_in_group
    pcp_size = group.world_size
    local_slots = metadata.slot_mapping.view(pcp_size, -1)[rank]
    state_cache = compressor.state_cache.kv_cache
    state_width = state_cache.shape[-1] // 2
    local_positions = positions.new_zeros(local_slots.shape)
    local_positions[: positions.numel()].copy_(positions)

    def save_states(
        values: torch.Tensor, token_positions: torch.Tensor, slots: torch.Tensor
    ) -> None:
        if slots.numel() == 0:
            return
        kv, score = values.split(state_width, dim=-1)
        _SAVE_PARTIAL_STATES_KERNEL(
            kv=kv,
            score=score,
            ape=compressor.ape,
            positions=token_positions,
            state_cache=state_cache,
            slot_mapping=slots,
            block_size=metadata.block_size,
            state_width=state_width,
            compress_ratio=compressor.compress_ratio,
        )

    save_states(kv_score, local_positions, local_slots)
    boundary = plan.boundary
    if boundary.local_indices.shape[1] > 0:
        values = kv_score[boundary.local_indices[rank]].contiguous()
        values = group.all_gather(values, dim=0)
        # Replicating unfinished windows also preserves their history when
        # chunked prefill or decode assigns the next token to another rank.
        save_states(values, boundary.positions, boundary.slot_mapping)

    output = plan.output
    num_outputs = output.local_indices.shape[1]
    if num_outputs == 0:
        return
    output_slots = k_cache_metadata.slot_mapping[output.global_indices]
    output_slots = torch.where(output.slot_mapping >= 0, output_slots, -1)
    local_slice = slice(rank * num_outputs, (rank + 1) * num_outputs)
    local_metadata = CompressorMetadata(
        block_table=metadata.block_table,
        slot_mapping=output.slot_mapping[local_slice],
        block_size=metadata.block_size,
        token_to_req_indices=output.token_to_req_indices[local_slice],
        num_decode_tokens=0,
    )
    local_k_metadata = replace(k_cache_metadata, slot_mapping=output_slots[local_slice])
    compressor._compress_norm_rope_store(
        local_metadata,
        state_cache,
        state_width,
        output.positions[local_slice],
        rotary_emb,
        {compressor.k_cache_prefix: local_k_metadata},
        {},
    )
    kv_cache = compressor._static_forward_context[compressor.k_cache_prefix].kv_cache
    rows = torch.empty(
        (num_outputs, kv_cache.shape[-1]), dtype=torch.uint8, device=kv_cache.device
    )
    copy_pcp_cache_rows(
        kv_cache,
        rows,
        output.positions[local_slice],
        local_k_metadata.slot_mapping,
        compressor.compress_ratio,
        scatter=False,
    )
    rows = group.all_gather(rows, dim=0)
    copy_pcp_cache_rows(
        kv_cache,
        rows,
        output.positions,
        output_slots,
        compressor.compress_ratio,
        scatter=True,
    )


def copy_pcp_cache_rows(
    cache: torch.Tensor,
    rows: torch.Tensor,
    positions: torch.Tensor,
    slots: torch.Tensor,
    compress_ratio: int,
    *,
    scatter: bool,
) -> None:
    """Copy quantized bytes between a paged cache and packed communication rows.

    A cache page stores all token values first, then all token scales. A packed
    row instead keeps one token's values and scales together. This copies bytes
    between those layouts without quantizing or dequantizing them.

    Args:
        cache: Uint8 cache view shaped `[pages, block_size, row_bytes]`. Its first
            stride preserves page padding; the kernel computes split-layout
            addresses rather than indexing a token as `cache[page, token]`.
        rows: Contiguous uint8 communication rows shaped `[tokens, row_bytes]`.
        positions: Token positions used to select compression-group endings.
            With ratio 1, all positions pass and only the slot mask matters.
        slots: Cache slot per row; negative slots are ignored.
        compress_ratio: Group size, or 1 to copy uncompressed SWA KV rows.
        scatter: If true, copy rows into cache. Otherwise, read cache into rows
            and fill invalid or non-group-ending rows with zeros.

    """
    assert cache.dtype == rows.dtype == torch.uint8
    row_bytes = cache.shape[-1]
    assert row_bytes == rows.shape[-1]
    assert row_bytes in (584, 132, 68)
    assert cache.stride(-1) == 1 and rows.is_contiguous()
    # Bypass the V4.1 wrapper's ratio restriction and CUDA launch options.
    _copy_pipeline_cache_rows[(rows.shape[0],)](
        cache,
        rows,
        positions,
        slots,
        slots.numel(),
        CACHE_STRIDE=cache.stride(0),
        CACHE_BLOCK=cache.shape[1],
        ROW_BYTES=row_bytes,
        VALUE_BYTES={584: 576, 132: 128, 68: 64}[row_bytes],
        COMPRESS_RATIO=compress_ratio,
        SCATTER=scatter,
        BLOCK=triton.next_power_of_2(row_bytes),
        num_warps=4,
    )
