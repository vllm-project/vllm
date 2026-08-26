# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Paged side-cache ownership and metadata for Qwen4Exp QSA.

Each QSA layer keeps a fixed circular buffer of raw index keys (the
compressor state) and one compressed key. MRoPE models pack exact three-axis
positions beside the raw keys; text models derive group positions from
logical positions. The compressor state uses one block per request, while
the compressed owner uses ``MLAAttentionSpec.compress_ratio`` so its block
table follows the main KV-cache lifecycle. Their physical tensor storage is
shared by the generic cache-layout planner.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import cache
from typing import ClassVar

import torch
from torch import nn

from vllm.config import CacheConfig, VllmConfig
from vllm.config.cache import CacheDType
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.platforms import current_platform
from vllm.triton_utils import HAS_TRITON, tl, triton
from vllm.utils.math_utils import cdiv
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionMetadata,
    AttentionMetadataBuilder,
    CommonAttentionMetadata,
)
from vllm.v1.attention.backends.utils import PAD_SLOT_ID
from vllm.v1.kv_cache_interface import (
    AttentionSpec,
    CircularBufferSpec,
    KVCacheSpec,
    MLAAttentionSpec,
)


def canonical_qsa_rope_positions(positions: torch.Tensor) -> torch.Tensor:
    """Return exact per-token positions as ``[tokens, 1, 3]`` int64 rows."""

    if positions.ndim == 1:
        positions = positions.unsqueeze(0).expand(3, -1)
    elif positions.ndim != 2 or positions.shape[0] not in (1, 3):
        raise ValueError("QSA RoPE positions must be [tokens] or [1|3, tokens]")
    if positions.shape[0] == 1:
        positions = positions.expand(3, -1)
    return positions.transpose(0, 1).unsqueeze(1).to(torch.int64)


def _logical_positions(
    query_start_loc: torch.Tensor,
    seq_lens: torch.Tensor,
    token_to_req: torch.Tensor,
    num_tokens: int,
) -> torch.Tensor:
    if num_tokens == 0:
        return seq_lens.new_empty((0,), dtype=torch.int64)
    arange = torch.arange(num_tokens, device=query_start_loc.device)
    requests = token_to_req[:num_tokens].long()
    query_lens = torch.diff(query_start_loc)
    within_query = arange - query_start_loc.index_select(0, requests)
    return (
        seq_lens.index_select(0, requests).long()
        - query_lens.index_select(0, requests).long()
        + within_query.long()
    )


def _logical_to_physical_qsa_slots(
    block_table: torch.Tensor,
    request_indices: torch.Tensor,
    logical_positions: torch.Tensor,
    block_size: int,
) -> torch.Tensor:
    if block_size <= 0:
        raise ValueError("QSA cache block size must be positive")
    if block_table.ndim != 2:
        raise ValueError("QSA block table must be two-dimensional")
    if request_indices.shape != logical_positions.shape:
        request_indices = torch.broadcast_to(request_indices, logical_positions.shape)

    requests = request_indices.to(device=block_table.device, dtype=torch.long)
    positions = logical_positions.to(device=block_table.device, dtype=torch.long)
    valid = (requests >= 0) & (requests < block_table.shape[0]) & (positions >= 0)
    logical_blocks = torch.div(
        positions.clamp_min(0), block_size, rounding_mode="floor"
    )
    valid &= logical_blocks < block_table.shape[1]
    safe_requests = requests.clamp(0, max(block_table.shape[0] - 1, 0))
    safe_blocks = logical_blocks.clamp(0, max(block_table.shape[1] - 1, 0))
    if not all(block_table.shape):
        return torch.full_like(positions, PAD_SLOT_ID)
    physical_blocks = block_table[safe_requests, safe_blocks].long()
    valid &= physical_blocks >= 0
    slots = physical_blocks * block_size + positions.remainder(block_size)
    return torch.where(valid, slots, PAD_SLOT_ID)


def circular_qsa_slot_mapping(
    block_table: torch.Tensor,
    token_to_req: torch.Tensor,
    logical_positions: torch.Tensor,
    compressor_state_size: int,
    query_start_loc: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Map each request to its fixed physical block as a circular token ring."""

    if compressor_state_size <= 0:
        raise ValueError("QSA circular buffer size must be positive")
    if block_table.ndim != 2:
        raise ValueError("QSA block table must be two-dimensional")

    requests = token_to_req.to(device=block_table.device, dtype=torch.long)
    positions = logical_positions.to(device=block_table.device, dtype=torch.long)
    if not all(block_table.shape):
        slots = torch.full_like(positions, PAD_SLOT_ID)
    else:
        valid = (requests >= 0) & (requests < block_table.shape[0]) & (positions >= 0)
        safe_requests = requests.clamp(0, block_table.shape[0] - 1)
        physical_blocks = block_table[safe_requests, 0].long()
        valid &= physical_blocks >= 0
        slots = physical_blocks * compressor_state_size + positions.remainder(
            compressor_state_size
        )
        slots = torch.where(valid, slots, PAD_SLOT_ID)

    if query_start_loc is not None:
        if query_start_loc.ndim != 1 or query_start_loc.shape[0] < 2:
            raise ValueError("QSA query starts must contain a terminal offset")
        query_start_loc = query_start_loc.to(block_table.device)
        num_requests = query_start_loc.shape[0] - 1
        safe_requests = requests.clamp(0, num_requests - 1)
        request_ends = query_start_loc.index_select(0, safe_requests + 1)
        rows = torch.arange(slots.numel(), device=slots.device)
        keep = (
            (requests >= 0)
            & (requests < num_requests)
            & (rows + compressor_state_size >= request_ends)
        )
        slots = torch.where(keep, slots, PAD_SLOT_ID)

    slots = slots.to(torch.int64)
    if out is not None:
        out.fill_(PAD_SLOT_ID)
        out[: slots.numel()].copy_(slots)
        return out[: slots.numel()]
    return slots


def compressed_qsa_slot_mapping(
    block_table: torch.Tensor,
    token_to_req: torch.Tensor,
    logical_positions: torch.Tensor,
    storage_block_size: int,
    compress_ratio: int,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Build boundary-only slots for an ``MLAAttentionSpec`` QSA cache."""

    if storage_block_size <= 0 or compress_ratio <= 0:
        raise ValueError("QSA block size and compression ratio must be positive")
    compressed_positions = torch.div(
        logical_positions.clamp_min(0), compress_ratio, rounding_mode="floor"
    )
    slots = _logical_to_physical_qsa_slots(
        block_table,
        token_to_req,
        compressed_positions,
        storage_block_size,
    )
    valid = (logical_positions >= 0) & (
        (logical_positions + 1).remainder(compress_ratio) == 0
    )
    slots = torch.where(valid, slots, PAD_SLOT_ID).to(torch.int64)
    if out is not None:
        out.fill_(PAD_SLOT_ID)
        out[: slots.numel()].copy_(slots)
        return out[: slots.numel()]
    return slots


@cache
def _metadata_launch_pdl() -> bool:
    return current_platform.is_arch_support_pdl()


@triton.jit(
    do_not_specialize=[
        "num_reqs",
        "num_mapped_tokens",
        "num_tokens",
        "max_num_work",
        "num_search_steps",
        "work_search_steps",
    ]
)
def _build_qsa_metadata_kernel(
    query_start_loc_ptr,
    seq_lens_ptr,
    common_slot_mapping_ptr,
    block_table_ptr,
    token_to_req_ptr,
    logical_positions_ptr,
    slot_mapping_ptr,
    k_work_metadata_ptr,
    block_table_stride_0: tl.constexpr,
    block_table_stride_1: tl.constexpr,
    num_reqs,
    num_mapped_tokens,
    num_tokens,
    max_num_work,
    num_search_steps,
    work_search_steps,
    storage_block_size: tl.constexpr,
    compress_ratio: tl.constexpr,
    circular_buffer_size: tl.constexpr,
    num_block_table_columns: tl.constexpr,
    launch_pdl: tl.constexpr,
    TOKEN_BLOCK_SIZE: tl.constexpr,
    REQUEST_SCAN_SIZE: tl.constexpr,
    WORK_BLOCK_SIZE: tl.constexpr,
):
    if launch_pdl:
        tl.extra.cuda.gdc_wait()

    pid = tl.program_id(0)
    token_idx = pid * TOKEN_BLOCK_SIZE + tl.arange(0, TOKEN_BLOCK_SIZE)
    store_mask = token_idx < num_tokens
    mapped = token_idx < num_mapped_tokens
    search_token_idx = tl.minimum(token_idx, num_mapped_tokens - 1)
    request_idx = tl.zeros((TOKEN_BLOCK_SIZE,), tl.int32)
    # Find the last query start at or before each token. The dynamic loop avoids
    # compiling a kernel variant for every ceil(log2(num_reqs)).
    for step in tl.range(0, num_search_steps):
        candidate = request_idx + (1 << (num_search_steps - step - 1))
        valid_candidate = candidate < num_reqs
        candidate_start = tl.load(
            query_start_loc_ptr + candidate,
            mask=valid_candidate,
            other=num_mapped_tokens + 1,
        )
        advance = valid_candidate & (candidate_start <= search_token_idx)
        request_idx = tl.where(advance, candidate, request_idx)
    query_start = tl.load(query_start_loc_ptr + request_idx, mask=mapped, other=0)
    query_end = tl.load(query_start_loc_ptr + request_idx + 1, mask=mapped, other=0)
    seq_len = tl.load(seq_lens_ptr + request_idx, mask=mapped, other=0)
    logical_position = seq_len - (query_end - query_start) + token_idx - query_start
    logical_position = tl.where(mapped, logical_position, -1)
    tl.store(
        token_to_req_ptr + token_idx,
        tl.where(mapped, request_idx, 0),
        mask=store_mask,
    )
    tl.store(
        logical_positions_ptr + token_idx,
        logical_position,
        mask=store_mask,
    )

    # circular_buffer_size is constexpr, so each builder instance compiles out
    # the other QSA cache owner's slot-mapping rule.
    if circular_buffer_size > 0:
        valid = (
            mapped
            & (logical_position >= 0)
            & (token_idx + circular_buffer_size >= query_end)
            & (num_block_table_columns > 0)
        )
        physical_block = tl.load(
            block_table_ptr + request_idx * block_table_stride_0,
            mask=valid,
            other=-1,
        )
        valid &= physical_block >= 0
        slot = physical_block * circular_buffer_size + (
            logical_position % circular_buffer_size
        )
    elif compress_ratio != 1:
        compressed_position = tl.maximum(logical_position, 0) // compress_ratio
        logical_block = compressed_position // storage_block_size
        valid = (
            mapped
            & (logical_position >= 0)
            & ((logical_position + 1) % compress_ratio == 0)
            & (logical_block < num_block_table_columns)
        )
        physical_block = tl.load(
            block_table_ptr
            + request_idx * block_table_stride_0
            + logical_block * block_table_stride_1,
            mask=valid,
            other=-1,
        )
        valid &= physical_block >= 0
        valid &= (
            tl.load(common_slot_mapping_ptr + token_idx, mask=mapped, other=-1) >= 0
        )
        slot = physical_block * storage_block_size + (
            compressed_position % storage_block_size
        )
    if (circular_buffer_size > 0) or (compress_ratio != 1):
        tl.store(
            slot_mapping_ptr + token_idx,
            tl.where(valid, slot, -1),
            mask=store_mask,
        )
    work_tile_start = pid * WORK_BLOCK_SIZE
    has_work_tile = work_tile_start < max_num_work
    if k_work_metadata_ptr is not None and has_work_tile:
        # Every work CTA builds the request prefix in registers. Recomputing this
        # small vector lets CTAs write disjoint work tiles without a grid barrier.
        requests = tl.arange(0, REQUEST_SCAN_SIZE)
        valid_request = requests < num_reqs
        request_query_start = tl.load(
            query_start_loc_ptr + requests, mask=valid_request, other=0
        )
        request_query_end = tl.load(
            query_start_loc_ptr + requests + 1, mask=valid_request, other=0
        )
        request_seq_len = tl.load(seq_lens_ptr + requests, mask=valid_request, other=0)
        request_query_len = request_query_end - request_query_start
        chunk_start = request_seq_len - request_query_len
        num_groups = request_seq_len // compress_ratio - chunk_start // compress_ratio
        # Nonempty requests need one item even without a completed compression
        # group because work item zero also commits the current raw-K suffix.
        work_counts = tl.where(request_query_len > 0, tl.maximum(num_groups, 1), 0)
        work_ends = tl.cumsum(tl.where(valid_request, work_counts, 0), axis=0)
        total_work = tl.sum(tl.where(valid_request, work_counts, 0), axis=0)
        work_offsets = tl.arange(0, WORK_BLOCK_SIZE)
        work = work_tile_start + work_offsets
        in_bounds = work < max_num_work
        active = in_bounds & (work < total_work)
        request = tl.zeros((WORK_BLOCK_SIZE,), dtype=tl.int32)
        # Request zero starts at zero for every active work item. Descending
        # steps find the last request starting at or before this item.
        for step_idx in tl.range(0, work_search_steps):
            step = 1 << (work_search_steps - step_idx - 1)
            candidate = request + step
            valid_candidate = candidate < num_reqs
            candidate_start = tl.gather(work_ends, candidate - 1, 0)
            advance = active & valid_candidate & (candidate_start <= work)
            request = tl.where(advance, candidate, request)

        if launch_pdl:
            # Let the dependent grid start launch setup while these CTAs finish
            # stores; its gdc_wait still orders access to the completed metadata.
            tl.extra.cuda.gdc_launch_dependents()

        owner_work_start = tl.gather(work_ends, tl.maximum(request - 1, 0), 0)
        owner_work_start = tl.where(request == 0, 0, owner_work_start)
        work_in_request = work - owner_work_start
        tl.store(
            k_work_metadata_ptr + work * 2,
            tl.where(active, request, -1),
            mask=in_bounds,
        )
        tl.store(
            k_work_metadata_ptr + work * 2 + 1,
            tl.where(active, work_in_request, -1),
            mask=in_bounds,
        )

    else:
        if launch_pdl:
            # A dependent grid launches only after every CTA has signaled.
            tl.extra.cuda.gdc_launch_dependents()


def build_qsa_metadata_triton(
    common_attn_metadata: CommonAttentionMetadata,
    token_to_req_buffer: torch.Tensor,
    logical_positions_buffer: torch.Tensor,
    slot_mapping_buffer: torch.Tensor,
    *,
    storage_block_size: int,
    compress_ratio: int,
    circular_buffer_size: int = 0,
    k_work_metadata_buffer: torch.Tensor | None = None,
    request_capacity: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build QSA side-cache and optional pre-indexer work metadata."""
    num_tokens = common_attn_metadata.num_actual_tokens
    num_mapped_tokens = int(common_attn_metadata.query_start_loc_cpu[-1])
    token_to_req = token_to_req_buffer[:num_tokens]
    logical_positions = logical_positions_buffer[:num_tokens]
    slot_mapping = slot_mapping_buffer[:num_tokens]
    num_reqs = common_attn_metadata.query_start_loc.shape[0] - 1
    assert num_reqs > 0

    if k_work_metadata_buffer is not None:
        if request_capacity is None:
            request_capacity = num_reqs
        assert request_capacity >= num_reqs
        # Pad for tl.arange while keeping the scan width stable across live batches.
        request_scan_size = 1 << int(math.ceil(math.log2(request_capacity)))
        max_num_work = k_work_metadata_buffer.shape[0]
    else:
        request_scan_size = 1
        max_num_work = 0

    if num_tokens == 0 and k_work_metadata_buffer is None:
        return token_to_req, logical_positions, slot_mapping

    block_table = common_attn_metadata.block_table_tensor
    num_search_steps = int(math.ceil(math.log2(num_reqs)))
    work_search_steps = int(math.ceil(math.log2(num_reqs)))
    # The same grid covers token tiles and, for the compressed cache, work tiles.
    num_token_blocks = cdiv(num_tokens, 128)
    num_work_blocks = (
        cdiv(max_num_work, 256) if k_work_metadata_buffer is not None else 0
    )
    _build_qsa_metadata_kernel[(max(num_token_blocks, num_work_blocks, 1),)](
        common_attn_metadata.query_start_loc,
        common_attn_metadata.seq_lens,
        common_attn_metadata.slot_mapping,
        block_table,
        token_to_req,
        logical_positions,
        slot_mapping,
        k_work_metadata_buffer,
        block_table.stride(0),
        block_table.stride(1),
        num_reqs,
        num_mapped_tokens,
        num_tokens,
        max_num_work,
        num_search_steps,
        work_search_steps,
        storage_block_size,
        compress_ratio,
        circular_buffer_size,
        block_table.shape[1],
        launch_pdl=_metadata_launch_pdl(),
        TOKEN_BLOCK_SIZE=128,
        REQUEST_SCAN_SIZE=request_scan_size,
        WORK_BLOCK_SIZE=256,
        num_warps=4,
    )
    if circular_buffer_size == 0 and compress_ratio == 1:
        slot_mapping = common_attn_metadata.slot_mapping[:num_tokens]
    return token_to_req, logical_positions, slot_mapping


def _build_qsa_metadata_torch(
    common_attn_metadata: CommonAttentionMetadata,
    token_to_req_buffer: torch.Tensor,
    logical_positions_buffer: torch.Tensor,
    slot_mapping_buffer: torch.Tensor,
    *,
    storage_block_size: int,
    compress_ratio: int,
    circular_buffer_size: int = 0,
    k_work_metadata_buffer: torch.Tensor | None = None,
    request_capacity: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    del request_capacity
    num_tokens = common_attn_metadata.num_actual_tokens
    num_mapped_tokens = int(common_attn_metadata.query_start_loc_cpu[-1])
    logical_positions = logical_positions_buffer[:num_tokens]

    token_to_req = common_attn_metadata.token_to_req_indices(token_to_req_buffer)[
        :num_tokens
    ]
    logical_positions[:num_mapped_tokens].copy_(
        _logical_positions(
            common_attn_metadata.query_start_loc,
            common_attn_metadata.seq_lens,
            token_to_req[:num_mapped_tokens],
            num_mapped_tokens,
        )
    )
    if num_mapped_tokens < num_tokens:
        logical_positions[num_mapped_tokens:].fill_(-1)
    if circular_buffer_size > 0:
        slot_mapping = circular_qsa_slot_mapping(
            common_attn_metadata.block_table_tensor,
            token_to_req,
            logical_positions,
            circular_buffer_size,
            query_start_loc=common_attn_metadata.query_start_loc,
            out=slot_mapping_buffer,
        )
    elif compress_ratio == 1:
        slot_mapping = common_attn_metadata.slot_mapping[:num_tokens]
    else:
        slot_mapping = compressed_qsa_slot_mapping(
            common_attn_metadata.block_table_tensor,
            token_to_req,
            logical_positions,
            storage_block_size,
            compress_ratio,
            slot_mapping_buffer,
        )
        slot_mapping.masked_fill_(
            common_attn_metadata.slot_mapping[:num_tokens] < 0, -1
        )
    if k_work_metadata_buffer is not None:
        query_lens = (
            common_attn_metadata.query_start_loc[1:]
            - common_attn_metadata.query_start_loc[:-1]
        )
        chunk_starts = common_attn_metadata.seq_lens - query_lens
        num_work_per_request = (
            common_attn_metadata.seq_lens // compress_ratio
            - chunk_starts // compress_ratio
        )
        num_work_per_request = torch.where(
            query_lens > 0, num_work_per_request.clamp_min(1), 0
        )
        k_start_loc = torch.empty(
            query_lens.shape[0] + 1,
            dtype=torch.int32,
            device=query_lens.device,
        )
        k_start_loc[0] = 0
        torch.cumsum(num_work_per_request, 0, out=k_start_loc[1:])
        work = torch.arange(
            k_work_metadata_buffer.shape[0],
            device=k_work_metadata_buffer.device,
        )
        requests = torch.searchsorted(k_start_loc[1:], work, right=True)
        active = work < k_start_loc[-1]
        work_in_request = (
            work - k_start_loc[requests.clamp_max(query_lens.shape[0] - 1)]
        )
        k_work_metadata_buffer[:, 0].copy_(
            torch.where(active, requests, -1).to(torch.int32)
        )
        k_work_metadata_buffer[:, 1].copy_(
            torch.where(active, work_in_request, -1).to(torch.int32)
        )
    return token_to_req, logical_positions, slot_mapping


# Resolve the fallback outside the per-step metadata hot path.
build_qsa_metadata = (
    build_qsa_metadata_triton if HAS_TRITON else _build_qsa_metadata_torch
)


@dataclass
class QSAForwardMetadata(AttentionMetadata):
    """Common per-forward metadata for one QSA side cache."""

    block_table: torch.Tensor
    slot_mapping: torch.Tensor
    seq_lens: torch.Tensor
    query_start_loc: torch.Tensor
    token_to_req: torch.Tensor
    logical_positions: torch.Tensor
    k_work_metadata: torch.Tensor
    num_actual_tokens: int
    storage_block_size: int
    compress_ratio: int


class QSAMetadataBuilder(AttentionMetadataBuilder[QSAForwardMetadata]):
    """Build QSA metadata from vLLM's cache-group-specific common metadata."""

    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.UNIFORM_BATCH

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ) -> None:
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        self.is_circular_buffer = isinstance(kv_cache_spec, CircularBufferSpec)
        if isinstance(kv_cache_spec, MLAAttentionSpec):
            self.compress_ratio = kv_cache_spec.compress_ratio
        else:
            self.compress_ratio = 1
        self.storage_block_size = kv_cache_spec.storage_block_size
        max_tokens = vllm_config.scheduler_config.max_num_batched_tokens
        self.token_to_req_buffer = torch.empty(
            max_tokens, dtype=torch.int32, device=device
        )
        self.slot_mapping_buffer = torch.empty(
            max_tokens, dtype=torch.int64, device=device
        )
        self.logical_positions_buffer = torch.empty(
            max_tokens, dtype=torch.int64, device=device
        )
        max_requests = vllm_config.scheduler_config.max_num_seqs
        self.request_capacity = max_requests
        if not self.is_circular_buffer and self.compress_ratio != 1:
            max_k_work = (
                max_tokens + (self.compress_ratio - 1) * max_requests
            ) // self.compress_ratio
            self.k_work_metadata_buffer = torch.empty(
                max_k_work, 2, dtype=torch.int32, device=device
            )
        else:
            self.k_work_metadata_buffer = torch.empty(
                0, 2, dtype=torch.int32, device=device
            )

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> QSAForwardMetadata:
        del common_prefix_len, fast_build
        num_tokens = common_attn_metadata.num_actual_tokens
        build_k_work = not self.is_circular_buffer and self.compress_ratio != 1
        k_work_metadata = self.k_work_metadata_buffer
        request_capacity = None
        if build_k_work:
            num_requests = common_attn_metadata.query_start_loc.shape[0] - 1
            request_capacity = self.request_capacity
            max_num_work = (
                num_tokens + (self.compress_ratio - 1) * num_requests
            ) // self.compress_ratio
            k_work_metadata = self.k_work_metadata_buffer[:max_num_work]
        token_to_req, logical_positions, slot_mapping = build_qsa_metadata(
            common_attn_metadata,
            self.token_to_req_buffer,
            self.logical_positions_buffer,
            self.slot_mapping_buffer,
            storage_block_size=self.storage_block_size,
            compress_ratio=self.compress_ratio,
            circular_buffer_size=(
                self.kv_cache_spec.block_size if self.is_circular_buffer else 0
            ),
            k_work_metadata_buffer=k_work_metadata if build_k_work else None,
            request_capacity=request_capacity,
        )
        return QSAForwardMetadata(
            block_table=common_attn_metadata.block_table_tensor,
            slot_mapping=slot_mapping,
            seq_lens=common_attn_metadata.seq_lens,
            query_start_loc=common_attn_metadata.query_start_loc,
            token_to_req=token_to_req,
            logical_positions=logical_positions,
            k_work_metadata=k_work_metadata,
            num_actual_tokens=num_tokens,
            storage_block_size=self.storage_block_size,
            compress_ratio=self.compress_ratio,
        )


class QSAStateBackend(AttentionBackend):
    """Key-only dummy backend for out-of-band BF16 QSA side-cache operations."""

    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.bfloat16]
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = ["auto", "bfloat16"]

    @staticmethod
    def get_name() -> str:
        return "QWEN4_EXP_EXP_QSA_STATE"

    @staticmethod
    def get_impl_cls():
        raise NotImplementedError(
            "QSA state caches run out-of-band and have no attention impl"
        )

    @staticmethod
    def get_builder_cls() -> type[QSAMetadataBuilder]:
        return QSAMetadataBuilder

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        del cache_dtype_str
        if num_kv_heads != 1:
            raise ValueError("QSA side caches require exactly one KV head")
        return (num_blocks, block_size, num_kv_heads, head_size)

    @classmethod
    def indexes_kv_by_block_stride(cls) -> bool:
        return True

    @staticmethod
    def get_kv_cache_stride_order(
        include_num_layers_dimension: bool = False,
    ) -> tuple[int, ...]:
        if include_num_layers_dimension:
            return (0, 1, 2, 3, 4)
        return (0, 1, 2, 3)


class _QSAStateCache(nn.Module, AttentionLayerBase):
    supports_dcp = False

    def __init__(
        self,
        *,
        head_size: int,
        dtype: torch.dtype,
        cache_config: CacheConfig,
        prefix: str,
        vllm_config: VllmConfig,
        compress_ratio: int = 1,
    ) -> None:
        super().__init__()
        if head_size <= 0:
            raise ValueError("QSA cache head size must be positive")
        if compress_ratio <= 0:
            raise ValueError("QSA compression ratio must be positive")
        if cache_config.block_size % compress_ratio:
            raise ValueError(
                "QSA cache block size must be divisible by the compression ratio"
            )
        self.head_size = head_size
        self.dtype = dtype
        self.cache_config = cache_config
        self.prefix = prefix
        self.compress_ratio = compress_ratio
        self.kv_cache = torch.tensor([])

        static_context = vllm_config.compilation_config.static_forward_context
        if prefix in static_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        static_context[prefix] = self

    def forward(self) -> None: ...

    def get_attn_backend(self) -> type[AttentionBackend]:
        return QSAStateBackend


class QSAKeyStateCache(_QSAStateCache):
    """Raw BF16 key, optionally followed by exact int64 MRoPE positions."""

    _BF16_PER_INT64 = 4
    _NUM_ROPE_AXES = 3

    def __init__(self, *, cache_rope_positions: bool = False, **kwargs) -> None:
        key_head_size = int(kwargs.pop("head_size"))
        self.key_head_size = key_head_size
        self.cache_rope_positions = bool(cache_rope_positions)
        self.rope_position_offset = (
            (key_head_size + self._BF16_PER_INT64 - 1) // self._BF16_PER_INT64
        ) * self._BF16_PER_INT64
        storage_head_size = key_head_size
        if self.cache_rope_positions:
            storage_head_size = self.rope_position_offset + (
                self._NUM_ROPE_AXES * self._BF16_PER_INT64
            )
        super().__init__(head_size=storage_head_size, **kwargs)

    def bind_kv_cache(self, kv_cache: torch.Tensor) -> None:
        if kv_cache.ndim != 4 or kv_cache.shape[2] != 1:
            raise ValueError("QSA raw cache must be [blocks, block_size, 1, width]")
        if kv_cache.dtype != torch.bfloat16 or kv_cache.shape[3] != self.head_size:
            raise ValueError("QSA raw cache does not match its packed BF16 cache spec")
        super().bind_kv_cache(kv_cache)
        self.key_cache = kv_cache[..., : self.key_head_size]
        if self.cache_rope_positions:
            position_tail = kv_cache[..., self.rope_position_offset :]
            self.rope_position_cache = position_tail.view(torch.int64)
        else:
            self.rope_position_cache = None

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec:
        # Hold the open group's committed keys plus every row a speculative
        # step stores before acceptance is known, rounded up to whole groups so
        # the ring divides the attention block size (it joins the LCM that sets
        # the scheduler block size). Anything narrower lets a rejected draft row
        # overwrite a committed key the next step needs to close the group.
        span = self.compress_ratio + vllm_config.num_speculative_tokens
        capacity = self.compress_ratio * cdiv(span, self.compress_ratio)
        assert self.cache_config.block_size % capacity == 0, (
            f"QSA ring capacity {capacity} must divide the attention block "
            f"size {self.cache_config.block_size}"
        )
        return CircularBufferSpec(
            block_size=capacity,
            num_kv_heads=1,
            head_size=self.head_size,
            head_size_v=0,
            dtype=self.dtype,
        )


class QSACompressedKeyCache(_QSAStateCache):
    """Normalized, group-first-RoPE BF16 key at one row per complete group."""

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec:
        del vllm_config
        return MLAAttentionSpec(
            block_size=self.cache_config.block_size,
            num_kv_heads=1,
            head_size=self.head_size,
            dtype=self.dtype,
            compress_ratio=self.compress_ratio,
        )


__all__ = [
    "QSACompressedKeyCache",
    "QSAForwardMetadata",
    "QSAKeyStateCache",
    "QSAMetadataBuilder",
    "QSAStateBackend",
    "canonical_qsa_rope_positions",
    "circular_qsa_slot_mapping",
    "compressed_qsa_slot_mapping",
]
