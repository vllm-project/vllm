# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tile-union QSA sparse attention for prefill (RFC vllm#55394).

In prefill, consecutive query rows select nearly the same compressed blocks
(Jaccard ~0.9 at 8k context on Qwen3.8-Flash-Next), yet the split-K kernel in
qsa.py gathers every row's selection on its own and runs the GQA dot at
M = one head group. This path groups R consecutive rows of one request into a
tile, iterates the UNION of the tile's selected blocks, gathers each block
once, and applies a per-row membership mask inside the online softmax, so each
row still attends exactly its own selection (results match the split-K kernel
up to summation order).

Dataflow per call (all shapes known on the host, no device -> host sync):

    row -> (tile, slot) map from query_start_loc      small torch ops, int32
    pack kernel:  block ids -> sorted keys input       one read of the ids
    torch.sort over each tile's keys
    build kernel (in place over the sorted keys):      union as physical token
        physical bases, membership, count, tails       bases; tails resolved too
    attention kernel                                   no block-table reads

The algorithm is generic; the tile (rows, blocks per step, warps) is tuned on
SM121 (GB10: 24 MiB L2, 99 KiB shared memory per block), the only part it is
enabled on by default. VLLM_QSA_TILE_UNION=1 forces that tile on any part,
"R,BNB,warps,min_rows" forces an explicit one (bring-up elsewhere), 0 disables.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass

import torch

import vllm.envs as envs
from vllm.logger import init_logger
from vllm.model_executor.warmup.jit_warmup_triton_helper import (
    TritonWarmupTensor,
    triton_scalar_specialization_rep,
)
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton

logger = init_logger(__name__)


@dataclass(frozen=True)
class QSATileUnionConfig:
    rows_per_tile: int  # R: query rows sharing one gather
    blocks_per_step: int  # BNB: union blocks per inner step (BN = BNB * CR tokens)
    num_warps: int
    min_rows: int  # below this the split-K kernel is faster
    # Below this many rows per prefill request on average, the tiles share
    # little and the union build is overhead (fragmented batches).
    min_rows_per_request: int = 64

    def __post_init__(self) -> None:
        # The kernels use tl.arange over R * GP, BNB and TAIL_COLS, and the
        # packed key keeps 3 bits for the row within its tile.
        if self.rows_per_tile not in (1, 2, 4, 8):
            raise ValueError("rows_per_tile must be 1, 2, 4 or 8")
        if not (0 < self.blocks_per_step <= 32) or (
            self.blocks_per_step & (self.blocks_per_step - 1)
        ):
            raise ValueError("blocks_per_step must be a power of two <= 32")
        if self.num_warps not in (1, 2, 4, 8):
            raise ValueError("num_warps must be 1, 2, 4 or 8")
        if self.min_rows < 0 or self.min_rows_per_request < 0:
            raise ValueError("min_rows and min_rows_per_request must be >= 0")


@dataclass(frozen=True)
class QSATileUnionInputs:
    """The indexer's selection before expansion, plus the batch layout.

    block_indices: [rows, block_topk] int32 — the top-k output as written by
    the indexer: valid ids first, then -1 (both NVIDIA top-k kernels pad the
    unused slots with -1, so no visible-block bound is needed here).
    logical_positions: [rows] int64 (the QSA metadata buffer's dtype).
    query_start_loc: [num_requests + 1] int32.
    """

    block_indices: torch.Tensor
    logical_positions: torch.Tensor
    query_start_loc: torch.Tensor
    num_decode_tokens: int
    num_prefills: int
    compress_ratio: int
    token_topk: int
    # (tile_row0, tile_request, num_tiles) from qsa_tile_union_layout, shared
    # by every QSA layer of one forward; computed here when None.
    layout: tuple[torch.Tensor, torch.Tensor, int] | None = None


_TILE_UNION_TABLE: dict[tuple[int, int], QSATileUnionConfig] = {
    # GB10: R=2/BN=32/4 warps beat R=4 at both 8k and 30k context; R=4 at
    # BN=32 spills (M=64 accumulator), BN=128 exceeds the 99 KiB smem budget.
    (12, 1): QSATileUnionConfig(
        rows_per_tile=2, blocks_per_step=8, num_warps=4, min_rows=1024
    ),
}
_TILE_UNION_ROW_BITS = 3  # packed key = block_id << 3 | row_in_tile
_SENTINEL_BLOCK_VALUE = 1 << 27  # sorts after every real block id
_SENTINEL_VALUE = (_SENTINEL_BLOCK_VALUE << _TILE_UNION_ROW_BITS) | (
    (1 << _TILE_UNION_ROW_BITS) - 1
)
# Triton kernels may only read module globals declared as constexpr.
_TILE_UNION_SENTINEL_BLOCK = tl.constexpr(_SENTINEL_BLOCK_VALUE)
_TILE_UNION_SENTINEL = tl.constexpr(_SENTINEL_VALUE)


def _parse_tile_union_config(spec: str) -> QSATileUnionConfig:
    """ "R,BNB,warps,min_rows[,min_rows_per_request]", e.g. "2,8,4,1024" (the
    SM121 tile)."""
    try:
        parts = [int(part) for part in spec.split(",")]
        if len(parts) not in (4, 5):
            raise ValueError
    except ValueError:
        raise ValueError(
            "VLLM_QSA_TILE_UNION must be auto, 0, 1 or "
            f"R,BNB,warps,min_rows[,min_rows_per_request]; got {spec!r}"
        ) from None
    try:
        return QSATileUnionConfig(*parts)
    except ValueError as exc:
        raise ValueError(f"VLLM_QSA_TILE_UNION: invalid tile {spec!r}: {exc}") from None


@functools.cache
def qsa_tile_union_config() -> QSATileUnionConfig | None:
    """The tile for this device, or None (split-K kernel only)."""
    mode = envs.VLLM_QSA_TILE_UNION
    config: QSATileUnionConfig | None
    if mode == "0":
        return None
    if mode == "1" or "," in mode:
        config = (
            _TILE_UNION_TABLE[(12, 1)]
            if mode == "1"
            else _parse_tile_union_config(mode)
        )
        logger.info(
            "QSA tile-union path forced on (VLLM_QSA_TILE_UNION=%s) with tile "
            "%s; not tuned for this device unless you measured it.",
            mode,
            config,
        )
        return config
    if mode != "auto":
        raise ValueError(
            "VLLM_QSA_TILE_UNION must be auto, 0, 1 or R,BNB,warps,min_rows; "
            f"got {mode!r}"
        )
    capability = current_platform.get_device_capability()
    if capability is None:
        return None
    config = _TILE_UNION_TABLE.get((capability.major, capability.minor))
    if config is not None:
        logger.info("QSA tile-union prefill path enabled: %s.", config)
    return config


def qsa_tile_union_static_ok(
    compress_ratio: int, token_topk: int, page_size: int, table_width: int
) -> bool:
    """The model/cache constants the kernels can handle (also gates warmup)."""
    if compress_ratio < 2 or compress_ratio & (compress_ratio - 1):
        return False  # tl.arange(0, CR)
    if token_topk <= 0 or token_topk % compress_ratio:
        return False
    if page_size <= 0 or page_size % compress_ratio:
        return False  # a compressed block must not straddle a page
    # The packed key encodes block ids below the sentinel block.
    return table_width * (page_size // compress_ratio) < _SENTINEL_BLOCK_VALUE


_BLOCK_INDICES_WORKSPACE: dict[tuple[int, int, torch.device], torch.Tensor] = {}


def qsa_tile_union_workspace(
    max_tokens: int, block_topk: int, device: torch.device
) -> torch.Tensor:
    """One [max_tokens, block_topk] int32 selection workspace per device,
    shared by every QSA layer (the selection lives only from a layer's
    indexer call to the same layer's attention, and layers run in sequence;
    reuse steps never read it). 2 KiB per token once, not per layer."""
    key = (max_tokens, block_topk, device)
    workspace = _BLOCK_INDICES_WORKSPACE.get(key)
    if workspace is None:
        workspace = torch.empty(
            max_tokens, block_topk, dtype=torch.int32, device=device
        )
        _BLOCK_INDICES_WORKSPACE[key] = workspace
    return workspace


def _tile_union_tail_cols(config: QSATileUnionConfig, compress_ratio: int) -> int:
    # tl.dot needs N >= 16.
    return max(16, triton.next_power_of_2(config.rows_per_tile * (compress_ratio - 1)))


def qsa_tile_union_eligible(
    inputs: QSATileUnionInputs,
    num_rows: int,
    k_cache: torch.Tensor,
    block_table: torch.Tensor,
    config: QSATileUnionConfig,
) -> bool:
    """Decided from host-side metadata only (no device reads)."""
    if num_rows < config.min_rows or inputs.num_decode_tokens:
        # Decode rows keep the split-K kernel; mixed batches are not tiled.
        return False
    if inputs.num_prefills <= 0 or num_rows < (
        config.min_rows_per_request * inputs.num_prefills
    ):
        # Fragmented prefill batch: tiles would share little.
        return False
    ratio = inputs.compress_ratio
    if not qsa_tile_union_static_ok(
        ratio, inputs.token_topk, k_cache.shape[1], block_table.shape[1]
    ):
        return False
    if k_cache.shape[0] * k_cache.shape[1] >= 2**31:
        return False  # the union stores page * PAGE_SIZE + offset as int32
    block_topk = inputs.token_topk // ratio
    num_requests = block_table.shape[0]
    b, p, qsl = inputs.block_indices, inputs.logical_positions, inputs.query_start_loc
    device = k_cache.device
    return (
        b.shape == (num_rows, block_topk)
        and b.dtype == torch.int32
        and b.device == device
        and b.stride(1) == 1
        and p.shape == (num_rows,)
        and p.dtype == torch.int64
        and p.device == device
        and p.stride(0) == 1
        and qsl.ndim == 1
        and qsl.numel() >= num_requests + 1
        and qsl.dtype == torch.int32
        and qsl.device == device
        and qsl.stride(0) == 1
    )


# ---------------------------------------------------------------------------
# Kernels
# ---------------------------------------------------------------------------


@triton.jit(do_not_specialize=["num_rows", "num_requests"])
def _qsa_tile_union_pack_kernel(
    block_indices_ptr,
    packed_ptr,
    tile_row0_ptr,
    tile_request_ptr,
    query_start_loc_ptr,
    token_to_req_ptr,
    stride_blocks_row,
    stride_packed,
    num_rows,
    num_requests,
    E: tl.constexpr,
    E_PAD: tl.constexpr,
    N: tl.constexpr,
    R: tl.constexpr,
):
    """One program per tile: read each of the tile's rows' block ids once and
    write the sort input (block_id << 3 | slot, or the sentinel for -1 ids,
    padding rows and rows whose request id is invalid); the N - R * E pad
    columns get the sentinel too."""
    tile = tl.program_id(0)
    row0 = tl.load(tile_row0_ptr + tile)
    has_rows = row0 >= 0
    request = tl.load(tile_request_ptr + tile)
    request = tl.minimum(tl.maximum(request, 0), num_requests - 1)
    request_end = tl.load(query_start_loc_ptr + request + 1, mask=has_rows, other=0)
    request_end = tl.where(request == num_requests - 1, num_rows, request_end)
    n_rows = tl.where(has_rows, tl.minimum(R, request_end - row0), 0)
    e = tl.arange(0, E_PAD)
    for r in tl.static_range(R):
        row = tl.maximum(row0, 0) + r
        structural = r < n_rows
        row_request = tl.load(
            token_to_req_ptr + tl.minimum(row, num_rows - 1), mask=structural, other=-1
        )
        live = structural & (row_request == request)
        ids = tl.load(
            block_indices_ptr + row * stride_blocks_row + e,
            mask=live & (e < E),
            other=-1,
        )
        keys = tl.where(ids >= 0, (ids << 3) | r, _TILE_UNION_SENTINEL)
        tl.store(packed_ptr + tile * stride_packed + r * E + e, keys, mask=e < E)
    if N > R * E:
        # N - R * E need not be a power of two (tl.arange requires one).
        pad = tl.arange(0, N)
        tl.store(
            packed_ptr + tile * stride_packed + R * E + pad,
            tl.full((N,), _TILE_UNION_SENTINEL, tl.int32),
            mask=pad < N - R * E,
        )


@triton.jit(
    do_not_specialize=["num_rows", "num_requests", "table_width", "num_cache_blocks"]
)
def _qsa_tile_union_build_kernel(
    keys_ptr,
    mem_ptr,
    cnt_ptr,
    tail_ptr,
    tile_row0_ptr,
    tile_request_ptr,
    query_start_loc_ptr,
    block_table_ptr,
    positions_ptr,
    stride_keys,
    stride_mem_tile,
    stride_mem_row,
    stride_tail,
    stride_table_req,
    num_rows,
    num_requests,
    table_width,
    num_cache_blocks,
    N: tl.constexpr,
    R: tl.constexpr,
    CR: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    TAIL_COLS: tl.constexpr,
):
    """Per tile, from its sorted packed keys (block_id << 3 | row,
    sentinel-padded to exactly N), IN PLACE: keys[0:count] become the union's
    physical token bases (page * PAGE_SIZE + offset of the block's first token,
    -1 if the page is invalid); plus the int8 [R, N] membership matrix, the
    union count, and each row's causal-tail tokens resolved the same way."""
    BPP: tl.constexpr = PAGE_SIZE // CR
    tile = tl.program_id(0)
    i = tl.arange(0, N)
    packed = tl.load(keys_ptr + tile * stride_keys + i)
    prev = tl.load(keys_ptr + tile * stride_keys + i - 1, mask=i > 0, other=-8)
    blk = packed // 8
    r = packed % 8
    valid = blk < _TILE_UNION_SENTINEL_BLOCK
    first = (blk != prev // 8) & valid
    pos = tl.cumsum(first.to(tl.int32)) - 1
    row0 = tl.load(tile_row0_ptr + tile)
    has_rows = row0 >= 0
    request = tl.load(tile_request_ptr + tile)
    request = tl.minimum(tl.maximum(request, 0), num_requests - 1)
    # Page lookup only for first occurrences (the stored ones).
    logical_page = blk // BPP
    page_ok = first & has_rows & (logical_page < table_width)
    physical_page = tl.load(
        block_table_ptr
        + request * stride_table_req
        + tl.minimum(logical_page, table_width - 1),
        mask=page_ok,
        other=-1,
    )
    page_ok &= (physical_page >= 0) & (physical_page < num_cache_blocks)
    phys = tl.where(page_ok, physical_page * PAGE_SIZE + (blk % BPP) * CR, -1)
    # All loads above precede these stores: the in-place rewrite is safe.
    tl.store(keys_ptr + tile * stride_keys + pos, phys, mask=first)
    tl.store(
        mem_ptr + tile * stride_mem_tile + r * stride_mem_row + pos,
        tl.full((N,), 1, tl.int8),
        mask=valid,
    )
    tl.store(cnt_ptr + tile, tl.sum(first.to(tl.int32)))
    # Causal tails (the expansion kernel's rule: start = ((q + 1) // CR) * CR,
    # count = q + 1 - start < CR), resolved to physical tokens here.
    request_end = tl.load(query_start_loc_ptr + request + 1, mask=has_rows, other=0)
    request_end = tl.where(request == num_requests - 1, num_rows, request_end)
    n_rows = tl.where(has_rows, tl.minimum(R, request_end - row0), 0)
    tt = tl.arange(0, TAIL_COLS)
    r_t = tt // (CR - 1)
    j_t = tt % (CR - 1)
    tmask = r_t < n_rows
    # int64 end to end (the positions contract); only the final physical
    # location, bounded by the int32 cache check in eligibility, narrows.
    position = tl.load(
        positions_ptr + tl.minimum(tl.maximum(row0, 0) + r_t, num_rows - 1),
        mask=tmask,
        other=-1,
    )
    tail_start = ((position + 1) // CR) * CR
    tail_count = position + 1 - tail_start
    tail_token = tail_start + j_t
    tail_ok = tmask & (position >= 0) & (j_t < tail_count)
    tail_page = tail_token // PAGE_SIZE
    tail_ok &= tail_page < table_width
    tail_phys_page = tl.load(
        block_table_ptr
        + request * stride_table_req
        + tl.minimum(tail_page, table_width - 1),
        mask=tail_ok,
        other=-1,
    )
    tail_ok &= (tail_phys_page >= 0) & (tail_phys_page < num_cache_blocks)
    tail_phys = tl.where(
        tail_ok, tail_phys_page.to(tl.int64) * PAGE_SIZE + tail_token % PAGE_SIZE, -1
    ).to(tl.int32)
    tl.store(tail_ptr + tile * stride_tail + tt, tail_phys)


@triton.jit(do_not_specialize=["num_rows", "num_requests"])
def _qsa_tile_union_attn_kernel(
    q_ptr,
    k_cache_ptr,
    v_cache_ptr,
    uni_ptr,
    mem_ptr,
    cnt_ptr,
    tail_ptr,
    tile_row0_ptr,
    tile_request_ptr,
    query_start_loc_ptr,
    token_to_req_ptr,
    out_ptr,
    stride_q_row,
    stride_q_head,
    stride_k_block,
    stride_k_token,
    stride_k_head,
    stride_v_block,
    stride_v_token,
    stride_v_head,
    stride_uni,
    stride_mem_tile,
    stride_mem_row,
    stride_tail,
    stride_out_row,
    stride_out_head,
    num_rows,
    num_requests,
    R: tl.constexpr,
    GP: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BNB: tl.constexpr,
    CR: tl.constexpr,
    TAIL_COLS: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
):
    tile = tl.program_id(0)
    kv_head = tl.program_id(1)
    M: tl.constexpr = R * GP
    BN: tl.constexpr = BNB * CR
    TAIL_PER_ROW: tl.constexpr = CR - 1
    m_off = tl.arange(0, M)
    r_of_m = m_off // GP
    h_of_m = m_off % GP
    dim_offsets = tl.arange(0, HEAD_DIM)
    b_off = tl.arange(0, BNB)
    j_off = tl.arange(0, CR)
    row0 = tl.load(tile_row0_ptr + tile)
    has_rows = row0 >= 0
    request = tl.load(tile_request_ptr + tile)
    request = tl.minimum(tl.maximum(request, 0), num_requests - 1)
    request_end = tl.load(query_start_loc_ptr + request + 1, mask=has_rows, other=0)
    request_end = tl.where(request == num_requests - 1, num_rows, request_end)
    n_rows = tl.where(has_rows, tl.minimum(R, request_end - row0), 0)
    row = tl.maximum(row0, 0) + r_of_m
    rmask = r_of_m < n_rows
    # rmask: the tile's structural rows (always written, zeros if masked);
    # live: rows whose request id is valid. Split-K contract: a row whose
    # request id is invalid (padding) is masked and written as zeros.
    row_request = tl.load(
        token_to_req_ptr + tl.minimum(row, num_rows - 1), mask=rmask, other=-1
    )
    live = rmask & (row_request == request)
    qmask = live & (h_of_m < GROUP_SIZE)
    store_mask = rmask & (h_of_m < GROUP_SIZE)
    first_head = kv_head * GROUP_SIZE
    query = tl.load(
        q_ptr
        + row[:, None] * stride_q_row
        + (first_head + h_of_m[:, None]) * stride_q_head
        + dim_offsets[None, :],
        mask=qmask[:, None],
        other=0.0,
    )
    max_value = tl.full((M,), -1.0e20, dtype=tl.float32)
    normalizer = tl.zeros((M,), dtype=tl.float32)
    accumulator = tl.zeros((M, HEAD_DIM), dtype=tl.float32)
    softmax_scale_log2: tl.constexpr = (HEAD_DIM**-0.5) * 1.4426950408889634
    # Pass 1: the tile's union of whole compressed blocks, gathered once,
    # per-row membership from the int8 matrix.
    ubound = tl.load(cnt_ptr + tile)
    for t in range(0, ubound, BNB):
        emask = (t + b_off) < ubound
        phys = tl.load(uni_ptr + tile * stride_uni + t + b_off, mask=emask, other=-1)
        tok2 = tl.where((phys >= 0)[:, None], phys[:, None] + j_off[None, :], -1)
        physical_token = tl.reshape(tok2, (BN,))
        valid = physical_token >= 0
        safe_token = tl.maximum(physical_token, 0)
        # A block never straddles a page (PAGE_SIZE % CR == 0), so page and
        # offset come back from the base without a table lookup.
        safe_page = (safe_token // PAGE_SIZE).to(tl.int64)
        page_offset = safe_token % PAGE_SIZE
        keys = tl.load(
            k_cache_ptr
            + safe_page[None, :] * stride_k_block
            + page_offset[None, :] * stride_k_token
            + kv_head * stride_k_head
            + dim_offsets[:, None],
            mask=valid[None, :],
            other=0.0,
        )
        values = tl.load(
            v_cache_ptr
            + safe_page[:, None] * stride_v_block
            + page_offset[:, None] * stride_v_token
            + kv_head * stride_v_head
            + dim_offsets[None, :],
            mask=valid[:, None],
            other=0.0,
        )
        memb = tl.load(
            mem_ptr
            + tile * stride_mem_tile
            + r_of_m[:, None] * stride_mem_row
            + t
            + b_off[None, :],
            mask=emask[None, :],
            other=0,
        )
        memt = tl.reshape(tl.broadcast_to(memb[:, :, None], (M, BNB, CR)), (M, BN))
        active = (memt > 0) & valid[None, :] & live[:, None]
        scores = tl.dot(query, keys) * softmax_scale_log2
        scores = tl.where(active, scores, -1.0e20)
        next_max = tl.maximum(max_value, tl.max(scores, axis=1))
        alpha = tl.math.exp2(max_value - next_max)
        probabilities = tl.where(active, tl.math.exp2(scores - next_max[:, None]), 0.0)
        accumulator = tl.dot(
            probabilities.to(values.dtype), values, acc=accumulator * alpha[:, None]
        )
        normalizer = normalizer * alpha + tl.sum(probabilities, axis=1)
        max_value = next_max
    # Pass 2: each row's causal tail (< CR tokens of its open block), already
    # physical.
    tt = tl.arange(0, TAIL_COLS)
    slot_row = tt // TAIL_PER_ROW
    tail_phys = tl.load(
        tail_ptr + tile * stride_tail + tt, mask=tt < R * TAIL_PER_ROW, other=-1
    )
    valid = tail_phys >= 0
    safe_token = tl.maximum(tail_phys, 0)
    safe_page = (safe_token // PAGE_SIZE).to(tl.int64)
    page_offset = safe_token % PAGE_SIZE
    keys = tl.load(
        k_cache_ptr
        + safe_page[None, :] * stride_k_block
        + page_offset[None, :] * stride_k_token
        + kv_head * stride_k_head
        + dim_offsets[:, None],
        mask=valid[None, :],
        other=0.0,
    )
    values = tl.load(
        v_cache_ptr
        + safe_page[:, None] * stride_v_block
        + page_offset[:, None] * stride_v_token
        + kv_head * stride_v_head
        + dim_offsets[None, :],
        mask=valid[:, None],
        other=0.0,
    )
    active = (r_of_m[:, None] == slot_row[None, :]) & valid[None, :] & live[:, None]
    scores = tl.dot(query, keys) * softmax_scale_log2
    scores = tl.where(active, scores, -1.0e20)
    next_max = tl.maximum(max_value, tl.max(scores, axis=1))
    alpha = tl.math.exp2(max_value - next_max)
    probabilities = tl.where(active, tl.math.exp2(scores - next_max[:, None]), 0.0)
    accumulator = tl.dot(
        probabilities.to(values.dtype), values, acc=accumulator * alpha[:, None]
    )
    normalizer = normalizer * alpha + tl.sum(probabilities, axis=1)
    has_values = normalizer > 0
    result = tl.where(
        has_values[:, None], accumulator / tl.maximum(normalizer[:, None], 1.0e-20), 0.0
    )
    tl.store(
        out_ptr
        + row[:, None] * stride_out_row
        + (first_head + h_of_m[:, None]) * stride_out_head
        + dim_offsets[None, :],
        result.to(tl.bfloat16),
        mask=store_mask[:, None],
    )


# ---------------------------------------------------------------------------
# Host side
# ---------------------------------------------------------------------------


def qsa_tile_union_layout(
    query_start_loc: torch.Tensor, num_rows: int, num_requests: int, R: int
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """row -> (tile, slot) from the batch layout alone: rows are contiguous per
    request and each request starts a fresh tile, so tiles never straddle
    requests. Rows past query_start_loc[-1] (padding) belong structurally to
    the last request and are masked by token_to_req in the kernels.
    Returns tile_row0 [T] int32 (-1 = unused tile), tile_request [T] int32, T.
    Everything is int32 and sync-free."""
    device = query_start_loc.device
    qsl = query_start_loc[: num_requests + 1]
    row = torch.arange(num_rows, device=device, dtype=torch.int32)
    request = torch.searchsorted(qsl[1:], row, right=True, out_int32=True).clamp_(
        max=num_requests - 1
    )
    lengths = qsl[1:] - qsl[:-1]
    tiles_per_request = (lengths + R - 1) // R
    tile_base = (
        torch.cumsum(tiles_per_request, 0, dtype=torch.int32) - tiles_per_request
    )
    offset = row - qsl[request]
    tile = tile_base[request] + offset // R
    slot = offset % R
    num_tiles = (num_rows + R - 1) // R + num_requests  # >= sum(ceil(len / R))
    # One junk entry at index num_tiles absorbs the non-start rows.
    scatter_index = torch.where(slot == 0, tile, num_tiles).to(torch.int64)
    tile_row0 = torch.full((num_tiles + 1,), -1, dtype=torch.int32, device=device)
    tile_row0.scatter_(0, scatter_index, row)
    tile_request = torch.zeros(num_tiles + 1, dtype=torch.int32, device=device)
    tile_request.scatter_(0, scatter_index, request)
    return tile_row0[:num_tiles], tile_request[:num_tiles], num_tiles


def qsa_tile_union_attention(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_table: torch.Tensor,
    token_to_req: torch.Tensor,
    out: torch.Tensor,
    inputs: QSATileUnionInputs,
    config: QSATileUnionConfig,
) -> torch.Tensor:
    rows = q.shape[0]
    R = config.rows_per_tile
    ratio = inputs.compress_ratio
    block_topk = inputs.token_topk // ratio
    num_requests = block_table.shape[0]
    device = q.device
    if inputs.layout is not None:
        tile_row0, tile_request, num_tiles = inputs.layout
    else:
        tile_row0, tile_request, num_tiles = qsa_tile_union_layout(
            inputs.query_start_loc, rows, num_requests, R
        )
    N = triton.next_power_of_2(R * block_topk)
    keys = torch.empty((num_tiles, N), dtype=torch.int32, device=device)
    _qsa_tile_union_pack_kernel[(num_tiles,)](
        inputs.block_indices,
        keys,
        tile_row0,
        tile_request,
        inputs.query_start_loc,
        token_to_req,
        inputs.block_indices.stride(0),
        keys.stride(0),
        rows,
        num_requests,
        E=block_topk,
        E_PAD=triton.next_power_of_2(block_topk),
        N=N,
        R=R,
        num_warps=4,
    )
    # Keys-only would do; torch.sort also materialises the index tensor.
    keys = torch.sort(keys, dim=1).values
    tail_cols = _tile_union_tail_cols(config, ratio)
    mem = torch.zeros((num_tiles, R, N), dtype=torch.int8, device=device)
    cnt = torch.empty(num_tiles, dtype=torch.int32, device=device)
    tails = torch.empty((num_tiles, tail_cols), dtype=torch.int32, device=device)
    _qsa_tile_union_build_kernel[(num_tiles,)](
        keys,
        mem,
        cnt,
        tails,
        tile_row0,
        tile_request,
        inputs.query_start_loc,
        block_table,
        inputs.logical_positions,
        keys.stride(0),
        mem.stride(0),
        mem.stride(1),
        tails.stride(0),
        block_table.stride(0),
        rows,
        num_requests,
        block_table.shape[1],
        k_cache.shape[0],
        N=N,
        R=R,
        CR=ratio,
        PAGE_SIZE=k_cache.shape[1],
        TAIL_COLS=tail_cols,
        num_warps=4,
    )
    group_size = q.shape[1] // k_cache.shape[2]
    _qsa_tile_union_attn_kernel[(num_tiles, k_cache.shape[2])](
        q,
        k_cache,
        v_cache,
        keys,  # in place: the union's physical bases
        mem,
        cnt,
        tails,
        tile_row0,
        tile_request,
        inputs.query_start_loc,
        token_to_req,
        out,
        q.stride(0),
        q.stride(1),
        k_cache.stride(0),
        k_cache.stride(1),
        k_cache.stride(2),
        v_cache.stride(0),
        v_cache.stride(1),
        v_cache.stride(2),
        keys.stride(0),
        mem.stride(0),
        mem.stride(1),
        tails.stride(0),
        out.stride(0),
        out.stride(1),
        rows,
        num_requests,
        R=R,
        GP=triton.next_power_of_2(group_size),
        GROUP_SIZE=group_size,
        HEAD_DIM=q.shape[2],
        BNB=config.blocks_per_step,
        CR=ratio,
        TAIL_COLS=tail_cols,
        PAGE_SIZE=k_cache.shape[1],
        num_warps=config.num_warps,
        num_stages=1,
    )
    return out


def warmup_qsa_tile_union(
    kv_cache: torch.Tensor,
    block_table: torch.Tensor,
    *,
    num_query_heads: int,
    compress_ratio: int,
    token_topk: int,
    config: QSATileUnionConfig,
) -> tuple[int, int, int] | None:
    """Compile the three tile-union kernels for this deployment's constants;
    None when the model/cache constants are outside what they handle."""
    head_dim = kv_cache.shape[-1] // 2
    key_cache, value_cache = kv_cache.transpose(1, 2).split(head_dim, dim=-1)
    if not qsa_tile_union_static_ok(
        compress_ratio, token_topk, key_cache.shape[1], block_table.shape[1]
    ):
        return None
    num_kv_heads = key_cache.shape[2]
    group_size = num_query_heads // num_kv_heads
    R = config.rows_per_tile
    block_topk = token_topk // compress_ratio
    N = triton.next_power_of_2(R * block_topk)
    tail_cols = _tile_union_tail_cols(config, compress_ratio)
    num_tiles = 4
    num_rows = num_tiles * R
    num_requests = 2

    i32 = torch.int32
    block_indices_ptr = TritonWarmupTensor(i32, shape=(num_rows, block_topk))
    keys_ptr = TritonWarmupTensor(i32, shape=(num_tiles, N))
    mem_ptr = TritonWarmupTensor(torch.int8, shape=(num_tiles, R, N))
    cnt_ptr = TritonWarmupTensor(i32)
    tail_ptr = TritonWarmupTensor(i32, shape=(num_tiles, tail_cols))
    tile_row0_ptr = TritonWarmupTensor(i32)
    tile_request_ptr = TritonWarmupTensor(i32)
    qsl_ptr = TritonWarmupTensor(i32)
    token_to_req_ptr = TritonWarmupTensor(i32)
    block_table_ptr = TritonWarmupTensor(
        block_table.dtype,
        shape=tuple(block_table.shape),
        strides=tuple(block_table.stride()),
    )
    positions_ptr = TritonWarmupTensor(torch.int64)
    q_ptr = TritonWarmupTensor(
        torch.bfloat16, shape=(num_rows, num_query_heads, head_dim)
    )
    k_cache_ptr = TritonWarmupTensor(
        key_cache.dtype, shape=tuple(key_cache.shape), strides=tuple(key_cache.stride())
    )
    v_cache_ptr = TritonWarmupTensor(
        value_cache.dtype,
        shape=tuple(value_cache.shape),
        strides=tuple(value_cache.stride()),
    )
    out_ptr = TritonWarmupTensor(
        torch.bfloat16, shape=(num_rows, num_query_heads, head_dim)
    )
    num_cache_blocks = triton_scalar_specialization_rep(kv_cache.shape[0])
    head_stride = head_dim
    row_stride = num_query_heads * head_dim

    _qsa_tile_union_pack_kernel.warmup(
        block_indices_ptr,
        keys_ptr,
        tile_row0_ptr,
        tile_request_ptr,
        qsl_ptr,
        token_to_req_ptr,
        block_topk,
        N,
        num_rows,
        num_requests,
        E=block_topk,
        E_PAD=triton.next_power_of_2(block_topk),
        N=N,
        R=R,
        num_warps=4,
        grid=(num_tiles,),
    )
    _qsa_tile_union_build_kernel.warmup(
        keys_ptr,
        mem_ptr,
        cnt_ptr,
        tail_ptr,
        tile_row0_ptr,
        tile_request_ptr,
        qsl_ptr,
        block_table_ptr,
        positions_ptr,
        N,
        R * N,
        N,
        tail_cols,
        block_table.stride(0),
        num_rows,
        num_requests,
        block_table.shape[1],
        num_cache_blocks,
        N=N,
        R=R,
        CR=compress_ratio,
        PAGE_SIZE=key_cache.shape[1],
        TAIL_COLS=tail_cols,
        num_warps=4,
        grid=(num_tiles,),
    )
    _qsa_tile_union_attn_kernel.warmup(
        q_ptr,
        k_cache_ptr,
        v_cache_ptr,
        keys_ptr,
        mem_ptr,
        cnt_ptr,
        tail_ptr,
        tile_row0_ptr,
        tile_request_ptr,
        qsl_ptr,
        token_to_req_ptr,
        out_ptr,
        row_stride,
        head_stride,
        key_cache.stride(0),
        key_cache.stride(1),
        key_cache.stride(2),
        value_cache.stride(0),
        value_cache.stride(1),
        value_cache.stride(2),
        N,
        R * N,
        N,
        tail_cols,
        row_stride,
        head_stride,
        num_rows,
        num_requests,
        R=R,
        GP=triton.next_power_of_2(group_size),
        GROUP_SIZE=group_size,
        HEAD_DIM=head_dim,
        BNB=config.blocks_per_step,
        CR=compress_ratio,
        TAIL_COLS=tail_cols,
        PAGE_SIZE=key_cache.shape[1],
        num_warps=config.num_warps,
        num_stages=1,
        grid=(num_tiles, num_kv_heads),
    )
    return (R, config.blocks_per_step * compress_ratio, config.num_warps)
