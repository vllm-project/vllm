# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Blackwell (sm_100/sm_103) fp8 path for large QSA sparse-attention batches.

A slot pass translates each row's logical tokens through the page table once
for all KV heads, then a single-pass kernel attends over the selected tokens,
gathering K/V as int16 pairs of e4m3 codes that feed the MMAs without
byte-wise shared-memory loads. qsa.py's qsa_sparse_paged_attention dispatches
here via use_fp8_decode_kernel; every QK/PV product is the split-K kernel's
(QK in bf16, PV in fp16, both exact) -- only fp32 summation grouping differs.
"""

import torch

from vllm.model_executor.warmup.jit_warmup_triton_helper import (
    TritonWarmupTensor,
    triton_scalar_specialization_rep,
)
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton


def _is_sm100_family() -> bool:
    """True on sm_100/sm_103 (B200/B300): enables the fp8 decode kernel."""
    return current_platform.is_device_capability_family(100)


@triton.jit(do_not_specialize=["num_requests"])
def _qsa_physical_slots_kernel(
    indices_ptr,
    block_table_ptr,
    token_to_req_ptr,
    slots_ptr,
    stride_indices_row,
    stride_table_req,
    stride_slots_row,
    num_cache_blocks,
    num_requests,
    TOPK: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    PAGE_TABLE_WIDTH: tl.constexpr,
    SLOT_COLUMNS: tl.constexpr,
    OFFSET_BITS: tl.constexpr,
    BLOCK: tl.constexpr,
) -> None:
    """Translate a row's logical tokens through the page table once, into
    page << OFFSET_BITS | page_offset slots shared by every KV head. An entry
    the split-K kernel masks, and the padding past TOPK, becomes -1."""
    row = tl.program_id(0)
    columns = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    request = tl.load(token_to_req_ptr + row)
    safe_request = tl.minimum(tl.maximum(request, 0), num_requests - 1)
    logical_token = tl.load(
        indices_ptr + row * stride_indices_row + columns,
        mask=columns < TOPK,
        other=-1,
    )
    safe_token = tl.maximum(logical_token, 0)
    logical_page = safe_token // PAGE_SIZE
    page_offset = safe_token % PAGE_SIZE
    valid = (
        (request >= 0)
        & (request < num_requests)
        & (logical_token >= 0)
        & (logical_page < PAGE_TABLE_WIDTH)
    )
    physical_page = tl.load(
        block_table_ptr
        + safe_request * stride_table_req
        + tl.minimum(logical_page, PAGE_TABLE_WIDTH - 1),
        mask=valid,
        other=-1,
    )
    valid &= (physical_page >= 0) & (physical_page < num_cache_blocks)
    slot = tl.where(valid, (physical_page << OFFSET_BITS) | page_offset, -1)
    tl.store(
        slots_ptr + row * stride_slots_row + columns,
        slot,
        mask=columns < SLOT_COLUMNS,
    )


@triton.jit
def _e4m3_pairs_to_fp16(pairs, HIGH: tl.constexpr):
    """fp16 of one byte of each int16 pair of e4m3 codes (byte 0 = dim 2i,
    byte 1 = dim 2i + 1); exact."""
    if HIGH:
        return tl.inline_asm_elementwise(
            asm="""
            {
            .reg .b32 t;
            .reg .b16 lo, hi;
            prmt.b32 t, $1, 0, 0x3120;
            mov.b32 {lo, hi}, t;
            cvt.rn.f16x2.e4m3x2 $0, hi;
            }
            """,
            constraints="=r,r",
            args=[pairs],
            dtype=tl.float16,
            is_pure=True,
            pack=2,
        )
    else:
        return tl.inline_asm_elementwise(
            asm="""
            {
            .reg .b32 t;
            .reg .b16 lo, hi;
            prmt.b32 t, $1, 0, 0x3120;
            mov.b32 {lo, hi}, t;
            cvt.rn.f16x2.e4m3x2 $0, lo;
            }
            """,
            constraints="=r,r",
            args=[pairs],
            dtype=tl.float16,
            is_pure=True,
            pack=2,
        )


@triton.jit
def _qsa_sparse_paged_decode_kernel(
    q_ptr,
    k_pairs_ptr,
    v_pairs_ptr,
    slots_ptr,
    indices_ptr,
    output_ptr,
    softmax_scale,
    output_scale,
    output_gate_ptr,
    stride_q_row,
    stride_q_head,
    stride_kv_block,
    stride_kv_token,
    stride_k_head,
    stride_v_head,
    stride_indices_row,
    stride_slots_row,
    stride_output_row,
    stride_output_head,
    stride_output_gate_row,
    stride_output_gate_head,
    TOPK: tl.constexpr,
    OFFSET_BITS: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    NUM_TILES: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    NUM_STAGES: tl.constexpr,
) -> None:
    """The split-K kernel's NUM_SPLITS == 1 math over precomputed slots.

    K and V are gathered as int16 pairs of e4m3 codes, which feed the MMA
    operands without byte-wise shared-memory loads, and split exactly into
    even/odd-dim halves: scores = Q_even K_even + Q_odd K_odd in bf16 (the
    split-K kernel's QK dtype) and O = [P V_even, P V_odd] in fp16. Every
    product is the split-K kernel's; only the fp32 summation grouping differs.
    """
    HALF: tl.constexpr = HEAD_DIM // 2
    row = tl.program_id(0)
    kv_head = tl.program_id(1)
    valid_count = tl.load(indices_ptr + row * stride_indices_row + TOPK)

    head_offsets = tl.arange(0, BLOCK_M)
    dim_offsets = tl.arange(0, HEAD_DIM)
    pair_offsets = tl.arange(0, HALF)
    column_offsets = tl.arange(0, BLOCK_N)
    first_head = kv_head * GROUP_SIZE
    head_mask = head_offsets[:, None] < GROUP_SIZE
    q_pairs = (
        q_ptr
        + row * stride_q_row
        + (first_head + head_offsets[:, None]) * stride_q_head
        + 2 * pair_offsets[None, :]
    )
    # QK stays in bf16 like the split-K kernel: e4m3 -> bf16 is exact, so the
    # half-dots multiply the same values; only fp32 summation regroups.
    query_even = tl.load(q_pairs, mask=head_mask, other=0.0)
    query_odd = tl.load(q_pairs + 1, mask=head_mask, other=0.0)
    score_scale = softmax_scale * 1.4426950408889634

    max_value = tl.full((BLOCK_M,), -1.0e20, dtype=tl.float32)
    normalizer = tl.zeros((BLOCK_M,), dtype=tl.float32)
    acc_even = tl.zeros((BLOCK_M, HALF), dtype=tl.float32)
    acc_odd = tl.zeros((BLOCK_M, HALF), dtype=tl.float32)

    tile_end = tl.minimum(NUM_TILES, tl.cdiv(tl.minimum(valid_count, TOPK), BLOCK_N))
    slot_row_ptr = slots_ptr + row * stride_slots_row
    k_head_ptr = k_pairs_ptr + kv_head * stride_k_head
    v_head_ptr = v_pairs_ptr + kv_head * stride_v_head
    for tile in tl.range(0, tile_end, num_stages=NUM_STAGES):
        slot = tl.load(slot_row_ptr + tile * BLOCK_N + column_offsets)
        valid = slot >= 0
        safe_slot = tl.maximum(slot, 0)
        # int64: the block stride spans every layer of the KV pool.
        page = (safe_slot >> OFFSET_BITS).to(tl.int64)
        page_offset = (safe_slot & ((1 << OFFSET_BITS) - 1)).to(tl.int64)
        token_offset = page * stride_kv_block + page_offset * stride_kv_token
        # Cache rows start on 16 bytes (the launcher asserts the strides).
        token_offset = tl.multiple_of(token_offset, 8)
        pair_pointers = token_offset[:, None] + pair_offsets[None, :]
        key_pairs = tl.trans(
            tl.load(k_head_ptr + pair_pointers, mask=valid[:, None], other=0)
        )
        value_pairs = tl.load(v_head_ptr + pair_pointers, mask=valid[:, None], other=0)
        scores = tl.dot(
            query_even, _e4m3_pairs_to_fp16(key_pairs, False).to(tl.bfloat16)
        ) + tl.dot(query_odd, _e4m3_pairs_to_fp16(key_pairs, True).to(tl.bfloat16))
        scores *= score_scale
        scores = tl.where(valid[None, :], scores, -1.0e20)
        next_max = tl.maximum(max_value, tl.max(scores, axis=1))
        alpha = tl.math.exp2(max_value - next_max)
        probabilities = tl.where(
            valid[None, :], tl.math.exp2(scores - next_max[:, None]), 0.0
        )
        probabilities_fp16 = probabilities.to(tl.float16)
        acc_even = tl.dot(
            probabilities_fp16,
            _e4m3_pairs_to_fp16(value_pairs, False),
            acc=acc_even * alpha[:, None],
        )
        acc_odd = tl.dot(
            probabilities_fp16,
            _e4m3_pairs_to_fp16(value_pairs, True),
            acc=acc_odd * alpha[:, None],
        )
        normalizer = normalizer * alpha + tl.sum(probabilities, axis=1)
        max_value = next_max

    # Epilogue as the split-K kernel's NUM_SPLITS == 1 path.
    has_values = normalizer > 0
    inv_normalizer = output_scale / tl.maximum(normalizer, 1.0e-20)
    output_even = tl.where(has_values[:, None], acc_even * inv_normalizer[:, None], 0.0)
    output_odd = tl.where(has_values[:, None], acc_odd * inv_normalizer[:, None], 0.0)
    normalized_output = tl.reshape(
        tl.join(
            output_even.to(output_ptr.dtype.element_ty),
            output_odd.to(output_ptr.dtype.element_ty),
        ),
        (BLOCK_M, HEAD_DIM),
    )
    output_gate = tl.load(
        output_gate_ptr
        + row * stride_output_gate_row
        + (first_head + head_offsets[:, None]) * stride_output_gate_head
        + dim_offsets[None, :],
        mask=head_mask,
        other=0.0,
    ).to(tl.float32)
    normalized_output = normalized_output.to(tl.float32) * tl.sigmoid(output_gate)
    tl.store(
        output_ptr
        + row * stride_output_row
        + (first_head + head_offsets[:, None]) * stride_output_head
        + dim_offsets[None, :],
        normalized_output,
        mask=head_mask,
    )


# At or below this many (row, KV-head) programs, split-K is faster (GB200).
_DECODE_MIN_PROGRAMS = 64


def use_fp8_decode_kernel(base_programs: int, cache_dtype: torch.dtype) -> bool:
    """Large Blackwell fp8 batches take qsa_sparse_paged_decode.

    Prefill included: measured on GB200 the kernel wins at prefill shapes too
    (1.4x; 2.2-2.6x at decode/verify shapes), so the gate is purely the size
    crossover against split-K.
    """
    return (
        cache_dtype == torch.float8_e4m3fn
        and base_programs > _DECODE_MIN_PROGRAMS
        and _is_sm100_family()
    )


# Decode-kernel launch shape: 64-token tiles, 4 warps, a 5-deep cp.async
# pipeline.
_DECODE_BLOCK_N = 64
_DECODE_NUM_WARPS = 4
_DECODE_NUM_STAGES = 5
_SLOTS_BLOCK = 1024


def _decode_slot_layout(
    num_cache_blocks: int, page_size: int, selection_width: int
) -> tuple[int, int, int]:
    """(offset_bits, num_tiles, slot_columns) of the decode kernel's slots."""
    offset_bits = max(1, (page_size - 1).bit_length())
    assert num_cache_blocks << offset_bits < 2**31
    num_tiles = triton.cdiv(selection_width, _DECODE_BLOCK_N)
    return offset_bits, num_tiles, num_tiles * _DECODE_BLOCK_N


def qsa_sparse_paged_decode(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    logical_indices: torch.Tensor,
    block_table: torch.Tensor,
    token_to_req: torch.Tensor,
    out: torch.Tensor,
    output_gate: torch.Tensor,
    softmax_scale: float,
    output_scale: float,
) -> torch.Tensor:
    """Sparse GQA over an fp8-e4m3 cache for large batches on Blackwell.

    A slot pass resolves each row's page-table lookups once for every KV head,
    so the attention loop's gathers depend on one load instead of two, and the
    int16-pair K/V operands take 4 warps without spilling.
    """
    # e4m3 code pairs; the cache views' strides and V's offset are even.
    k_pairs = k_cache.view(torch.int16)
    v_pairs = v_cache.view(torch.int16)
    # Cache rows start on 16 bytes, which the gathers' vector width relies on.
    assert k_pairs.stride(0) % 8 == 0 and k_pairs.stride(1) % 8 == 0
    assert v_pairs.stride()[:2] == k_pairs.stride()[:2]
    num_rows, num_query_heads, head_dim = q.shape
    num_cache_blocks, page_size, num_kv_heads = k_cache.shape[:3]
    selection_width = logical_indices.shape[1] - 1
    offset_bits, num_tiles, slot_columns = _decode_slot_layout(
        num_cache_blocks, page_size, selection_width
    )
    slots = torch.empty((num_rows, slot_columns), dtype=torch.int32, device=q.device)
    _qsa_physical_slots_kernel[(num_rows, triton.cdiv(slot_columns, _SLOTS_BLOCK))](
        logical_indices,
        block_table,
        token_to_req,
        slots,
        logical_indices.stride(0),
        block_table.stride(0),
        slots.stride(0),
        num_cache_blocks,
        block_table.shape[0],
        TOPK=selection_width,
        PAGE_SIZE=page_size,
        PAGE_TABLE_WIDTH=block_table.shape[1],
        SLOT_COLUMNS=slot_columns,
        OFFSET_BITS=offset_bits,
        BLOCK=_SLOTS_BLOCK,
        num_warps=4,
    )
    group_size = num_query_heads // num_kv_heads
    _qsa_sparse_paged_decode_kernel[(num_rows, num_kv_heads)](
        q,
        k_pairs,
        v_pairs,
        slots,
        logical_indices,
        out,
        softmax_scale,
        output_scale,
        output_gate,
        q.stride(0),
        q.stride(1),
        k_pairs.stride(0),
        k_pairs.stride(1),
        k_pairs.stride(2),
        v_pairs.stride(2),
        logical_indices.stride(0),
        slots.stride(0),
        out.stride(0),
        out.stride(1),
        output_gate.stride(0),
        output_gate.stride(1),
        TOPK=selection_width,
        OFFSET_BITS=offset_bits,
        GROUP_SIZE=group_size,
        HEAD_DIM=head_dim,
        NUM_TILES=num_tiles,
        BLOCK_M=triton.next_power_of_2(group_size),
        BLOCK_N=_DECODE_BLOCK_N,
        NUM_STAGES=_DECODE_NUM_STAGES,
        num_warps=_DECODE_NUM_WARPS,
        num_stages=_DECODE_NUM_STAGES,
    )
    return out


def warmup_qsa_fp8_decode(
    kv_cache: torch.Tensor,
    block_table: torch.Tensor,
    *,
    num_query_heads: int,
    selection_width: int,
) -> tuple[tuple[str, int, int], ...]:
    """Compile the slot-pass and decode-kernel specializations (no-op off sm_100)."""
    head_dim = kv_cache.shape[-1] // 2
    key_cache, value_cache = kv_cache.transpose(1, 2).split(head_dim, dim=-1)
    is_fp8 = kv_cache.dtype == torch.uint8
    cache_dtype = (
        (
            torch.float8_e4m3fn
            if current_platform.has_device_capability(89)
            else torch.uint8
        )
        if is_fp8
        else key_cache.dtype
    )
    if not use_fp8_decode_kernel(_DECODE_MIN_PROGRAMS + 1, cache_dtype):
        return ()
    num_kv_heads = key_cache.shape[2]
    group_size = num_query_heads // num_kv_heads
    block_m = triton.next_power_of_2(group_size)
    num_rows = 16
    num_requests = 16
    q_ptr = TritonWarmupTensor(
        torch.bfloat16, shape=(num_rows, num_query_heads, head_dim)
    )
    indices_ptr = TritonWarmupTensor(torch.int32, shape=(num_rows, selection_width + 1))
    block_table_ptr = TritonWarmupTensor(
        block_table.dtype,
        shape=tuple(block_table.shape),
        strides=tuple(block_table.stride()),
    )
    token_to_req_ptr = TritonWarmupTensor(torch.int32)
    output_ptr = TritonWarmupTensor(
        torch.bfloat16, shape=(num_rows, num_query_heads, head_dim)
    )
    output_gate_ptr = TritonWarmupTensor(
        torch.bfloat16, shape=(num_rows, num_query_heads, head_dim)
    )
    row_stride = num_query_heads * head_dim
    head_stride = head_dim
    num_cache_blocks = triton_scalar_specialization_rep(kv_cache.shape[0])
    key_pairs = key_cache.view(torch.int16)
    value_pairs = value_cache.view(torch.int16)
    offset_bits, num_tiles, slot_columns = _decode_slot_layout(
        kv_cache.shape[0], key_cache.shape[1], selection_width
    )
    slots_ptr = TritonWarmupTensor(torch.int32, shape=(num_rows, slot_columns))
    _qsa_physical_slots_kernel.warmup(
        indices_ptr,
        block_table_ptr,
        token_to_req_ptr,
        slots_ptr,
        selection_width + 1,
        block_table.stride(0),
        slot_columns,
        num_cache_blocks,
        num_requests,
        TOPK=selection_width,
        PAGE_SIZE=key_cache.shape[1],
        PAGE_TABLE_WIDTH=block_table.shape[1],
        SLOT_COLUMNS=slot_columns,
        OFFSET_BITS=offset_bits,
        BLOCK=_SLOTS_BLOCK,
        num_warps=4,
        grid=(num_rows, triton.cdiv(slot_columns, _SLOTS_BLOCK)),
    )
    _qsa_sparse_paged_decode_kernel.warmup(
        q_ptr,
        TritonWarmupTensor(
            torch.int16,
            shape=tuple(key_pairs.shape),
            strides=tuple(key_pairs.stride()),
        ),
        TritonWarmupTensor(
            torch.int16,
            shape=tuple(value_pairs.shape),
            strides=tuple(value_pairs.stride()),
        ),
        slots_ptr,
        indices_ptr,
        output_ptr,
        1.0,
        1.0,
        output_gate_ptr,
        row_stride,
        head_stride,
        key_pairs.stride(0),
        key_pairs.stride(1),
        key_pairs.stride(2),
        value_pairs.stride(2),
        selection_width + 1,
        slot_columns,
        row_stride,
        head_stride,
        row_stride,
        head_stride,
        TOPK=selection_width,
        OFFSET_BITS=offset_bits,
        GROUP_SIZE=group_size,
        HEAD_DIM=head_dim,
        NUM_TILES=num_tiles,
        BLOCK_M=block_m,
        BLOCK_N=_DECODE_BLOCK_N,
        NUM_STAGES=_DECODE_NUM_STAGES,
        num_warps=_DECODE_NUM_WARPS,
        num_stages=_DECODE_NUM_STAGES,
        grid=(num_rows, num_kv_heads),
    )
    return (("decode", _DECODE_BLOCK_N, _DECODE_NUM_WARPS),)
