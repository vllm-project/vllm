# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Blackwell (sm_100/sm_103) fp8 path for QSA sparse attention.

A slot pass translates each row's logical tokens through the page table once
for all KV heads (or fuses the lookup directly at tiny batches to avoid the
extra launch), then an int16-pair kernel attends over the selected tokens with
split-K and gated PDL overlap. Gathering K/V as int16 pairs of e4m3
codes feeds the MMAs without byte-wise shared-memory loads. qsa.py's
qsa_sparse_paged_attention dispatches here via use_fp8_decode_kernel; every
QK/PV product matches the split-K kernel's (QK in bf16, PV in fp16, both
exact) -- only fp32 summation grouping differs.
"""

import functools

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


@triton.jit(do_not_specialize=["num_requests", "num_cache_blocks"])
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
    launch_pdl: tl.constexpr,
) -> None:
    """Translate a row's logical tokens through the page table once, into
    page << OFFSET_BITS | page_offset slots shared by every KV head. An entry
    the split-K kernel masks, and the padding past TOPK, becomes -1."""
    row = tl.program_id(0)
    columns = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    if launch_pdl:
        tl.extra.cuda.gdc_wait()
    request = tl.load(token_to_req_ptr + row)
    valid_count = tl.load(indices_ptr + row * stride_indices_row + TOPK)
    safe_request = tl.minimum(tl.maximum(request, 0), num_requests - 1)
    logical_token = tl.load(
        indices_ptr + row * stride_indices_row + columns,
        mask=columns < tl.minimum(valid_count, TOPK),
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
    if launch_pdl:
        tl.extra.cuda.gdc_launch_dependents()


@functools.cache
def _ptxas_has_bf16x2_e4m3() -> bool:
    # cvt.rn.bf16x2.e4m3x2 needs PTX ISA 9.2 (ptxas >= 13.2); checked lazily
    # after imports (e.g. flashinfer.triton may override TRITON_PTXAS_BLACKWELL_PATH
    # with an older system ptxas).
    if not current_platform.is_cuda():
        return False
    from triton.backends.nvidia.compiler import get_ptxas

    major, minor = map(int, get_ptxas(100).version.split(".")[:2])
    return (major, minor) >= (13, 2)


@triton.jit
def _e4m3_pairs_to_bf16(pairs, HIGH: tl.constexpr, USE_BF16_CVT: tl.constexpr):
    """bf16 of one byte of each int16 pair of e4m3 codes (byte 0 = dim 2i,
    byte 1 = dim 2i + 1); exact via PTX 9.2 cvt.rn.bf16x2.e4m3x2 (sm_100f+),
    with fp16->bf16 fallback on older ptxas."""
    if not USE_BF16_CVT:
        return _e4m3_pairs_to_fp16(pairs, HIGH).to(tl.bfloat16)
    if HIGH:
        return tl.inline_asm_elementwise(
            asm="""
            {
            .reg .b32 t;
            .reg .b16 lo, hi;
            prmt.b32 t, $1, 0, 0x3120;
            mov.b32 {lo, hi}, t;
            cvt.rn.bf16x2.e4m3x2 $0, hi;
            }
            """,
            constraints="=r,r",
            args=[pairs],
            dtype=tl.bfloat16,
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
            cvt.rn.bf16x2.e4m3x2 $0, lo;
            }
            """,
            constraints="=r,r",
            args=[pairs],
            dtype=tl.bfloat16,
            is_pure=True,
            pack=2,
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


@triton.jit(do_not_specialize=["num_rows", "num_requests", "num_cache_blocks"])
def _qsa_sparse_paged_decode_kernel(
    q_ptr,
    k_pairs_ptr,
    v_pairs_ptr,
    slots_ptr,
    indices_ptr,
    block_table_ptr,
    token_to_req_ptr,
    partial_output_ptr,
    partial_lse_ptr,
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
    stride_table_req,
    stride_slots_row,
    stride_output_row,
    stride_output_head,
    stride_output_gate_row,
    stride_output_gate_head,
    num_rows,
    num_cache_blocks,
    num_requests,
    TOPK: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    PAGE_TABLE_WIDTH: tl.constexpr,
    OFFSET_BITS: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    NUM_QUERY_HEADS: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    NUM_TILES: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    NUM_STAGES: tl.constexpr,
    USE_BF16_CVT: tl.constexpr,
    launch_pdl: tl.constexpr,
) -> None:
    """Single-pass and split-K Blackwell fp8 attention kernel.

    K and V are gathered as int16 pairs of e4m3 codes, which feed the MMA
    operands without byte-wise shared-memory loads, and split exactly into
    even/odd-dim halves: scores = Q_even K_even + Q_odd K_odd in bf16 and
    O = [P V_even, P V_odd] in fp16.

    When slots_ptr is None (tiny batches), page-table lookups are fused into
    the loop to avoid the slot-pass launch overhead. Otherwise slots_ptr holds
    precomputed page << OFFSET_BITS | page_offset entries.
    """
    HALF: tl.constexpr = HEAD_DIM // 2
    row = tl.program_id(0)
    kv_head = tl.program_id(1)
    split_id = tl.program_id(2)

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
    if slots_ptr is None:
        if launch_pdl:
            tl.extra.cuda.gdc_wait()
        query_even = tl.load(q_pairs, mask=head_mask, other=0.0)
        query_odd = tl.load(q_pairs + 1, mask=head_mask, other=0.0)
        valid_count = tl.load(indices_ptr + row * stride_indices_row + TOPK)
        request = tl.load(token_to_req_ptr + row)
        safe_request = tl.minimum(tl.maximum(request, 0), num_requests - 1)
    else:
        # Load Q and valid_count before gdc_wait so they overlap the slot pass
        # tail (the slot pass already waited on earlier producers).
        query_even = tl.load(q_pairs, mask=head_mask, other=0.0)
        query_odd = tl.load(q_pairs + 1, mask=head_mask, other=0.0)
        valid_count = tl.load(indices_ptr + row * stride_indices_row + TOPK)
        if launch_pdl:
            tl.extra.cuda.gdc_wait()
    score_scale = softmax_scale * 1.4426950408889634

    max_value = tl.full((BLOCK_M,), -1.0e20, dtype=tl.float32)
    normalizer = tl.zeros((BLOCK_M,), dtype=tl.float32)
    acc_even = tl.zeros((BLOCK_M, HALF), dtype=tl.float32)
    acc_odd = tl.zeros((BLOCK_M, HALF), dtype=tl.float32)

    tile_end = tl.minimum(NUM_TILES, tl.cdiv(tl.minimum(valid_count, TOPK), BLOCK_N))
    k_head_ptr = k_pairs_ptr + kv_head * stride_k_head
    v_head_ptr = v_pairs_ptr + kv_head * stride_v_head
    for tile in tl.range(split_id, tile_end, NUM_SPLITS, num_stages=NUM_STAGES):
        if slots_ptr is None:
            cols = tile * BLOCK_N + column_offsets
            logical_token = tl.load(
                indices_ptr + row * stride_indices_row + cols,
                mask=cols < tl.minimum(valid_count, TOPK),
                other=-1,
            )
            safe_token = tl.maximum(logical_token, 0)
            logical_page = safe_token // PAGE_SIZE
            page_off = safe_token % PAGE_SIZE
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
            page = tl.maximum(physical_page, 0).to(tl.int64)
            page_offset = page_off.to(tl.int64)
        else:
            slot = tl.load(
                slots_ptr + row * stride_slots_row + tile * BLOCK_N + column_offsets
            )
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
            query_even, _e4m3_pairs_to_bf16(key_pairs, False, USE_BF16_CVT)
        ) + tl.dot(query_odd, _e4m3_pairs_to_bf16(key_pairs, True, USE_BF16_CVT))
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

    has_values = normalizer > 0
    inv_normalizer = output_scale / tl.maximum(normalizer, 1.0e-20)
    output_even = tl.where(has_values[:, None], acc_even * inv_normalizer[:, None], 0.0)
    output_odd = tl.where(has_values[:, None], acc_odd * inv_normalizer[:, None], 0.0)
    if NUM_SPLITS == 1:
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
    else:
        partial_lse = tl.where(
            has_values,
            max_value + tl.math.log2(tl.maximum(normalizer, 1.0e-20)),
            -float("inf"),
        )
        partial_row = split_id.to(tl.int64) * num_rows + row
        out_fp32 = tl.reshape(tl.join(output_even, output_odd), (BLOCK_M, HEAD_DIM))
        tl.store(
            partial_output_ptr
            + (partial_row * NUM_QUERY_HEADS + first_head + head_offsets[:, None])
            * HEAD_DIM
            + dim_offsets[None, :],
            out_fp32,
            mask=head_mask,
        )
        tl.store(
            partial_lse_ptr + partial_row * NUM_QUERY_HEADS + first_head + head_offsets,
            partial_lse,
            mask=head_offsets < GROUP_SIZE,
        )
        if launch_pdl:
            tl.extra.cuda.gdc_launch_dependents()


def use_fp8_decode_kernel(base_programs: int, cache_dtype: torch.dtype) -> bool:
    """All Blackwell fp8 logical-index batches take qsa_sparse_paged_decode."""
    return (
        base_programs > 0 and cache_dtype == torch.float8_e4m3fn and _is_sm100_family()
    )


_SLOTS_BLOCK = 1024
_DECODE_NUM_WARPS = 4

# (max_base_programs, block_n, target_splits, num_stages, fused_slots)
# Tuned on GB300 (148 SMs, 3 CTAs/SM): tiny batches (bp <= 16) fuse page-table
# lookups to skip the slot-pass launch; mid batches split-K to fill SMs and
# avoid wave-tail cliffs (e.g. bp=512 -> 2 splits).
_DECODE_CONFIG_TABLE: tuple[tuple[int, int, int, int, bool], ...] = (
    (8, 32, 64, 3, True),
    (16, 64, 16, 5, True),
    (32, 64, 8, 5, False),
    (64, 64, 4, 5, False),
    (160, 64, 2, 5, False),
    (320, 64, 1, 5, False),
    (640, 64, 2, 5, False),
)
_DECODE_DEFAULT_CONFIG = (64, 1, 5, False)


def _select_decode_config(
    base_programs: int, selection_width: int
) -> tuple[int, int, int, int, int, bool, bool]:
    """Select (block_n, num_tiles, num_splits, num_warps, num_stages,
    fused_slots, use_pdl) for Blackwell fp8 sparse attention."""
    block_n, target_splits, num_stages, fused_slots = _DECODE_DEFAULT_CONFIG
    for max_bp, bn, splits, stages, fused in _DECODE_CONFIG_TABLE:
        if base_programs <= max_bp:
            block_n, target_splits, num_stages, fused_slots = bn, splits, stages, fused
            break
    num_tiles = triton.cdiv(selection_width, block_n)
    num_splits = min(target_splits, num_tiles)
    # Gate PDL on split-K: overlaps slots -> decode -> merge on small/mid grids,
    # while avoiding SM/L2 contention on multi-wave single-pass grids.
    use_pdl = num_splits > 1 and current_platform.is_arch_support_pdl()
    return (
        block_n,
        num_tiles,
        num_splits,
        _DECODE_NUM_WARPS,
        num_stages,
        fused_slots,
        use_pdl,
    )


def _decode_slot_layout(
    num_cache_blocks: int, page_size: int, num_tiles: int, block_n: int
) -> tuple[int, int]:
    """(offset_bits, slot_columns) of the decode kernel's slots."""
    offset_bits = max(1, (page_size - 1).bit_length())
    assert num_cache_blocks << offset_bits < 2**31
    return offset_bits, num_tiles * block_n


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
    """Sparse GQA over an fp8-e4m3 cache on Blackwell (single-pass & split-K)."""
    # e4m3 code pairs; the cache views' strides and V's offset are even.
    k_pairs = k_cache.view(torch.int16)
    v_pairs = v_cache.view(torch.int16)
    # Cache rows start on 16 bytes, which the gathers' vector width relies on.
    assert k_pairs.stride(0) % 8 == 0 and k_pairs.stride(1) % 8 == 0
    assert v_pairs.stride()[:2] == k_pairs.stride()[:2]
    num_rows, num_query_heads, head_dim = q.shape
    assert head_dim >= 32, f"Expected head_dim >= 32, got {head_dim}"
    num_cache_blocks, page_size, num_kv_heads = k_cache.shape[:3]
    selection_width = logical_indices.shape[1] - 1
    base_programs = num_rows * num_kv_heads
    (
        block_n,
        num_tiles,
        num_splits,
        num_warps,
        num_stages,
        fused_slots,
        use_pdl,
    ) = _select_decode_config(base_programs, selection_width)
    offset_bits, slot_columns = _decode_slot_layout(
        num_cache_blocks, page_size, num_tiles, block_n
    )
    if fused_slots:
        slots = None
        table_arg = block_table
        req_arg = token_to_req
        stride_slots_row = 0
    else:
        slots = torch.empty(
            (num_rows, slot_columns), dtype=torch.int32, device=q.device
        )
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
            launch_pdl=use_pdl,
        )
        table_arg = None
        req_arg = None
        stride_slots_row = slots.stride(0)

    if num_splits == 1:
        partial_output = None
        partial_lse = None
    else:
        partial_output = torch.empty(
            (num_splits, num_rows, num_query_heads, head_dim),
            dtype=torch.float32,
            device=q.device,
        )
        partial_lse = torch.empty(
            (num_splits, num_rows, num_query_heads),
            dtype=torch.float32,
            device=q.device,
        )

    group_size = num_query_heads // num_kv_heads
    _qsa_sparse_paged_decode_kernel[(num_rows, num_kv_heads, num_splits)](
        q,
        k_pairs,
        v_pairs,
        slots,
        logical_indices,
        table_arg,
        req_arg,
        partial_output,
        partial_lse,
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
        block_table.stride(0),
        stride_slots_row,
        out.stride(0),
        out.stride(1),
        output_gate.stride(0),
        output_gate.stride(1),
        num_rows,
        num_cache_blocks,
        block_table.shape[0],
        TOPK=selection_width,
        PAGE_SIZE=page_size,
        PAGE_TABLE_WIDTH=block_table.shape[1],
        OFFSET_BITS=offset_bits,
        GROUP_SIZE=group_size,
        HEAD_DIM=head_dim,
        NUM_QUERY_HEADS=num_query_heads,
        NUM_SPLITS=num_splits,
        NUM_TILES=num_tiles,
        BLOCK_M=triton.next_power_of_2(group_size),
        BLOCK_N=block_n,
        NUM_STAGES=num_stages,
        USE_BF16_CVT=_ptxas_has_bf16x2_e4m3(),
        num_warps=num_warps,
        num_stages=num_stages,
        launch_pdl=use_pdl,
    )
    if num_splits > 1:
        from .qsa import _qsa_merge_splitk_kernel

        _qsa_merge_splitk_kernel[(num_rows, num_query_heads)](
            partial_output,
            partial_lse,
            out,
            output_gate,
            out.stride(0),
            out.stride(1),
            output_gate.stride(0),
            output_gate.stride(1),
            num_rows,
            HEAD_DIM=head_dim,
            NUM_QUERY_HEADS=num_query_heads,
            NUM_SPLITS=num_splits,
            BLOCK_SPLITS=triton.next_power_of_2(num_splits),
            num_warps=2,
            num_stages=1,
            launch_pdl=use_pdl,
        )
    return out


def warmup_qsa_fp8_decode(
    kv_cache: torch.Tensor,
    block_table: torch.Tensor,
    *,
    num_query_heads: int,
    selection_width: int,
) -> tuple[tuple[str, int, int, int], ...]:
    """Compile all reachable slot-pass, decode, and merge specializations."""
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
    if not use_fp8_decode_kernel(1, cache_dtype):
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
    key_pairs_ptr = TritonWarmupTensor(
        torch.int16,
        shape=tuple(key_pairs.shape),
        strides=tuple(key_pairs.stride()),
    )
    value_pairs_ptr = TritonWarmupTensor(
        torch.int16,
        shape=tuple(value_pairs.shape),
        strides=tuple(value_pairs.stride()),
    )

    profiles = sorted(
        {
            _select_decode_config(rows * num_kv_heads, selection_width)
            for rows in range(1, 8193)
        }
    )

    warmed_slots: set[tuple[int, bool]] = set()
    warmed_merges: set[tuple[int, bool]] = set()
    warmed: list[tuple[str, int, int, int]] = []
    for (
        block_n,
        num_tiles,
        num_splits,
        num_warps,
        num_stages,
        fused_slots,
        use_pdl,
    ) in profiles:
        offset_bits, slot_columns = _decode_slot_layout(
            kv_cache.shape[0], key_cache.shape[1], num_tiles, block_n
        )
        if fused_slots:
            slots_ptr = None
            table_arg = block_table_ptr
            req_arg = token_to_req_ptr
            stride_slots_row = 0
        else:
            slots_ptr = TritonWarmupTensor(torch.int32, shape=(num_rows, slot_columns))
            table_arg = None
            req_arg = None
            stride_slots_row = slot_columns
            if (slot_columns, use_pdl) not in warmed_slots:
                warmed_slots.add((slot_columns, use_pdl))
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
                    launch_pdl=use_pdl,
                    grid=(num_rows, triton.cdiv(slot_columns, _SLOTS_BLOCK)),
                )

        if num_splits == 1:
            partial_output_ptr = None
            partial_lse_ptr = None
        else:
            partial_output_ptr = TritonWarmupTensor(
                torch.float32,
                shape=(num_splits, num_rows, num_query_heads, head_dim),
            )
            partial_lse_ptr = TritonWarmupTensor(
                torch.float32, shape=(num_splits, num_rows, num_query_heads)
            )

        _qsa_sparse_paged_decode_kernel.warmup(
            q_ptr,
            key_pairs_ptr,
            value_pairs_ptr,
            slots_ptr,
            indices_ptr,
            table_arg,
            req_arg,
            partial_output_ptr,
            partial_lse_ptr,
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
            block_table.stride(0),
            stride_slots_row,
            row_stride,
            head_stride,
            row_stride,
            head_stride,
            num_rows,
            num_cache_blocks,
            num_requests,
            TOPK=selection_width,
            PAGE_SIZE=key_cache.shape[1],
            PAGE_TABLE_WIDTH=block_table.shape[1],
            OFFSET_BITS=offset_bits,
            GROUP_SIZE=group_size,
            HEAD_DIM=head_dim,
            NUM_QUERY_HEADS=num_query_heads,
            NUM_SPLITS=num_splits,
            NUM_TILES=num_tiles,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            NUM_STAGES=num_stages,
            USE_BF16_CVT=_ptxas_has_bf16x2_e4m3(),
            num_warps=num_warps,
            num_stages=num_stages,
            launch_pdl=use_pdl,
            grid=(num_rows, num_kv_heads, num_splits),
        )
        if num_splits > 1 and (num_splits, use_pdl) not in warmed_merges:
            from .qsa import _qsa_merge_splitk_kernel

            warmed_merges.add((num_splits, use_pdl))
            _qsa_merge_splitk_kernel.warmup(
                partial_output_ptr,
                partial_lse_ptr,
                output_ptr,
                output_gate_ptr,
                row_stride,
                head_stride,
                row_stride,
                head_stride,
                num_rows,
                HEAD_DIM=head_dim,
                NUM_QUERY_HEADS=num_query_heads,
                NUM_SPLITS=num_splits,
                BLOCK_SPLITS=triton.next_power_of_2(num_splits),
                num_warps=2,
                num_stages=1,
                launch_pdl=use_pdl,
                grid=(num_rows, num_query_heads),
            )
        warmed.append(("decode", block_n, num_splits, num_warps))
    return tuple(warmed)
