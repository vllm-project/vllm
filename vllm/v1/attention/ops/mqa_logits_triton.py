# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Triton FP8 MQA logits kernels for DeepSeek Sparse Attention (DSA).

Provides CUDA fallback implementations of `fp8_mqa_logits` (prefill) and
`fp8_paged_mqa_logits` (decode) when DeepGEMM is unavailable or unsupported
(such as NVIDIA SM80/SM86 Ampere GPUs).

On compute capability < 8.9 (where Triton cannot emit native `fp8e4nv` PTX),
FP8 E4M3FN bytes are decoded via integer bit manipulation to BF16 and reduced
with BF16 tensor cores (`tl.dot(..., out_dtype=tl.float32)`).
"""

import torch

from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.utils.platform_utils import num_compute_units
from vllm.v1.attention.ops.common import e4m3_uint8_to_f32


def _should_emulate_fp8() -> bool:
    return current_platform.is_cuda() and not current_platform.has_device_capability(89)


def _compute_mqa_splits(num_rows: int, num_tiles: int, device_index: int | None) -> int:
    if num_rows <= 0 or num_tiles <= 0:
        return 1
    sm_count = max(1, num_compute_units(device_index))
    target_ctas = 8 * sm_count
    if num_rows >= target_ctas:
        # Amortize Q decode across multiple key tiles when rows already fill SMs.
        splits = triton.cdiv(num_tiles, 8)
    else:
        splits = triton.cdiv(target_ctas, num_rows)
    return max(1, min(num_tiles, splits, 65535))


@triton.jit(do_not_specialize=["max_seqlen_k", "num_keys", "num_splits"])
def _fp8_mqa_logits_kernel(
    q_ptr,  # [M, H, D] uint8 (EMULATE_FP8) or fp8
    k_ptr,  # [N, D] uint8 (EMULATE_FP8) or fp8
    k_scale_ptr,  # [N] fp32
    weights_ptr,  # [M, H] fp32
    cu_ks_ptr,  # [M] int32
    cu_ke_ptr,  # [M] int32
    out_ptr,  # [M, max_seqlen_k] fp32
    stride_qm: tl.int64,
    stride_qh: tl.int64,
    stride_kn: tl.int64,
    stride_ks: tl.int64,
    stride_wm: tl.int64,
    stride_wh: tl.int64,
    stride_om: tl.int64,
    max_seqlen_k,
    num_keys,
    num_splits,
    NUM_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_N: tl.constexpr,
    EMULATE_FP8: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    split = tl.program_id(1)

    ks = tl.load(cu_ks_ptr + row)
    ke = tl.load(cu_ke_ptr + row)
    row_len = tl.minimum(tl.maximum(ke - ks, 0), max_seqlen_k)
    row_tiles = tl.cdiv(row_len, BLOCK_N)
    if split >= row_tiles:
        return

    offs_h = tl.arange(0, BLOCK_H)
    offs_d = tl.arange(0, BLOCK_D)
    mask_h = offs_h < NUM_HEADS
    mask_d = offs_d < HEAD_DIM

    q_offs = (
        row * stride_qm + offs_h[:, None].to(tl.int64) * stride_qh + offs_d[None, :]
    )
    if EMULATE_FP8:
        q_raw = tl.load(q_ptr + q_offs, mask=mask_h[:, None] & mask_d[None, :], other=0)
        q_bf16 = e4m3_uint8_to_f32(q_raw).to(tl.bfloat16)
    else:
        q_raw = tl.load(
            q_ptr + q_offs, mask=mask_h[:, None] & mask_d[None, :], other=0.0
        )
        q_bf16 = q_raw.to(tl.bfloat16)

    w = tl.load(
        weights_ptr + row * stride_wm + offs_h.to(tl.int64) * stride_wh,
        mask=mask_h,
        other=0.0,
    ).to(tl.float32)

    offs_n = tl.arange(0, BLOCK_N)
    for tile_id in range(split, row_tiles, num_splits):
        col_offs = tile_id * BLOCK_N + offs_n
        valid_col = col_offs < row_len
        k_idx = ks + col_offs
        valid_k = valid_col & (k_idx >= 0) & (k_idx < num_keys)

        k_offs = k_idx[:, None].to(tl.int64) * stride_kn + offs_d[None, :]
        if EMULATE_FP8:
            k_raw = tl.load(
                k_ptr + k_offs, mask=valid_k[:, None] & mask_d[None, :], other=0
            )
            k_bf16 = e4m3_uint8_to_f32(k_raw).to(tl.bfloat16)
        else:
            k_raw = tl.load(
                k_ptr + k_offs, mask=valid_k[:, None] & mask_d[None, :], other=0.0
            )
            k_bf16 = k_raw.to(tl.bfloat16)

        k_scale = tl.load(
            k_scale_ptr + k_idx.to(tl.int64) * stride_ks,
            mask=valid_k,
            other=0.0,
        ).to(tl.float32)

        # [BLOCK_H, BLOCK_D] @ [BLOCK_D, BLOCK_N] -> [BLOCK_H, BLOCK_N]
        dot = tl.dot(q_bf16, tl.trans(k_bf16), out_dtype=tl.float32)
        weighted = tl.sum(tl.maximum(dot, 0.0) * w[:, None], axis=0)
        scores = tl.where(valid_col, weighted * k_scale, float("-inf"))
        tl.store(
            out_ptr + row * stride_om + col_offs,
            scores,
            mask=col_offs < max_seqlen_k,
        )


def fp8_mqa_logits_triton(
    q: torch.Tensor,
    kv: tuple[torch.Tensor, torch.Tensor],
    weights: torch.Tensor,
    cu_seqlen_ks: torch.Tensor,
    cu_seqlen_ke: torch.Tensor,
    max_seqlen_k: int,
) -> torch.Tensor:
    """Compute compressed prefill FP8 MQA logits `[M, max_seqlen_k]` in FP32.

    Matches DeepGEMM's `fp8_fp4_mqa_logits` contract: row `i` holds scores for
    keys `[cu_seqlen_ks[i], cu_seqlen_ke[i])` starting at column 0.

    Args:
        q: `[M, H, D]` FP8 E4M3 queries (Q scales already folded into `weights`).
        kv: `(k_quant, k_scale)` where `k_quant` is `[N, D]` FP8 E4M3 and
            `k_scale` is `[N]` FP32 per-token scales.
        weights: `[M, H]` FP32 head weights.
        cu_seqlen_ks: `[M]` int32 inclusive start indices into `k_quant`.
        cu_seqlen_ke: `[M]` int32 exclusive end indices into `k_quant`.
        max_seqlen_k: Upper bound on `cu_seqlen_ke - cu_seqlen_ks`.

    """
    k_quant, k_scale = kv
    assert q.ndim == 3 and k_quant.ndim == 2
    num_rows, num_heads, head_dim = q.shape
    num_keys, k_head_dim = k_quant.shape
    assert head_dim == k_head_dim, f"q head_dim {head_dim} != k head_dim {k_head_dim}"
    assert k_scale.ndim == 1 and k_scale.shape[0] == num_keys
    assert weights.shape == (num_rows, num_heads)

    out = torch.empty((num_rows, max_seqlen_k), dtype=torch.float32, device=q.device)
    if num_rows == 0 or max_seqlen_k == 0:
        return out

    q_contig = q if q.stride(-1) == 1 else q.contiguous()
    k_contig = k_quant if k_quant.stride(-1) == 1 else k_quant.contiguous()
    emulate_fp8 = _should_emulate_fp8()
    q_arg = q_contig.view(torch.uint8) if emulate_fp8 else q_contig
    k_arg = k_contig.view(torch.uint8) if emulate_fp8 else k_contig

    block_h = max(16, triton.next_power_of_2(num_heads))
    block_d = max(16, triton.next_power_of_2(head_dim))
    block_n = 64
    num_tiles = triton.cdiv(max_seqlen_k, block_n)
    num_splits = _compute_mqa_splits(num_rows, num_tiles, q.device.index)

    _fp8_mqa_logits_kernel[(num_rows, num_splits)](
        q_arg,
        k_arg,
        k_scale,
        weights,
        cu_seqlen_ks,
        cu_seqlen_ke,
        out,
        q_arg.stride(0),
        q_arg.stride(1),
        k_arg.stride(0),
        k_scale.stride(0),
        weights.stride(0),
        weights.stride(1),
        out.stride(0),
        max_seqlen_k,
        num_keys,
        num_splits,
        NUM_HEADS=num_heads,
        HEAD_DIM=head_dim,
        BLOCK_H=block_h,
        BLOCK_D=block_d,
        BLOCK_N=block_n,
        EMULATE_FP8=emulate_fp8,
        num_warps=4,
    )
    return out


@triton.jit(
    do_not_specialize=["max_model_len", "max_blocks", "num_blocks", "num_splits"]
)
def _fp8_paged_mqa_logits_kernel(
    q_ptr,  # [B * next_n, H, D] uint8 (EMULATE_FP8) or fp8
    kv_u8_ptr,  # uint8 view of [num_blocks, block_size, (1,) D + 4]
    weights_ptr,  # [B * next_n, H] fp32
    ctx_lens_ptr,  # [B, next_n] or [B] int32
    block_tables_ptr,  # [B_bt, max_blocks] int32
    indices_ptr,  # optional [B * next_n] int32
    out_ptr,  # [B * next_n, max_model_len] fp32
    stride_qm: tl.int64,
    stride_qh: tl.int64,
    kv_block_stride_bytes: tl.int64,
    stride_wm: tl.int64,
    stride_wh: tl.int64,
    stride_ctx_b: tl.int64,
    stride_ctx_n: tl.int64,
    stride_bt_b: tl.int64,
    stride_bt_k: tl.int64,
    stride_om: tl.int64,
    max_model_len,
    max_blocks,
    num_blocks,
    num_splits,
    BLOCK_SIZE: tl.constexpr,
    NEXT_N: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_N: tl.constexpr,
    CTX_2D: tl.constexpr,
    HAS_INDICES: tl.constexpr,
    EMULATE_FP8: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    split = tl.program_id(1)

    b = row // NEXT_N
    n = row % NEXT_N
    if HAS_INDICES:
        req_idx = tl.load(indices_ptr + row).to(tl.int64)
        if req_idx < 0:
            return
    else:
        req_idx = b

    if CTX_2D:
        seq_len = tl.load(ctx_lens_ptr + b * stride_ctx_b + n * stride_ctx_n)
    else:
        raw_len = tl.load(ctx_lens_ptr + b * stride_ctx_b)
        seq_len = tl.maximum(raw_len - (NEXT_N - 1 - n.to(tl.int32)), 0)

    seq_len = tl.minimum(tl.maximum(seq_len, 0), max_model_len)
    row_tiles = tl.cdiv(seq_len, BLOCK_N)
    if split >= row_tiles:
        return

    offs_h = tl.arange(0, BLOCK_H)
    offs_d = tl.arange(0, BLOCK_D)
    mask_h = offs_h < NUM_HEADS
    mask_d = offs_d < HEAD_DIM

    q_offs = (
        row * stride_qm + offs_h[:, None].to(tl.int64) * stride_qh + offs_d[None, :]
    )
    if EMULATE_FP8:
        q_raw = tl.load(q_ptr + q_offs, mask=mask_h[:, None] & mask_d[None, :], other=0)
        q_bf16 = e4m3_uint8_to_f32(q_raw).to(tl.bfloat16)
    else:
        q_raw = tl.load(
            q_ptr + q_offs, mask=mask_h[:, None] & mask_d[None, :], other=0.0
        )
        q_bf16 = q_raw.to(tl.bfloat16)

    w = tl.load(
        weights_ptr + row * stride_wm + offs_h.to(tl.int64) * stride_wh,
        mask=mask_h,
        other=0.0,
    ).to(tl.float32)

    offs_n = tl.arange(0, BLOCK_N)
    bt_row_ptr = block_tables_ptr + req_idx * stride_bt_b
    scale_section_bytes: tl.constexpr = BLOCK_SIZE * HEAD_DIM

    for tile_id in range(split, row_tiles, num_splits):
        pos = tile_id * BLOCK_N + offs_n
        valid_pos = pos < seq_len
        logical_block = pos // BLOCK_SIZE
        offset_in_block = pos % BLOCK_SIZE
        valid_blk = valid_pos & (logical_block < max_blocks)

        phys_block = tl.load(
            bt_row_ptr + logical_block.to(tl.int64) * stride_bt_k,
            mask=valid_blk,
            other=-1,
        )
        valid_k = valid_blk & (phys_block >= 0) & (phys_block < num_blocks)
        block_base = phys_block.to(tl.int64) * kv_block_stride_bytes

        # Indexer cache page layout:
        #   [0, block_size * head_dim)           : FP8 E4M3 keys
        #   [block_size * head_dim, +block_size*4): FP32 per-token scales
        k_byte_offs = (
            block_base[:, None]
            + offset_in_block[:, None].to(tl.int64) * HEAD_DIM
            + offs_d[None, :]
        )
        if EMULATE_FP8:
            k_raw = tl.load(
                kv_u8_ptr + k_byte_offs,
                mask=valid_k[:, None] & mask_d[None, :],
                other=0,
            )
            k_bf16 = e4m3_uint8_to_f32(k_raw).to(tl.bfloat16)
        else:
            k_fp8_ptr = kv_u8_ptr.to(tl.pointer_type(tl.float8e4nv))
            k_raw = tl.load(
                k_fp8_ptr + k_byte_offs,
                mask=valid_k[:, None] & mask_d[None, :],
                other=0.0,
            )
            k_bf16 = k_raw.to(tl.bfloat16)

        scale_byte_offs = (
            block_base + scale_section_bytes + offset_in_block.to(tl.int64) * 4
        )
        scale_ptrs = (kv_u8_ptr + scale_byte_offs).to(tl.pointer_type(tl.float32))
        k_scale = tl.load(scale_ptrs, mask=valid_k, other=0.0)

        dot = tl.dot(q_bf16, tl.trans(k_bf16), out_dtype=tl.float32)
        weighted = tl.sum(tl.maximum(dot, 0.0) * w[:, None], axis=0)
        scores = tl.where(valid_pos, weighted * k_scale, float("-inf"))
        tl.store(
            out_ptr + row * stride_om + pos,
            scores,
            mask=pos < max_model_len,
        )


def fp8_paged_mqa_logits_triton(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    weights: torch.Tensor,
    context_lens: torch.Tensor,
    block_tables: torch.Tensor,
    max_model_len: int,
    indices: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute paged decode FP8 MQA logits `[B * next_n, max_model_len]` in FP32.

    Matches DeepGEMM's `fp8_fp4_paged_mqa_logits` contract:
    - Handles both 2D `(B, next_n)` (standard on `main`, including `(B, 1)` for
      non-MTP) and 1D `(B,)` `context_lens`.
    - Always sizes output to the static configured `max_model_len` for CUDA
      graph capture safety while skipping tiles past each row's `context_lens`.

    Args:
        q: `[B, next_n, H, D]` FP8 E4M3 queries.
        kv_cache: `[num_blocks, block_size, 1, D + 4]` (or 3D without the
            singleton dim) uint8 indexer KV cache.
        weights: `[B * next_n, H]` FP32 head weights.
        context_lens: `[B, next_n]` or `[B]` int32 per-row/request lengths.
        block_tables: `[B, max_blocks]` int32 page table.
        max_model_len: Configured max model length used to size output columns.
        indices: Optional `[B * next_n]` int32 request index per row.

    """
    assert q.ndim == 4, f"Expected 4D q [B, next_n, H, D], got {q.shape}"
    batch_size, next_n, num_heads, head_dim = q.shape
    num_rows = batch_size * next_n

    if kv_cache.ndim == 4:
        assert kv_cache.shape[2] == 1, (
            f"Expected kv_cache[..., 1, D+4], got {kv_cache.shape}"
        )
        num_blocks, block_size, _, row_bytes = kv_cache.shape
    elif kv_cache.ndim == 3:
        num_blocks, block_size, row_bytes = kv_cache.shape
    else:
        raise ValueError(f"Expected 3D or 4D kv_cache, got shape {kv_cache.shape}")

    assert kv_cache.dtype == torch.uint8
    assert row_bytes >= head_dim + 4, (
        f"kv_cache row width {row_bytes} too small for "
        f"head_dim={head_dim} + 4-byte scale"
    )
    kv_block_stride_bytes = int(kv_cache.stride(0))
    assert kv_block_stride_bytes % 4 == 0 and (block_size * head_dim) % 4 == 0, (
        "Indexer KV cache scale section must be 4-byte aligned"
    )

    assert weights.shape == (num_rows, num_heads)
    assert block_tables.ndim == 2
    max_blocks = block_tables.shape[1]

    if context_lens.ndim == 2:
        assert context_lens.shape == (batch_size, next_n), (
            f"Expected context_lens {(batch_size, next_n)}, "
            f"got {tuple(context_lens.shape)}"
        )
        ctx_2d = True
        stride_ctx_b = context_lens.stride(0)
        stride_ctx_n = context_lens.stride(1)
    elif context_lens.ndim == 1:
        assert context_lens.shape[0] == batch_size, (
            f"Expected context_lens ({batch_size},), got {tuple(context_lens.shape)}"
        )
        ctx_2d = False
        stride_ctx_b = context_lens.stride(0)
        stride_ctx_n = 0
    else:
        raise ValueError(f"Expected 1D or 2D context_lens, got {context_lens.ndim}D")

    out = torch.empty((num_rows, max_model_len), dtype=torch.float32, device=q.device)
    if num_rows == 0 or max_model_len == 0:
        return out

    q_flat = q.reshape(num_rows, num_heads, head_dim)
    if q_flat.stride(-1) != 1:
        q_flat = q_flat.contiguous()

    emulate_fp8 = _should_emulate_fp8()
    q_arg = q_flat.view(torch.uint8) if emulate_fp8 else q_flat

    block_h = max(16, triton.next_power_of_2(num_heads))
    block_d = max(16, triton.next_power_of_2(head_dim))
    block_n = 64
    num_tiles = triton.cdiv(max_model_len, block_n)
    num_splits = _compute_mqa_splits(num_rows, num_tiles, q.device.index)

    _fp8_paged_mqa_logits_kernel[(num_rows, num_splits)](
        q_arg,
        kv_cache,
        weights,
        context_lens,
        block_tables,
        indices if indices is not None else context_lens,
        out,
        q_arg.stride(0),
        q_arg.stride(1),
        kv_block_stride_bytes,
        weights.stride(0),
        weights.stride(1),
        stride_ctx_b,
        stride_ctx_n,
        block_tables.stride(0),
        block_tables.stride(1),
        out.stride(0),
        max_model_len,
        max_blocks,
        num_blocks,
        num_splits,
        BLOCK_SIZE=block_size,
        NEXT_N=next_n,
        NUM_HEADS=num_heads,
        HEAD_DIM=head_dim,
        BLOCK_H=block_h,
        BLOCK_D=block_d,
        BLOCK_N=block_n,
        CTX_2D=ctx_2d,
        HAS_INDICES=indices is not None,
        EMULATE_FP8=emulate_fp8,
        num_warps=4,
    )
    return out
