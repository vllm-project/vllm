# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Triton sparse-MLA attention kernels for BF16/FP16 latent KV caches.

Supports SM80+ CUDA GPUs (including Ampere A100/A800) with both standard
RoPE MLA (`dim_qk = kv_lora_rank + qk_rope_head_dim`, e.g. 576 = 512 + 64)
and NoPE MLA (`qk_rope_head_dim = 0`, e.g. 512).
"""

import torch

import vllm.envs as envs
from vllm.triton_utils import LOG2E, LOGE2, tl, triton
from vllm.utils.platform_utils import num_compute_units


@triton.jit(do_not_specialize=["num_kv_rows", "index_topk"])
def _triton_mla_sparse_fwd_kernel(
    q_ptr,  # [num_tokens, num_heads, dim_qk]
    kv_ptr,  # [num_kv_rows, dim_qk]
    indices_ptr,  # [num_tokens, index_topk] int32 (-1 padded)
    out_ptr,  # [num_tokens, num_heads, kv_lora_rank]
    lse_ptr,  # [num_tokens, num_heads] fp32 (only written if RETURN_LSE)
    stride_q_tok: tl.int64,
    stride_q_head: tl.int64,
    stride_kv_row: tl.int64,
    stride_idx_tok: tl.int64,
    stride_out_tok: tl.int64,
    stride_out_head: tl.int64,
    stride_lse_tok: tl.int64,
    num_kv_rows,
    index_topk,
    sm_scale_log2,
    NUM_HEADS: tl.constexpr,
    KV_LORA_RANK: tl.constexpr,
    ROPE_DIM: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_N: tl.constexpr,
    RETURN_LSE: tl.constexpr,
    LOGE2: tl.constexpr,
):
    cur_q = tl.program_id(0).to(tl.int64)
    head_block = tl.program_id(1)

    offs_h = head_block * BLOCK_H + tl.arange(0, BLOCK_H)
    mask_h = offs_h < NUM_HEADS
    offs_dv = tl.arange(0, KV_LORA_RANK)

    q_base = q_ptr + cur_q * stride_q_tok + offs_h[:, None].to(tl.int64) * stride_q_head
    q_nope = tl.load(q_base + offs_dv[None, :], mask=mask_h[:, None], other=0.0)

    if ROPE_DIM > 0:
        offs_pe = KV_LORA_RANK + tl.arange(0, ROPE_DIM)
        q_pe = tl.load(q_base + offs_pe[None, :], mask=mask_h[:, None], other=0.0)

    e_max = tl.full([BLOCK_H], float("-inf"), dtype=tl.float32)
    e_sum = tl.zeros([BLOCK_H], dtype=tl.float32)
    acc = tl.zeros([BLOCK_H, KV_LORA_RANK], dtype=tl.float32)

    idx_row_ptr = indices_ptr + cur_q * stride_idx_tok
    offs_n = tl.arange(0, BLOCK_N)

    for start_n in range(0, index_topk, BLOCK_N):
        cur_n = start_n + offs_n
        in_range = cur_n < index_topk
        idx = tl.load(idx_row_ptr + cur_n, mask=in_range, other=-1)
        valid_kv = in_range & (idx >= 0) & (idx < num_kv_rows)

        idx_i64 = idx.to(tl.int64)
        # Load kv_c once [BLOCK_N, KV_LORA_RANK] and reuse for both QK and PV.
        kv_c = tl.load(
            kv_ptr + idx_i64[:, None] * stride_kv_row + offs_dv[None, :],
            mask=valid_kv[:, None],
            other=0.0,
        )
        qk = tl.dot(q_nope, tl.trans(kv_c.to(q_nope.dtype)), out_dtype=tl.float32)

        if ROPE_DIM > 0:
            offs_pe = KV_LORA_RANK + tl.arange(0, ROPE_DIM)
            k_pe = tl.load(
                kv_ptr + idx_i64[:, None] * stride_kv_row + offs_pe[None, :],
                mask=valid_kv[:, None],
                other=0.0,
            )
            qk += tl.dot(q_pe, tl.trans(k_pe.to(q_pe.dtype)), out_dtype=tl.float32)

        qk = qk * sm_scale_log2
        qk = tl.where(valid_kv[None, :], qk, float("-inf"))

        tile_max = tl.max(qk, axis=1)
        n_e_max = tl.maximum(e_max, tile_max)
        # Guard (-inf) - (-inf) = NaN when all keys seen so far are masked.
        re_scale = tl.where(e_max == n_e_max, 1.0, tl.exp2(e_max - n_e_max))
        p = tl.where(valid_kv[None, :], tl.exp2(qk - n_e_max[:, None]), 0.0)

        acc = acc * re_scale[:, None]
        acc += tl.dot(p.to(kv_c.dtype), kv_c, out_dtype=tl.float32)
        e_sum = e_sum * re_scale + tl.sum(p, axis=1)
        e_max = n_e_max

    has_valid = e_sum > 0.0
    safe_sum = tl.where(has_valid, e_sum, 1.0)
    out_vals = tl.where(has_valid[:, None], acc / safe_sum[:, None], 0.0)

    out_base = (
        out_ptr
        + cur_q * stride_out_tok
        + offs_h[:, None].to(tl.int64) * stride_out_head
    )
    tl.store(
        out_base + offs_dv[None, :],
        out_vals.to(out_ptr.dtype.element_ty),
        mask=mask_h[:, None],
    )

    if RETURN_LSE:
        lse = tl.where(has_valid, (e_max + tl.log2(safe_sum)) * LOGE2, float("-inf"))
        tl.store(lse_ptr + cur_q * stride_lse_tok + offs_h, lse, mask=mask_h)


@triton.jit(do_not_specialize=["num_kv_rows", "index_topk", "num_kv_splits"])
def _triton_mla_sparse_splitk_kernel(
    q_ptr,  # [num_tokens, num_heads, dim_qk]
    kv_ptr,  # [num_kv_rows, dim_qk]
    indices_ptr,  # [num_tokens, index_topk] int32
    partial_out_ptr,  # [num_tokens, num_heads, num_kv_splits, kv_lora_rank] fp32
    partial_lse_ptr,  # [num_tokens, num_heads, num_kv_splits] fp32 (log2 domain)
    stride_q_tok: tl.int64,
    stride_q_head: tl.int64,
    stride_kv_row: tl.int64,
    stride_idx_tok: tl.int64,
    stride_po_tok: tl.int64,
    stride_po_head: tl.int64,
    stride_po_split: tl.int64,
    stride_pl_tok: tl.int64,
    stride_pl_head: tl.int64,
    num_kv_rows,
    index_topk,
    num_kv_splits,
    sm_scale_log2,
    NUM_HEADS: tl.constexpr,
    KV_LORA_RANK: tl.constexpr,
    ROPE_DIM: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    cur_q = tl.program_id(0).to(tl.int64)
    head_block = tl.program_id(1)
    split_id = tl.program_id(2)

    offs_h = head_block * BLOCK_H + tl.arange(0, BLOCK_H)
    mask_h = offs_h < NUM_HEADS
    offs_dv = tl.arange(0, KV_LORA_RANK)

    per_split = tl.cdiv(index_topk, num_kv_splits)
    split_start = split_id * per_split
    split_end = tl.minimum(split_start + per_split, index_topk)

    e_max = tl.full([BLOCK_H], float("-inf"), dtype=tl.float32)
    e_sum = tl.zeros([BLOCK_H], dtype=tl.float32)
    acc = tl.zeros([BLOCK_H, KV_LORA_RANK], dtype=tl.float32)

    if split_start < split_end:
        q_base = (
            q_ptr + cur_q * stride_q_tok + offs_h[:, None].to(tl.int64) * stride_q_head
        )
        q_nope = tl.load(q_base + offs_dv[None, :], mask=mask_h[:, None], other=0.0)
        if ROPE_DIM > 0:
            offs_pe = KV_LORA_RANK + tl.arange(0, ROPE_DIM)
            q_pe = tl.load(q_base + offs_pe[None, :], mask=mask_h[:, None], other=0.0)

        idx_row_ptr = indices_ptr + cur_q * stride_idx_tok
        offs_n = tl.arange(0, BLOCK_N)
        for start_n in range(split_start, split_end, BLOCK_N):
            cur_n = start_n + offs_n
            in_range = cur_n < split_end
            idx = tl.load(idx_row_ptr + cur_n, mask=in_range, other=-1)
            valid_kv = in_range & (idx >= 0) & (idx < num_kv_rows)

            idx_i64 = idx.to(tl.int64)
            kv_c = tl.load(
                kv_ptr + idx_i64[:, None] * stride_kv_row + offs_dv[None, :],
                mask=valid_kv[:, None],
                other=0.0,
            )
            qk = tl.dot(q_nope, tl.trans(kv_c.to(q_nope.dtype)), out_dtype=tl.float32)

            if ROPE_DIM > 0:
                offs_pe = KV_LORA_RANK + tl.arange(0, ROPE_DIM)
                k_pe = tl.load(
                    kv_ptr + idx_i64[:, None] * stride_kv_row + offs_pe[None, :],
                    mask=valid_kv[:, None],
                    other=0.0,
                )
                qk += tl.dot(q_pe, tl.trans(k_pe.to(q_pe.dtype)), out_dtype=tl.float32)

            qk = qk * sm_scale_log2
            qk = tl.where(valid_kv[None, :], qk, float("-inf"))

            tile_max = tl.max(qk, axis=1)
            n_e_max = tl.maximum(e_max, tile_max)
            re_scale = tl.where(e_max == n_e_max, 1.0, tl.exp2(e_max - n_e_max))
            p = tl.where(valid_kv[None, :], tl.exp2(qk - n_e_max[:, None]), 0.0)

            acc = acc * re_scale[:, None]
            acc += tl.dot(p.to(kv_c.dtype), kv_c, out_dtype=tl.float32)
            e_sum = e_sum * re_scale + tl.sum(p, axis=1)
            e_max = n_e_max

    has_valid = e_sum > 0.0
    safe_sum = tl.where(has_valid, e_sum, 1.0)
    out_vals = tl.where(has_valid[:, None], acc / safe_sum[:, None], 0.0)
    lse_log2 = tl.where(has_valid, e_max + tl.log2(safe_sum), float("-inf"))

    po_base = (
        partial_out_ptr
        + cur_q * stride_po_tok
        + offs_h[:, None].to(tl.int64) * stride_po_head
        + split_id.to(tl.int64) * stride_po_split
    )
    tl.store(po_base + offs_dv[None, :], out_vals, mask=mask_h[:, None])

    pl_base = (
        partial_lse_ptr
        + cur_q * stride_pl_tok
        + offs_h.to(tl.int64) * stride_pl_head
        + split_id
    )
    tl.store(pl_base, lse_log2, mask=mask_h)


@triton.jit
def _triton_mla_sparse_reduce_kernel(
    partial_out_ptr,  # [num_tokens, num_heads, num_kv_splits, kv_lora_rank] fp32
    partial_lse_ptr,  # [num_tokens, num_heads, num_kv_splits] fp32 (log2 domain)
    out_ptr,  # [num_tokens, num_heads, kv_lora_rank]
    lse_ptr,  # [num_tokens, num_heads] fp32 (natural-log domain)
    stride_po_tok: tl.int64,
    stride_po_head: tl.int64,
    stride_po_split: tl.int64,
    stride_pl_tok: tl.int64,
    stride_pl_head: tl.int64,
    stride_out_tok: tl.int64,
    stride_out_head: tl.int64,
    stride_lse_tok: tl.int64,
    NUM_KV_SPLITS: tl.constexpr,
    KV_LORA_RANK: tl.constexpr,
    RETURN_LSE: tl.constexpr,
    LOGE2: tl.constexpr,
):
    cur_q = tl.program_id(0).to(tl.int64)
    cur_h = tl.program_id(1).to(tl.int64)
    offs_dv = tl.arange(0, KV_LORA_RANK)

    pl_base = partial_lse_ptr + cur_q * stride_pl_tok + cur_h * stride_pl_head
    po_base = partial_out_ptr + cur_q * stride_po_tok + cur_h * stride_po_head

    global_max = float("-inf")
    for s in tl.static_range(NUM_KV_SPLITS):
        lse_s = tl.load(pl_base + s)
        global_max = tl.maximum(global_max, lse_s)

    acc = tl.zeros([KV_LORA_RANK], dtype=tl.float32)
    denom = 0.0
    for s in tl.static_range(NUM_KV_SPLITS):
        lse_s = tl.load(pl_base + s)
        is_valid = lse_s > float("-inf")
        w = tl.where(is_valid, tl.exp2(lse_s - global_max), 0.0)
        part = tl.load(po_base + s * stride_po_split + offs_dv)
        acc += w * part
        denom += w

    has_valid = denom > 0.0
    safe_denom = tl.where(has_valid, denom, 1.0)
    out_vals = tl.where(has_valid, acc / safe_denom, 0.0)
    tl.store(
        out_ptr + cur_q * stride_out_tok + cur_h * stride_out_head + offs_dv,
        out_vals.to(out_ptr.dtype.element_ty),
    )

    if RETURN_LSE:
        lse = tl.where(
            has_valid, (global_max + tl.log2(safe_denom)) * LOGE2, float("-inf")
        )
        tl.store(lse_ptr + cur_q * stride_lse_tok + cur_h, lse)


def _select_num_kv_splits(
    num_tokens: int,
    num_heads: int,
    index_topk: int,
    block_h: int,
    device_index: int | None,
) -> int:
    if envs.VLLM_BATCH_INVARIANT or index_topk <= 256 or num_tokens == 0:
        return 1
    sm_count = max(1, num_compute_units(device_index))
    base_ctas = num_tokens * triton.cdiv(num_heads, block_h)
    max_splits = min(8, triton.cdiv(index_topk, 256))
    if base_ctas >= 2 * sm_count or max_splits <= 1:
        return 1
    ideal = triton.next_power_of_2(triton.cdiv(2 * sm_count, base_ctas))
    return min(ideal, max_splits)


def triton_mla_sparse_fwd(
    q: torch.Tensor,  # [num_tokens, num_heads, dim_qk]
    kv: torch.Tensor,  # [num_kv_rows, dim_qk]
    indices: torch.Tensor,  # [num_tokens, index_topk] int32
    sm_scale: float,
    kv_lora_rank: int = 512,
    return_lse: bool = False,
    num_kv_splits: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Compute sparse MLA attention over a flat BF16/FP16 latent KV cache.

    Args:
        q: `[num_tokens, num_heads, kv_lora_rank + qk_rope_head_dim]`.
        kv: `[num_kv_rows, kv_lora_rank + qk_rope_head_dim]` flat row view.
        indices: `[num_tokens, index_topk]` int32 physical row indices (`-1`
            for masked slots).
        sm_scale: Softmax scale (applied before `exp`).
        kv_lora_rank: Latent dimension (must be a power of 2, typically 512).
        return_lse: If True, also return `[num_tokens, num_heads]` natural-log
            LSE in FP32.
        num_kv_splits: Optional override for split-KV parallelism.

    Returns:
        `(out, lse)` where `out` has shape `[num_tokens, num_heads, kv_lora_rank]`
        and dtype `q.dtype`, and `lse` is `[num_tokens, num_heads]` FP32 or None.

    """
    assert q.ndim == 3 and kv.ndim == 2 and indices.ndim == 2
    num_tokens, num_heads, dim_qk = q.shape
    num_kv_rows, kv_dim = kv.shape
    assert dim_qk == kv_dim, f"q dim {dim_qk} != kv dim {kv_dim}"
    assert indices.shape[0] == num_tokens
    index_topk = indices.shape[1]

    rope_dim = dim_qk - kv_lora_rank
    assert kv_lora_rank > 0 and (kv_lora_rank & (kv_lora_rank - 1)) == 0, (
        f"kv_lora_rank must be a power of 2, got {kv_lora_rank}"
    )
    assert rope_dim == 0 or (rope_dim > 0 and (rope_dim & (rope_dim - 1)) == 0), (
        f"qk_rope_head_dim must be 0 or a power of 2, got {rope_dim}"
    )

    out = torch.empty(
        (num_tokens, num_heads, kv_lora_rank), dtype=q.dtype, device=q.device
    )
    lse = (
        torch.empty((num_tokens, num_heads), dtype=torch.float32, device=q.device)
        if return_lse
        else None
    )
    if num_tokens == 0:
        return out, lse

    q_contig = q if q.stride(-1) == 1 else q.contiguous()
    kv_contig = kv if kv.stride(-1) == 1 else kv.contiguous()
    idx_contig = indices if indices.stride(-1) == 1 else indices.contiguous()

    block_h = 16
    block_n = 32
    sm_scale_log2 = float(sm_scale) * LOG2E
    splits = (
        num_kv_splits
        if num_kv_splits is not None
        else _select_num_kv_splits(
            num_tokens, num_heads, index_topk, block_h, q.device.index
        )
    )
    splits = max(1, splits)
    if splits > 1 and (splits & (splits - 1)) != 0:
        splits = triton.next_power_of_2(splits)

    lse_arg = lse if lse is not None else out
    lse_stride = lse.stride(0) if lse is not None else 0
    num_head_blocks = triton.cdiv(num_heads, block_h)

    if splits == 1:
        _triton_mla_sparse_fwd_kernel[(num_tokens, num_head_blocks)](
            q_contig,
            kv_contig,
            idx_contig,
            out,
            lse_arg,
            q_contig.stride(0),
            q_contig.stride(1),
            kv_contig.stride(0),
            idx_contig.stride(0),
            out.stride(0),
            out.stride(1),
            lse_stride,
            num_kv_rows,
            index_topk,
            sm_scale_log2,
            NUM_HEADS=num_heads,
            KV_LORA_RANK=kv_lora_rank,
            ROPE_DIM=rope_dim,
            BLOCK_H=block_h,
            BLOCK_N=block_n,
            RETURN_LSE=return_lse,
            LOGE2=LOGE2,
            num_warps=4,
            num_stages=1,
        )
        return out, lse

    partial_out = torch.empty(
        (num_tokens, num_heads, splits, kv_lora_rank),
        dtype=torch.float32,
        device=q.device,
    )
    partial_lse = torch.empty(
        (num_tokens, num_heads, splits),
        dtype=torch.float32,
        device=q.device,
    )
    _triton_mla_sparse_splitk_kernel[(num_tokens, num_head_blocks, splits)](
        q_contig,
        kv_contig,
        idx_contig,
        partial_out,
        partial_lse,
        q_contig.stride(0),
        q_contig.stride(1),
        kv_contig.stride(0),
        idx_contig.stride(0),
        partial_out.stride(0),
        partial_out.stride(1),
        partial_out.stride(2),
        partial_lse.stride(0),
        partial_lse.stride(1),
        num_kv_rows,
        index_topk,
        splits,
        sm_scale_log2,
        NUM_HEADS=num_heads,
        KV_LORA_RANK=kv_lora_rank,
        ROPE_DIM=rope_dim,
        BLOCK_H=block_h,
        BLOCK_N=block_n,
        num_warps=4,
        num_stages=1,
    )
    _triton_mla_sparse_reduce_kernel[(num_tokens, num_heads)](
        partial_out,
        partial_lse,
        out,
        lse_arg,
        partial_out.stride(0),
        partial_out.stride(1),
        partial_out.stride(2),
        partial_lse.stride(0),
        partial_lse.stride(1),
        out.stride(0),
        out.stride(1),
        lse_stride,
        NUM_KV_SPLITS=splits,
        KV_LORA_RANK=kv_lora_rank,
        RETURN_LSE=return_lse,
        LOGE2=LOGE2,
        num_warps=4,
    )
    return out, lse
