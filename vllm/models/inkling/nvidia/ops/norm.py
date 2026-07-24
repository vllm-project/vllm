# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

from functools import lru_cache

import torch

from vllm.triton_utils import tl, triton

_MAX_FUSED_SIZE = 65536


def _get_num_warps_from_block_size(block_size: int) -> int:
    if block_size >= 32768:
        return 32
    if block_size >= 8192:
        return 16
    if block_size >= 2048:
        return 8
    return 4


def _largest_power_of_2(n: int) -> int:
    assert n > 0, f"{n=}"
    return 1 << (n.bit_length() - 1)


@lru_cache(maxsize=128)
def _get_grid_size_for_mem_bw_kernel(device: torch.device, factor: int = 8) -> int:
    num_sms = torch.cuda.get_device_properties(device).multi_processor_count
    return _largest_power_of_2(num_sms) * factor


@triton.jit
def _rmsnorm_fwd_kernel(
    x_ptr,
    weight_ptr,
    y_ptr,
    rstd_ptr,
    eps,
    x_stride_0,
    y_stride_0,
    n_cols,
    block_size_n: tl.constexpr,
):
    pid_m = tl.program_id(0).to(tl.int64)
    offs_n = tl.arange(0, block_size_n)
    mask_n = offs_n < n_cols

    weight = tl.load(weight_ptr + offs_n, mask=mask_n, other=0.0).to(tl.float32)
    x = tl.load(x_ptr + pid_m * x_stride_0 + offs_n, mask=mask_n, other=0.0).to(
        tl.float32
    )
    row_var = tl.sum(x * x, axis=0) / n_cols
    rstd = tl.math.rsqrt(row_var + eps)
    tl.store(rstd_ptr + pid_m, rstd)

    y = x * rstd * weight
    tl.store(y_ptr + pid_m * y_stride_0 + offs_n, y, mask=mask_n)


@triton.jit(do_not_specialize=["n_rows"])
def _rmsnorm_fwd_kernel_block_m(
    x_ptr,
    weight_ptr,
    y_ptr,
    rstd_ptr,
    eps,
    x_stride_0,
    y_stride_0,
    n_rows,
    n_cols,
    block_size_m: tl.constexpr,
    block_size_n: tl.constexpr,
):
    pid_m = tl.program_id(0).to(tl.int64)
    offs_n = tl.arange(0, block_size_n)
    mask_n = offs_n < n_cols

    num_blocks_m = tl.cdiv(n_rows, block_size_m)
    blocks_per_pid = tl.cdiv(num_blocks_m, tl.num_programs(0))
    block_id_start = pid_m * blocks_per_pid
    block_id_end = min(block_id_start + blocks_per_pid, num_blocks_m)

    weight = tl.load(weight_ptr + offs_n, mask=mask_n, other=0.0).to(tl.float32)

    for block_id in range(block_id_start, block_id_end):
        offs_m = block_id * block_size_m + tl.arange(0, block_size_m)
        mask_m = offs_m < n_rows
        mask_mn = mask_m[:, None] & mask_n[None, :]
        x = tl.load(
            x_ptr + offs_m[:, None] * x_stride_0 + offs_n[None, :],
            mask=mask_mn,
            other=0.0,
        ).to(tl.float32)
        row_var = tl.sum(x * x, axis=1) / n_cols
        rstd = tl.math.rsqrt(row_var + eps)
        tl.store(rstd_ptr + offs_m, rstd, mask=mask_m)

        y = x * rstd[:, None] * weight
        tl.store(
            y_ptr + offs_m[:, None] * y_stride_0 + offs_n[None, :],
            y,
            mask=mask_mn,
        )


@triton.jit
def _add_rmsnorm_fwd_kernel(
    res_ptr,  # [T, N] residual (read)
    delta_ptr,  # [T, N] delta to add (read)
    weight_ptr,
    y_ptr,  # [T, N] normed output
    res_out_ptr,  # [T, N] updated residual output
    eps,
    res_stride_0,
    delta_stride_0,
    y_stride_0,
    res_out_stride_0,
    n_cols,
    block_size_n: tl.constexpr,
):
    pid_m = tl.program_id(0).to(tl.int64)
    offs_n = tl.arange(0, block_size_n)
    mask_n = offs_n < n_cols

    weight = tl.load(weight_ptr + offs_n, mask=mask_n, other=0.0).to(tl.float32)
    r = tl.load(res_ptr + pid_m * res_stride_0 + offs_n, mask=mask_n, other=0.0).to(
        tl.float32
    )
    d = tl.load(delta_ptr + pid_m * delta_stride_0 + offs_n, mask=mask_n, other=0.0).to(
        tl.float32
    )
    # Round the sum to the residual dtype first (matches the eager
    # `residual + delta` then rmsnorm-on-bf16 sequence bit-for-bit).
    s = (r + d).to(res_out_ptr.dtype.element_ty)
    tl.store(res_out_ptr + pid_m * res_out_stride_0 + offs_n, s, mask=mask_n)
    x = s.to(tl.float32)
    row_var = tl.sum(x * x, axis=0) / n_cols
    rstd = tl.math.rsqrt(row_var + eps)
    y = x * rstd * weight
    tl.store(y_ptr + pid_m * y_stride_0 + offs_n, y, mask=mask_n)


def add_rmsnorm(
    residual: torch.Tensor,
    delta: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused ``res = residual + delta; y = rmsnorm(res)``.

    Returns ``(y, res)``; both are fresh tensors (cudagraph-friendly, no
    in-place update of the inputs).
    """
    assert residual.ndim == 2 and delta.ndim == 2, (residual.shape, delta.shape)
    n_rows, n_cols = residual.shape
    assert weight.shape[0] == n_cols
    y = torch.empty_like(residual)
    res_out = torch.empty_like(residual)
    if n_rows == 0:
        return y, res_out

    block_size_n = triton.next_power_of_2(n_cols)
    max_block_size_n = _MAX_FUSED_SIZE // residual.element_size()
    if max_block_size_n < block_size_n:
        raise RuntimeError(f"Large {n_cols=} is not supported")
    num_warps = _get_num_warps_from_block_size(block_size_n)
    _add_rmsnorm_fwd_kernel[(n_rows,)](
        residual,
        delta,
        weight,
        y,
        res_out,
        eps,
        residual.stride(0),
        delta.stride(0),
        y.stride(0),
        res_out.stride(0),
        n_cols,
        block_size_n,
        num_warps=num_warps,
    )
    return y, res_out


@triton.jit
def _embed_rmsnorm_kernel(
    ids_ptr,  # [T] token ids
    table_ptr,  # [V, N] embedding table
    weight_ptr,  # [N] (HAS_NORM only)
    chain_weight_ptr,  # [N] (HAS_CHAIN only)
    out_ptr,  # [T, N] rmsnorm(table[ids], weight)
    chain_out_ptr,  # [T, N] rmsnorm(out, chain_weight) (HAS_CHAIN only)
    eps,
    table_stride_0,
    n_cols,
    block_size_n: tl.constexpr,
    HAS_NORM: tl.constexpr,
    HAS_CHAIN: tl.constexpr,
):
    pid_m = tl.program_id(0).to(tl.int64)
    offs_n = tl.arange(0, block_size_n)
    mask_n = offs_n < n_cols
    row = tl.load(ids_ptr + pid_m).to(tl.int64)
    x = tl.load(table_ptr + row * table_stride_0 + offs_n, mask=mask_n, other=0.0)
    if HAS_NORM:
        xf = x.to(tl.float32)
        rstd = tl.math.rsqrt(tl.sum(xf * xf, axis=0) / n_cols + eps)
        w = tl.load(weight_ptr + offs_n, mask=mask_n, other=0.0).to(tl.float32)
        # Round to the output dtype so the chained norm is bit-exact vs the
        # unfused pair (which stores bf16 in between).
        x = (xf * rstd * w).to(out_ptr.dtype.element_ty)
    tl.store(out_ptr + pid_m * n_cols + offs_n, x, mask=mask_n)
    if HAS_CHAIN:
        xf = x.to(tl.float32)
        rstd = tl.math.rsqrt(tl.sum(xf * xf, axis=0) / n_cols + eps)
        w = tl.load(chain_weight_ptr + offs_n, mask=mask_n, other=0.0).to(tl.float32)
        tl.store(
            chain_out_ptr + pid_m * n_cols + offs_n,
            (xf * rstd * w).to(chain_out_ptr.dtype.element_ty),
            mask=mask_n,
        )


def embed_rmsnorm(
    input_ids: torch.Tensor,
    embed_table: torch.Tensor,
    weight: torch.Tensor | None,
    eps: float,
    chain_weight: torch.Tensor | None = None,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Fused ``rmsnorm(embed_table[input_ids], weight)`` row gather + norm.

    Requires the full vocab on-rank (replicated or tp_size == 1).
    ``weight=None`` skips the norm (``use_embed_norm=False``), leaving a pure
    embedding-table gather.
    ``chain_weight`` additionally emits ``rmsnorm(out, chain_weight)`` (the
    first decoder layer's pre-attention norm) as a second output, still one
    launch. Bit-exact vs the unfused module sequence."""
    ids = input_ids.view(-1)
    (T,) = ids.shape
    n = embed_table.shape[1]
    out = torch.empty(
        (*input_ids.shape, n), dtype=embed_table.dtype, device=embed_table.device
    )
    chain_out = torch.empty_like(out) if chain_weight is not None else None
    if T > 0:
        block_size_n = triton.next_power_of_2(n)
        _embed_rmsnorm_kernel[(T,)](
            ids,
            embed_table,
            weight if weight is not None else embed_table,
            chain_weight if chain_weight is not None else embed_table,
            out,
            chain_out if chain_out is not None else out,
            eps,
            embed_table.stride(0),
            n,
            block_size_n,
            HAS_NORM=weight is not None,
            HAS_CHAIN=chain_weight is not None,
            num_warps=_get_num_warps_from_block_size(block_size_n),
        )
    if chain_out is not None:
        return out, chain_out
    return out


@triton.jit
def _embed_dual_rmsnorm_cat_kernel(
    hidden_ptr,  # [T, N]
    emb_ptr,  # [T, N] embeddings, or the [V, N] embedding table when GATHER
    ids_ptr,  # [T] token ids (GATHER only)
    w_hidden_ptr,  # [N]
    w_pre_ptr,  # [N] chained pre-norm on the embed side (HAS_PRE_NORM only)
    w_embed_ptr,  # [N]
    out_ptr,  # [T, 2N]: [rmsnorm(hidden) | rmsnorm(rmsnorm?(emb))]
    eps,
    hidden_stride_0,
    emb_stride_0,
    n_cols,
    block_size_n: tl.constexpr,
    GATHER: tl.constexpr,
    HAS_PRE_NORM: tl.constexpr,
):
    pid_m = tl.program_id(0).to(tl.int64)
    which = tl.program_id(1)  # 0 -> hidden into cols [0, N); 1 -> emb into [N, 2N)
    offs_n = tl.arange(0, block_size_n)
    mask_n = offs_n < n_cols
    if which == 0:
        x = tl.load(
            hidden_ptr + pid_m * hidden_stride_0 + offs_n, mask=mask_n, other=0.0
        ).to(tl.float32)
        w = tl.load(w_hidden_ptr + offs_n, mask=mask_n, other=0.0).to(tl.float32)
    else:
        row = tl.load(ids_ptr + pid_m).to(tl.int64) if GATHER else pid_m
        x = tl.load(emb_ptr + row * emb_stride_0 + offs_n, mask=mask_n, other=0.0).to(
            tl.float32
        )
        if HAS_PRE_NORM:
            w_pre = tl.load(w_pre_ptr + offs_n, mask=mask_n, other=0.0).to(tl.float32)
            rstd = tl.math.rsqrt(tl.sum(x * x, axis=0) / n_cols + eps)
            # Round-trip through the output dtype so the chained norm is
            # bit-exact vs the unfused pair (which stores bf16 in between).
            x = (x * rstd * w_pre).to(out_ptr.dtype.element_ty).to(tl.float32)
        w = tl.load(w_embed_ptr + offs_n, mask=mask_n, other=0.0).to(tl.float32)
    rstd = tl.math.rsqrt(tl.sum(x * x, axis=0) / n_cols + eps)
    tl.store(
        out_ptr + pid_m * (2 * n_cols) + which * n_cols + offs_n,
        (x * rstd * w).to(out_ptr.dtype.element_ty),
        mask=mask_n,
    )


def embed_dual_rmsnorm_cat(
    hidden: torch.Tensor,
    hidden_weight: torch.Tensor,
    embed_weight: torch.Tensor,
    eps: float,
    *,
    embeds: torch.Tensor | None = None,
    input_ids: torch.Tensor | None = None,
    embed_table: torch.Tensor | None = None,
    pre_norm_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """The MTP depth-layer input in one launch:
    ``cat([rmsnorm(hidden, w_h), rmsnorm(pre?(emb), w_e)], -1)``.

    The embed side is either a fused row gather ``embed_table[input_ids]``
    (draft decode steps) or precomputed ``embeds`` ([T, N], the target-merged
    multimodal embeddings at draft prefill); ``pre_norm_weight`` chains the
    backbone embed_norm in front of the depth embed_norm (bit-exact vs the
    unfused sequence). The concat copies collapse into direct writes."""
    T, n = hidden.shape
    if embeds is not None:
        assert embeds.shape == hidden.shape
        src, ids, src_stride = embeds, embeds, embeds.stride(0)
        gather = False
    else:
        assert input_ids is not None and embed_table is not None
        assert input_ids.shape == (T,) and embed_table.shape[1] == n
        src, ids, src_stride = embed_table, input_ids, embed_table.stride(0)
        gather = True
    out = torch.empty((T, 2 * n), dtype=hidden.dtype, device=hidden.device)
    if T == 0:
        return out
    block_size_n = triton.next_power_of_2(n)
    _embed_dual_rmsnorm_cat_kernel[(T, 2)](
        hidden,
        src,
        ids,
        hidden_weight,
        pre_norm_weight if pre_norm_weight is not None else embed_weight,
        embed_weight,
        out,
        eps,
        hidden.stride(0),
        src_stride,
        n,
        block_size_n,
        GATHER=gather,
        HAS_PRE_NORM=pre_norm_weight is not None,
        num_warps=_get_num_warps_from_block_size(block_size_n),
    )
    return out


def rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    assert x.ndim == 2, f"{x.shape=}"
    assert weight.ndim == 1, f"{weight.shape=}"
    n_rows, n_cols = x.shape
    assert weight.shape[0] == n_cols, f"{weight.shape=} {x.shape=}"
    y = torch.empty_like(x)
    rstd = torch.empty((n_rows,), dtype=torch.float32, device=x.device)

    block_size_n = triton.next_power_of_2(n_cols)
    max_block_size_n = _MAX_FUSED_SIZE // x.element_size()
    if max_block_size_n < block_size_n:
        raise RuntimeError(f"Large {n_cols=} is not supported")
    block_size_m = max(1, 4096 // block_size_n)
    num_warps = _get_num_warps_from_block_size(block_size_n)

    if block_size_m == 1:
        _rmsnorm_fwd_kernel[(n_rows,)](
            x,
            weight,
            y,
            rstd,
            eps,
            x.stride(0),
            y.stride(0),
            n_cols,
            block_size_n,
            num_warps=num_warps,
        )
    else:
        grid_size = _get_grid_size_for_mem_bw_kernel(x.device)
        _rmsnorm_fwd_kernel_block_m[(grid_size,)](
            x,
            weight,
            y,
            rstd,
            eps,
            x.stride(0),
            y.stride(0),
            n_rows,
            n_cols,
            block_size_m,
            block_size_n,
            num_warps=num_warps,
        )
    return y
