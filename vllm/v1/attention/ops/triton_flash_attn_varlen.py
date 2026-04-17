# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Triton implementation of dense varlen (cu_seqlens) scaled dot-product attention
# for MLA prefill, following the numerics and masking patterns of
# `triton_prefill_attention.context_attention_fwd` and the online softmax
# structure used in `triton_unified_attention`.

from __future__ import annotations

import dataclasses
import functools
import math

import torch

from vllm.triton_utils import tl, triton
from vllm.utils.math_utils import RCP_LN2


def _assert_varlen_tensors_same_accelerator(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
) -> None:
    """Require Q/K/V and cu_seqlens on the same CUDA or XPU device."""
    dev = q.device
    if dev.type not in ("cuda", "xpu"):
        raise NotImplementedError(
            "triton_flash_attn_varlen expects tensors on CUDA or XPU; "
            f"got device type {dev.type!r}"
        )
    for name, t in (
        ("k", k),
        ("v", v),
        ("cu_seqlens_q", cu_seqlens_q),
        ("cu_seqlens_k", cu_seqlens_k),
    ):
        if t.device != dev:
            raise ValueError(
                f"{name} must be on the same device as q ({dev}), got {t.device}"
            )


@triton.jit
def _fwd_kernel_varlen(
    Q,
    K,
    V,
    sm_scale,
    cu_seqlens_q_ptr,
    cu_seqlens_k_ptr,
    Out,
    lse_ptr,
    stride_qbs,
    stride_qh,
    stride_kbs,
    stride_kh,
    stride_vbs,
    stride_vh,
    stride_obs,
    stride_oh,
    stride_lse_h,
    stride_lse_t,
    kv_group_num: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_DQK: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    BLOCK_N: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    HAS_LSE: tl.constexpr,
    D_QK: tl.constexpr,
    D_V: tl.constexpr,
    EVEN_DQK: tl.constexpr,
    EVEN_DV: tl.constexpr,
):
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)
    start_m = tl.program_id(2)

    cur_kv_head = cur_head // kv_group_num

    q_start = tl.load(cu_seqlens_q_ptr + cur_batch)
    q_end = tl.load(cu_seqlens_q_ptr + cur_batch + 1)
    sq = q_end - q_start

    k_start = tl.load(cu_seqlens_k_ptr + cur_batch)
    k_end = tl.load(cu_seqlens_k_ptr + cur_batch + 1)
    sk = k_end - k_start

    block_start_loc = BLOCK_M * start_m

    # Early exit for out-of-bounds query blocks.
    if block_start_loc >= sq:
        return

    offs_n = tl.arange(0, BLOCK_N)
    offs_d_qk = tl.arange(0, BLOCK_DQK)
    offs_d_v = tl.arange(0, BLOCK_DV)
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)

    off_q = (
        (q_start + offs_m[:, None]) * stride_qbs
        + cur_head * stride_qh
        + offs_d_qk[None, :]
    )
    if EVEN_DQK:
        q = tl.load(Q + off_q, mask=(offs_m[:, None] < sq), other=0.0)
    else:
        q = tl.load(
            Q + off_q,
            mask=(offs_m[:, None] < sq) & (offs_d_qk[None, :] < D_QK),
            other=0.0,
        )

    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DV], dtype=tl.float32)

    # Compute iteration bounds.
    end_n = sk
    if IS_CAUSAL:
        end_n = tl.minimum(sk, (start_m + 1) * BLOCK_M)

    # Split the inner loop into two phases:
    #   Phase 1 (unmasked): all keys in the block are valid — skip mask computation
    #       and use unconditional loads for better pipelining.
    #   Phase 2 (masked): boundary blocks that need causal / bounds masking.
    if IS_CAUSAL:
        # Keys with index < start_m * BLOCK_M are fully below the diagonal
        # for every query in [start_m * BLOCK_M, (start_m+1) * BLOCK_M).
        unmasked_end = tl.minimum((start_m * BLOCK_M // BLOCK_N) * BLOCK_N, end_n)
    else:
        # Only the last (partial) block may need a bounds mask.
        unmasked_end = (end_n // BLOCK_N) * BLOCK_N

    # ── Phase 1: unmasked inner loop ─────────────────────────────────── #
    for start_n in range(0, unmasked_end, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)

        k_offs = (
            (k_start + start_n + offs_n[None, :]) * stride_kbs
            + cur_kv_head * stride_kh
            + offs_d_qk[:, None]
        )
        if EVEN_DQK:
            k = tl.load(K + k_offs)
        else:
            k = tl.load(K + k_offs, mask=(offs_d_qk[:, None] < D_QK), other=0.0)

        qk = tl.dot(q, k)
        qk = qk * sm_scale
        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        qk -= m_ij[:, None]
        p = tl.math.exp2(qk)
        l_ij = tl.sum(p, 1)

        alpha = tl.math.exp2(m_i - m_ij)
        l_i = l_i * alpha + l_ij
        acc = acc * alpha[:, None]

        v_offs = (
            (k_start + start_n + offs_n[:, None]) * stride_vbs
            + cur_kv_head * stride_vh
            + offs_d_v[None, :]
        )
        if EVEN_DV:
            v = tl.load(V + v_offs)
        else:
            v = tl.load(V + v_offs, mask=(offs_d_v[None, :] < D_V), other=0.0)
        p = p.to(v.dtype)
        acc = tl.dot(p, v, acc)
        m_i = m_ij

    # ── Phase 2: masked inner loop ───────────────────────────────────── #
    for start_n in range(unmasked_end, end_n, BLOCK_N):
        pos_q = offs_m[:, None]
        pos_k = start_n + offs_n[None, :]

        mask = pos_k < sk
        if IS_CAUSAL:
            mask = mask & (pos_q >= pos_k)

        start_n = tl.multiple_of(start_n, BLOCK_N)

        k_offs = (
            (k_start + start_n + offs_n[None, :]) * stride_kbs
            + cur_kv_head * stride_kh
            + offs_d_qk[:, None]
        )
        if EVEN_DQK:
            k = tl.load(K + k_offs, mask=(pos_k < sk), other=0.0)
        else:
            k = tl.load(
                K + k_offs,
                mask=(pos_k < sk) & (offs_d_qk[:, None] < D_QK),
                other=0.0,
            )

        qk = tl.dot(q, k)
        qk = tl.where(mask, qk * sm_scale, -1.0e8)
        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        qk -= m_ij[:, None]
        p = tl.math.exp2(qk)
        l_ij = tl.sum(p, 1)

        alpha = tl.math.exp2(m_i - m_ij)
        l_i = l_i * alpha + l_ij
        acc = acc * alpha[:, None]

        v_offs = (
            (k_start + start_n + offs_n[:, None]) * stride_vbs
            + cur_kv_head * stride_vh
            + offs_d_v[None, :]
        )
        if EVEN_DV:
            v = tl.load(V + v_offs, mask=((start_n + offs_n[:, None]) < sk), other=0.0)
        else:
            v = tl.load(
                V + v_offs,
                mask=((start_n + offs_n[:, None]) < sk) & (offs_d_v[None, :] < D_V),
                other=0.0,
            )
        p = p.to(v.dtype)
        acc = tl.dot(p, v, acc)
        m_i = m_ij

    # ── Epilogue ─────────────────────────────────────────────────────── #
    acc = acc / l_i[:, None]

    off_o = (
        (q_start + offs_m[:, None]) * stride_obs
        + cur_head * stride_oh
        + offs_d_v[None, :]
    )
    if EVEN_DV:
        tl.store(Out + off_o, acc, mask=(offs_m[:, None] < sq))
    else:
        tl.store(
            Out + off_o,
            acc,
            mask=(offs_m[:, None] < sq) & (offs_d_v[None, :] < D_V),
        )

    if HAS_LSE:
        # Natural-log LSE: max(L) + log(sum exp(L - max)); m_i is max of L / ln2
        lse_row = m_i * 0.6931471805599453 + tl.log(l_i)
        lse_off = cur_head * stride_lse_h + (q_start + offs_m) * stride_lse_t
        tl.store(lse_ptr + lse_off, lse_row, mask=offs_m < sq)


@triton.jit
def _fwd_kernel_varlen_block_ptr(
    Q,
    K,
    V,
    sm_scale,
    cu_seqlens_q_ptr,
    cu_seqlens_k_ptr,
    Out,
    lse_ptr,
    stride_qbs,
    stride_qh,
    stride_kbs,
    stride_kh,
    stride_vbs,
    stride_vh,
    stride_obs,
    stride_oh,
    stride_lse_h,
    stride_lse_t,
    kv_group_num: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_DQK: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    BLOCK_N: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    HAS_LSE: tl.constexpr,
    D_QK: tl.constexpr,
    D_V: tl.constexpr,
    EVEN_DQK: tl.constexpr,
    EVEN_DV: tl.constexpr,
):
    """Block-pointer optimized version of _fwd_kernel_varlen.

    Uses tl.make_block_ptr for improved memory coalescing and reduced
    pointer arithmetic overhead, especially beneficial on XPU.
    """
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)
    start_m = tl.program_id(2)

    cur_kv_head = cur_head // kv_group_num

    q_start = tl.load(cu_seqlens_q_ptr + cur_batch)
    q_end = tl.load(cu_seqlens_q_ptr + cur_batch + 1)
    sq = q_end - q_start

    k_start = tl.load(cu_seqlens_k_ptr + cur_batch)
    k_end = tl.load(cu_seqlens_k_ptr + cur_batch + 1)
    sk = k_end - k_start

    block_start_loc = BLOCK_M * start_m

    # Early exit for out-of-bounds query blocks.
    if block_start_loc >= sq:
        return

    # Adjust base pointers to current sequence and head
    Q += (q_start * stride_qbs + cur_head * stride_qh).to(tl.int64)
    K += (k_start * stride_kbs + cur_kv_head * stride_kh).to(tl.int64)
    V += (k_start * stride_vbs + cur_kv_head * stride_vh).to(tl.int64)
    Out += (q_start * stride_obs + cur_head * stride_oh).to(tl.int64)

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)

    # Load Q using block pointer
    p_q = tl.make_block_ptr(
        Q,
        shape=(sq, D_QK),
        strides=(stride_qbs, 1),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_DQK),
        order=(1, 0),
    )
    if EVEN_DQK:
        q = tl.load(p_q, boundary_check=(0,), padding_option="zero")
    else:
        q = tl.load(p_q, boundary_check=(0, 1), padding_option="zero")

    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DV], dtype=tl.float32)

    # Compute iteration bounds.
    end_n = sk
    if IS_CAUSAL:
        end_n = tl.minimum(sk, (start_m + 1) * BLOCK_M)

    # Split the inner loop into two phases
    if IS_CAUSAL:
        unmasked_end = tl.minimum((start_m * BLOCK_M // BLOCK_N) * BLOCK_N, end_n)
    else:
        unmasked_end = (end_n // BLOCK_N) * BLOCK_N

    # ── Phase 1: unmasked inner loop ─────────────────────────────────── #
    # Create K and V block pointers for the loop
    p_k = tl.make_block_ptr(
        K,
        shape=(D_QK, sk),
        strides=(1, stride_kbs),
        offsets=(0, 0),
        block_shape=(BLOCK_DQK, BLOCK_N),
        order=(0, 1),
    )
    p_v = tl.make_block_ptr(
        V,
        shape=(sk, D_V),
        strides=(stride_vbs, 1),
        offsets=(0, 0),
        block_shape=(BLOCK_N, BLOCK_DV),
        order=(1, 0),
    )

    for start_n in range(0, unmasked_end, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)

        # Load K and V
        if EVEN_DQK:
            k = tl.load(p_k, boundary_check=(1,), padding_option="zero")
        else:
            k = tl.load(p_k, boundary_check=(0, 1), padding_option="zero")

        qk = tl.dot(q, k)
        qk = qk * sm_scale
        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        qk -= m_ij[:, None]
        p = tl.math.exp2(qk)
        l_ij = tl.sum(p, 1)

        alpha = tl.math.exp2(m_i - m_ij)
        l_i = l_i * alpha + l_ij
        acc = acc * alpha[:, None]

        if EVEN_DV:
            v = tl.load(p_v, boundary_check=(0,), padding_option="zero")
        else:
            v = tl.load(p_v, boundary_check=(0, 1), padding_option="zero")

        p = p.to(v.dtype)
        acc = tl.dot(p, v, acc)
        m_i = m_ij

        # Advance pointers
        p_k = tl.advance(p_k, (0, BLOCK_N))
        p_v = tl.advance(p_v, (BLOCK_N, 0))

    # ── Phase 2: masked inner loop ───────────────────────────────────── #
    # Reset pointers to unmasked_end position
    p_k = tl.make_block_ptr(
        K,
        shape=(D_QK, sk),
        strides=(1, stride_kbs),
        offsets=(0, unmasked_end),
        block_shape=(BLOCK_DQK, BLOCK_N),
        order=(0, 1),
    )
    p_v = tl.make_block_ptr(
        V,
        shape=(sk, D_V),
        strides=(stride_vbs, 1),
        offsets=(unmasked_end, 0),
        block_shape=(BLOCK_N, BLOCK_DV),
        order=(1, 0),
    )

    offs_n = tl.arange(0, BLOCK_N)
    for start_n in range(unmasked_end, end_n, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)

        pos_q = offs_m[:, None]
        pos_k = start_n + offs_n[None, :]

        mask = pos_k < sk
        if IS_CAUSAL:
            mask = mask & (pos_q >= pos_k)

        # Load K and V with masking
        if EVEN_DQK:
            k = tl.load(p_k, boundary_check=(1,), padding_option="zero")
        else:
            k = tl.load(p_k, boundary_check=(0, 1), padding_option="zero")

        qk = tl.dot(q, k)
        qk = tl.where(mask, qk * sm_scale, -1.0e8)
        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        qk -= m_ij[:, None]
        p = tl.math.exp2(qk)
        l_ij = tl.sum(p, 1)

        alpha = tl.math.exp2(m_i - m_ij)
        l_i = l_i * alpha + l_ij
        acc = acc * alpha[:, None]

        if EVEN_DV:
            v = tl.load(p_v, boundary_check=(0,), padding_option="zero")
        else:
            v = tl.load(p_v, boundary_check=(0, 1), padding_option="zero")

        p = p.to(v.dtype)
        acc = tl.dot(p, v, acc)
        m_i = m_ij

        # Advance pointers
        p_k = tl.advance(p_k, (0, BLOCK_N))
        p_v = tl.advance(p_v, (BLOCK_N, 0))

    # ── Epilogue ─────────────────────────────────────────────────────── #
    acc = acc / l_i[:, None]

    # Store output (use manual pointers for compatibility)
    offs_d_v = tl.arange(0, BLOCK_DV)
    off_o = offs_m[:, None] * stride_obs + offs_d_v[None, :]
    if EVEN_DV:
        tl.store(Out + off_o, acc, mask=(offs_m[:, None] < sq))
    else:
        tl.store(
            Out + off_o,
            acc,
            mask=(offs_m[:, None] < sq) & (offs_d_v[None, :] < D_V),
        )

    if HAS_LSE:
        # Natural-log LSE: max(L) + log(sum exp(L - max)); m_i is max of L / ln2
        lse_row = m_i * 0.6931471805599453 + tl.log(l_i)
        lse_off = cur_head * stride_lse_h + (q_start + offs_m) * stride_lse_t
        tl.store(lse_ptr + lse_off, lse_row, mask=offs_m < sq)


@dataclasses.dataclass(frozen=True)
class TritonFlashAttnVarlenConfig:
    """Platform-specific launch configuration for ``_fwd_kernel_varlen``.

    Attributes:
        BLOCK_M:    Tile size along the query-sequence (M) dimension.
        BLOCK_N:    Tile size along the key-sequence (N) dimension.
        num_warps:  Number of warps per Triton program instance.
        num_stages: Number of software-pipelining stages.
        use_block_ptr: Use Triton block pointers (``tl.make_block_ptr``) for
                       memory access. Block pointers can improve memory coalescing,
                       reduce pointer arithmetic overhead, and enable better compiler
                       optimizations. Especially beneficial on XPU.
    """

    BLOCK_M: int
    BLOCK_N: int
    num_warps: int
    num_stages: int
    use_block_ptr: bool = False


@functools.lru_cache(maxsize=8)
def _get_default_config(
    dtype: torch.dtype,
) -> TritonFlashAttnVarlenConfig:
    """Return platform-specific launch parameters.

    The heuristics mirror what the rest of vLLM's Triton ops use:

    * **CUDA SM >= 80** (A100 / H100): large tiles, 8 warps, 3 stages.
    * **XPU** (Intel Data Center GPU Max): smaller tiles, 4 warps, 2 stages
      — matching ``triton_reshape_and_cache_flash`` XPU defaults.
    * **Fallback** (older CUDA, ROCm, …): moderate tiles.
    """
    from vllm.platforms import current_platform

    # fp32 always gets a small BLOCK_M regardless of platform.
    if dtype == torch.float32:
        block_m = 32
    elif current_platform.is_xpu():
        block_m = 64
    elif current_platform.is_cuda_alike() and current_platform.has_device_capability(
        80
    ):
        block_m = 128
    else:
        block_m = 64

    if current_platform.is_xpu():
        return TritonFlashAttnVarlenConfig(
            BLOCK_M=block_m,
            BLOCK_N=min(block_m, 32),
            num_warps=8,  # Benchmarking shows 8 warps is optimal on XPU
            num_stages=2,
            use_block_ptr=True,  # Block pointers improve XPU performance
        )

    # CUDA / fallback defaults (tuned for A100-class GPUs with SM >= 80).
    return TritonFlashAttnVarlenConfig(
        BLOCK_M=block_m,
        BLOCK_N=min(block_m, 64),
        num_warps=8,
        num_stages=3,
        use_block_ptr=False,  # Keep manual pointers for CUDA (well-tested)
    )


def flash_attn_varlen_triton(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    causal: bool,
    softmax_scale: float | None = None,
    return_softmax_lse: bool = False,
    config: TritonFlashAttnVarlenConfig | None = None,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Varlen multi-head attention on packed Q, K, V (FlashAttention-style layout).

    Args:
        q: [total_q, num_q_heads, D_qk]
        k: [total_k, num_kv_heads, D_qk]
        v: [total_k, num_kv_heads, D_v]
        cu_seqlens_q: int32, shape [num_seqs + 1], cumulative lengths for Q
        cu_seqlens_k: int32, shape [num_seqs + 1], cumulative lengths for K
        max_seqlen_q: upper bound on per-sequence query length (grid sizing)
        max_seqlen_k: unused in kernel (API parity with FlashAttention)
        causal: if True, mask keys j with j <= query index i per sequence; requires
            equal sequence lengths for Q and K in each batch row.
        softmax_scale: typically 1/sqrt(D_qk); defaults to 1/sqrt(D_qk)
        return_softmax_lse: if True, also return LSE with shape [num_q_heads, total_q]
        config: optional platform-specific launch configuration; when ``None``
            the default for the current platform / dtype is used (see
            :func:`_get_default_config`).

    Returns:
        out: [total_q, num_q_heads, D_v]
        softmax_lse (optional): float32 [num_q_heads, total_q]
    """
    del max_seqlen_k  # API compatibility with flash_attn_varlen_func

    # Use tensor shapes directly to avoid GPU-sync from .item() calls.
    total_q, num_q_heads, d_qk = q.shape
    total_k, num_kv_heads, _ = k.shape
    d_v = v.shape[2]
    num_seqs = cu_seqlens_q.numel() - 1
    kv_group_num = num_q_heads // num_kv_heads

    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(float(d_qk))
    sm_scale = float(softmax_scale) * RCP_LN2

    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()

    out = torch.empty(
        total_q,
        num_q_heads,
        d_v,
        device=q.device,
        dtype=q.dtype,
    )
    lse = None
    if return_softmax_lse:
        lse = torch.empty(
            num_q_heads,
            total_q,
            device=q.device,
            dtype=torch.float32,
        )

    if config is None:
        config = _get_default_config(q.dtype)

    BLOCK_M = config.BLOCK_M
    BLOCK_DQK = triton.next_power_of_2(d_qk)
    BLOCK_DV = triton.next_power_of_2(d_v)
    EVEN_DQK = d_qk == BLOCK_DQK
    EVEN_DV = d_v == BLOCK_DV
    BLOCK_N = config.BLOCK_N

    grid = (num_seqs, num_q_heads, triton.cdiv(max_seqlen_q, BLOCK_M))

    # Select kernel based on config
    kernel = (
        _fwd_kernel_varlen_block_ptr if config.use_block_ptr else _fwd_kernel_varlen
    )

    kernel[grid](
        q,
        k,
        v,
        sm_scale,
        cu_seqlens_q,
        cu_seqlens_k,
        out,
        lse if return_softmax_lse else out,
        q.stride(0),
        q.stride(1),
        k.stride(0),
        k.stride(1),
        v.stride(0),
        v.stride(1),
        out.stride(0),
        out.stride(1),
        lse.stride(0) if lse is not None else 0,
        lse.stride(1) if lse is not None else 0,
        kv_group_num=kv_group_num,
        BLOCK_M=BLOCK_M,
        BLOCK_DQK=BLOCK_DQK,
        BLOCK_DV=BLOCK_DV,
        BLOCK_N=BLOCK_N,
        IS_CAUSAL=causal,
        HAS_LSE=return_softmax_lse,
        D_QK=d_qk,
        D_V=d_v,
        EVEN_DQK=EVEN_DQK,
        EVEN_DV=EVEN_DV,
        num_warps=config.num_warps,
        num_stages=config.num_stages,
    )

    if return_softmax_lse:
        return out, lse
    return out
