# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Songlin Yang, Yu Zhang
#
# This file contains code copied from the flash-linear-attention project.
# The original source code was licensed under the MIT license and included
# the following copyright notice:
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

# ruff: noqa: E501


import torch

from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton

from .index import prepare_chunk_indices
from .op import exp
from .utils import FLA_CHUNK_SIZE, check_shared_mem, is_nvidia_hopper

BKV_LIST = [64, 128] if check_shared_mem() else [32, 64]
NUM_WARPS = [2, 4] if is_nvidia_hopper else [2, 4, 8]

# Three things keep the fastest configuration out of this space on gfx942, in
# descending order of what they are worth.
#
# 1. The space never sets matrix_instr_nonkdim, so the backend emits
#    v_mfma_f32_32x32x8_bf16 and the 16x16 form is never tried. Adding it to the tiles
#    already here, changing nothing else, is worth 1.427x -- 60% of the total. This
#    kernel is bound by LDS traffic rather than MFMA throughput, so the trade is a win:
#    v_mfma_f32_16x16x16_bf16 needs more instructions (72 vs 40) but its fragment layout
#    matches the LDS tile, cutting ds_read 80 -> 36 and ds_write 36 -> 22.
#
# 2. num_stages=1 is not in the space. Dropping the pipeline frees the second LDS
#    buffer, 24576 B -> 8192 B, which moves the kernel off the LDS occupancy ceiling
#    onto the register ceiling: 2.0 -> 4.0 waves/SIMD. Worth 1.137x alone, and 1.580x
#    together with the MFMA width.
#
# 3. check_shared_mem() compares the device's LDS against Backend.DEFAULT = 102400 B,
#    a threshold taken from a table with only NVIDIA entries, so it is False on gfx942
#    and BK/BV=128 is dropped. BV=128 is restored here; BK=128 is not. BK is the K-loop
#    tile, so BK=128 reduces K in one block where the stock space uses two or four,
#    re-associating an fp32 sum: the final recurrent state is unchanged but the output
#    differs in the last bits of a few elements (at T=30720, 657 of 188743680, by at most
#    2.4e-4 absolute). It wins an isolated long-sequence microbenchmark and loses at this
#    model's chunked-prefill size, where 32768 B of LDS halves occupancy, so it is not
#    worth a change in numerics and is left out.
#
# Medians of 5 runs on MI325X at the Qwen3-Next GDN shapes, T=30720; stock arm stable
# to 0.3%, full patch to 3.3%. Every config added here is bit-identical to the stock
# result, output and final state alike.
#
# The space below is deliberately narrower than the stock one: num_warps stops at 4 where
# NUM_WARPS carries 8 off-Hopper, and num_stages at 2. A wider sweep over
# BK in {32,64,128} x BV in {64,128} x warps in {1,2,4} x stages in {1,2,3} finds the same
# winner at the same speed and costs 77.8 s of cold autotune against 49.9 s, so the
# candidates that never won were dropped.
#
# Add the configs behind on_gfx942() rather than moving the threshold, which would
# change every other backend too, and let the existing autotuner pick between them.
#
# ``matrix_instr_nonkdim`` is an option of Triton's AMD backend rather than a kernel
# argument: the backend lifts it out of the config kwargs at compile time. A Triton that
# does not recognize it raises instead of being skipped, because the autotuner only
# swallows ``OutOfResources``. That is the one way these configs can break a build that
# works today, so it is pinned by the tests rather than left implicit.
_GFX942_CONFIGS: list[triton.Config] = []
if current_platform.is_rocm():
    from vllm.platforms.rocm import on_gfx942

    if on_gfx942():
        _GFX942_CONFIGS = [
            triton.Config(
                {"BK": 64, "BV": BV, "matrix_instr_nonkdim": 16},
                num_warps=num_warps,
                num_stages=num_stages,
            )
            for BV in [64, 128]
            for num_warps in [2, 4]
            for num_stages in [1, 2]
        ]


@triton.heuristics(
    {
        "USE_G": lambda args: args["g"] is not None,
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
    }
)
@triton.autotune(
    configs=[
        triton.Config({"BK": BK, "BV": BV}, num_warps=num_warps, num_stages=num_stages)
        for BK in BKV_LIST
        for BV in BKV_LIST
        for num_warps in NUM_WARPS
        for num_stages in [2, 3, 4]
    ]
    + _GFX942_CONFIGS,
    key=["H", "K", "V", "BT"],
)
@triton.jit(do_not_specialize=["T"])
def chunk_fwd_kernel_o(
    q,
    k,
    v,
    h,
    g,
    o,
    cu_seqlens,
    chunk_indices,
    scale,
    T,
    H: tl.constexpr,
    Hg: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_G: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_v, i_t, i_bh = (
        tl.program_id(0),
        tl.program_id(1).to(tl.int64),
        tl.program_id(2).to(tl.int64),
    )
    i_b, i_h = i_bh // H, i_bh % H

    if IS_VARLEN:
        i_tg = i_t
        i_n, i_t = (
            tl.load(chunk_indices + i_t * 2).to(tl.int32),
            tl.load(chunk_indices + i_t * 2 + 1).to(tl.int64),
        )
        bos, eos = (
            tl.load(cu_seqlens + i_n).to(tl.int32),
            tl.load(cu_seqlens + i_n + 1).to(tl.int32),
        )
        T = eos - bos
        NT = tl.cdiv(T, BT)
    else:
        NT = tl.cdiv(T, BT)
        i_tg = i_b * NT + i_t
        bos, eos = i_b * T, i_b * T + T

    # offset calculation
    q += (bos * Hg + i_h // (H // Hg)) * K
    k += (bos * Hg + i_h // (H // Hg)) * K
    v += (bos * H + i_h) * V
    o += (bos * H + i_h) * V
    h += (i_tg * H + i_h).to(tl.int64) * V * K

    b_o = tl.zeros([BT, BV], dtype=tl.float32)
    b_A = tl.zeros([BT, BT], dtype=tl.float32)

    o_t = i_t * BT + tl.arange(0, BT)
    m_t = o_t < T
    o_v = i_v * BV + tl.arange(0, BV)
    for i_k in range(tl.cdiv(K, BK)):
        o_k = i_k * BK + tl.arange(0, BK)
        m_k = o_k < K
        p_q = q + o_t[:, None] * (Hg * K) + o_k[None, :]
        p_k = k + o_k[:, None] + o_t[None, :] * (Hg * K)
        p_h = h + o_v[:, None] * K + o_k[None, :]
        # [BT, BK]
        b_q = tl.load(p_q, mask=m_t[:, None] & m_k[None, :], other=0.0)
        # [BK, BT]
        b_k = tl.load(p_k, mask=m_k[:, None] & m_t[None, :], other=0.0)
        # [BV, BK]
        b_h = tl.load(p_h, mask=(o_v[:, None] < V) & m_k[None, :], other=0.0)

        # [BT, BK] @ [BK, BV] -> [BT, BV]
        b_o += tl.dot(b_q, tl.trans(b_h))
        # [BT, BK] @ [BK, BT] -> [BT, BT]
        b_A += tl.dot(b_q, b_k)

    if USE_G:
        g += bos * H + i_h
        p_g = g + o_t * H
        b_g = tl.load(p_g, mask=m_t, other=0.0)
        b_o = b_o * exp(b_g)[:, None]
        b_A = b_A * exp(b_g[:, None] - b_g[None, :])

    m_A = (o_t[:, None] >= o_t[None, :]) & (m_t[:, None] & m_t)
    b_A = tl.where(m_A, b_A, 0)

    p_v = v + o_t[:, None] * (H * V) + o_v[None, :]
    p_o = o + o_t[:, None] * (H * V) + o_v[None, :]
    b_v = tl.load(p_v, mask=m_t[:, None] & (o_v < V)[None, :], other=0.0)

    # to fix mma -> mma layout conversion
    # already solved by triton v3.2 or higher
    b_o = b_o * scale + tl.dot(b_A.to(b_v.dtype), b_v) * scale
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=m_t[:, None] & (o_v < V)[None, :])


def chunk_fwd_o(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    h: torch.Tensor,
    g: torch.Tensor | None = None,  # cumsum of log decay
    scale: float | None = None,
    cu_seqlens: torch.Tensor | None = None,
    chunk_indices: torch.Tensor | None = None,
    chunk_size: int = FLA_CHUNK_SIZE,
    core_attn_out: torch.Tensor | None = None,
) -> torch.Tensor:
    B, T, Hg, K, V = *q.shape, v.shape[-1]
    H = v.shape[-2]
    BT = chunk_size
    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)
    if scale is None:
        scale = k.shape[-1] ** -0.5

    if core_attn_out is not None:
        assert core_attn_out.numel() >= v.numel(), (
            f"core_attn_out too small: {core_attn_out.numel()} < {v.numel()}"
        )
        o = core_attn_out[: v.numel()].view(*v.shape)
    else:
        o = torch.empty_like(v)

    def grid(meta):
        return (triton.cdiv(V, meta["BV"]), NT, B * H)

    chunk_fwd_kernel_o[grid](
        q,
        k,
        v,
        h,
        g,
        o,
        cu_seqlens,
        chunk_indices,
        scale,
        T=T,
        H=H,
        Hg=Hg,
        K=K,
        V=V,
        BT=BT,
    )
    return o
