# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This file contains code copied from the flash-linear-attention project.
# The original source code was licensed under the MIT license and included
# the following copyright notice:
# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# Adapted from
# https://github.com/fla-org/flash-linear-attention/blob/main/fla/ops/generalized_delta_rule/dplr/chunk_h_fwd.py
# Forward path only.
# ruff: noqa: E501

import torch

from vllm.triton_utils import tl, triton

from ..index import prepare_chunk_indices, prepare_chunk_offsets
from ..op import exp
from ..utils import check_shared_mem, is_amd, use_cuda_graph

NUM_WARPS_AUTOTUNE = [2, 4, 8, 16] if is_amd else [2, 4, 8, 16, 32]


@triton.heuristics(
    {
        "USE_INITIAL_STATE": lambda args: args["h0"] is not None,
        "STORE_FINAL_STATE": lambda args: args["ht"] is not None,
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
    }
)
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in NUM_WARPS_AUTOTUNE
        for num_stages in [2, 3, 4]
    ],
    key=["BT", "BK", "BV"],
    use_cuda_graph=use_cuda_graph,
)
@triton.jit(do_not_specialize=["T"])
def chunk_dplr_fwd_kernel_h(
    kg,
    v,
    w,
    bg,
    u,
    v_new,
    gk,
    h,
    h0,
    ht,
    cu_seqlens,
    chunk_offsets,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_k, i_v, i_nh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_n, i_h = i_nh // H, i_nh % H
    if IS_VARLEN:
        bos, eos = (
            tl.load(cu_seqlens + i_n).to(tl.int32),
            tl.load(cu_seqlens + i_n + 1).to(tl.int32),
        )
        T = eos - bos
        NT = tl.cdiv(T, BT)
        boh = tl.load(chunk_offsets + i_n).to(tl.int32)
    else:
        bos, eos = i_n * T, i_n * T + T
        NT = tl.cdiv(T, BT)
        boh = i_n * NT
    o_k = i_k * BK + tl.arange(0, BK)

    b_h = tl.zeros([BK, BV], dtype=tl.float32)
    if USE_INITIAL_STATE:
        p_h0 = tl.make_block_ptr(
            h0 + i_nh * K * V, (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0)
        )
        b_h = tl.load(p_h0, boundary_check=(0, 1)).to(tl.float32)

    for i_t in range(NT):
        p_h = tl.make_block_ptr(
            h + ((boh + i_t) * H + i_h) * K * V,
            (K, V),
            (V, 1),
            (i_k * BK, i_v * BV),
            (BK, BV),
            (1, 0),
        )
        tl.store(p_h, b_h.to(p_h.dtype.element_ty), boundary_check=(0, 1))

        b_hc = tl.zeros([BK, BV], dtype=tl.float32)
        for i_c in range(tl.cdiv(min(BT, T - i_t * BT), BC)):
            p_kg = tl.make_block_ptr(
                kg + (bos * H + i_h) * K,
                (K, T),
                (1, H * K),
                (i_k * BK, i_t * BT + i_c * BC),
                (BK, BC),
                (0, 1),
            )
            p_bg = tl.make_block_ptr(
                bg + (bos * H + i_h) * K,
                (K, T),
                (1, H * K),
                (i_k * BK, i_t * BT + i_c * BC),
                (BK, BC),
                (0, 1),
            )
            p_w = tl.make_block_ptr(
                w + (bos * H + i_h) * K,
                (T, K),
                (H * K, 1),
                (i_t * BT + i_c * BC, i_k * BK),
                (BC, BK),
                (1, 0),
            )
            p_v = tl.make_block_ptr(
                v + (bos * H + i_h) * V,
                (T, V),
                (H * V, 1),
                (i_t * BT + i_c * BC, i_v * BV),
                (BC, BV),
                (1, 0),
            )
            p_u = tl.make_block_ptr(
                u + (bos * H + i_h) * V,
                (T, V),
                (H * V, 1),
                (i_t * BT + i_c * BC, i_v * BV),
                (BC, BV),
                (1, 0),
            )
            p_v_new = tl.make_block_ptr(
                v_new + (bos * H + i_h) * V,
                (T, V),
                (H * V, 1),
                (i_t * BT + i_c * BC, i_v * BV),
                (BC, BV),
                (1, 0),
            )
            b_kg = tl.load(p_kg, boundary_check=(0, 1))
            b_v = tl.load(p_v, boundary_check=(0, 1))
            b_w = tl.load(p_w, boundary_check=(0, 1))
            b_bg = tl.load(p_bg, boundary_check=(0, 1))
            b_v2 = tl.dot(b_w, b_h.to(b_w.dtype)) + tl.load(p_u, boundary_check=(0, 1))
            b_hc += tl.dot(b_kg, b_v)
            b_hc += tl.dot(b_bg.to(b_hc.dtype), b_v2)
            tl.store(p_v_new, b_v2.to(p_v_new.dtype.element_ty), boundary_check=(0, 1))

        last_idx = min((i_t + 1) * BT, T) - 1
        b_g_last = tl.load(
            gk + (bos + last_idx) * H * K + i_h * K + o_k, mask=o_k < K
        ).to(tl.float32)
        b_h *= exp(b_g_last[:, None])
        b_h += b_hc

    if STORE_FINAL_STATE:
        p_ht = tl.make_block_ptr(
            ht + i_nh * K * V, (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0)
        )
        tl.store(
            p_ht,
            b_h.to(p_ht.dtype.element_ty, fp_downcast_rounding="rtne"),
            boundary_check=(0, 1),
        )


def chunk_dplr_fwd_h(
    kg: torch.Tensor,
    v: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    bg: torch.Tensor,
    gk: torch.Tensor,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    cu_seqlens: torch.Tensor | None = None,
    chunk_size: int = 64,
    chunk_indices: torch.Tensor | None = None,
    return_varlen_states: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    B, T, H, K, V = *kg.shape, u.shape[-1]
    BT = chunk_size

    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
    if cu_seqlens is None:
        N, NT, chunk_offsets = B, triton.cdiv(T, BT), None
    else:
        N = len(cu_seqlens) - 1
        NT = len(chunk_indices)
        chunk_offsets = prepare_chunk_offsets(cu_seqlens, BT)
    BK = max(triton.next_power_of_2(K), 16)
    assert BK <= 256, "current kernel does not support head dimension larger than 256."

    if check_shared_mem("hopper", kg.device.index):
        BV = 64
        BC = 64 if K <= 128 else 32
    elif check_shared_mem("ampere", kg.device.index):
        BV = 32
        BC = 32
    else:
        BV = 16
        BC = 16

    BC = min(BT, BC)
    NK = triton.cdiv(K, BK)
    NV = triton.cdiv(V, BV)
    assert NK == 1, "NK > 1 is not supported"

    h = kg.new_empty(B, NT, H, K, V)
    # When prefix caching is on we need per-chunk-END states too. Force-enable
    # final_state in that case since the very last chunk's end is the
    # per-sequence final state.
    if return_varlen_states:
        output_final_state = True
    final_state = (
        kg.new_empty(N, H, K, V, dtype=torch.float32) if output_final_state else None
    )
    v_new = torch.empty_like(u)
    grid = (NK, NV, N * H)
    chunk_dplr_fwd_kernel_h[grid](
        kg=kg,
        v=v,
        w=w,
        bg=bg,
        u=u,
        v_new=v_new,
        h=h,
        gk=gk,
        h0=initial_state,
        ht=final_state,
        cu_seqlens=cu_seqlens,
        chunk_offsets=chunk_offsets,
        T=T,
        H=H,
        K=K,
        V=V,
        BT=BT,
        BC=BC,
        BK=BK,
        BV=BV,
    )

    varlen_states: torch.Tensor | None = None
    if return_varlen_states:
        # The kernel writes ``h[k]`` = state at the START of chunk k. For
        # prefix caching we want state at the END of each chunk (= start of
        # the next chunk in the same sequence, or final_state for the
        # last chunk of each sequence). Keep varlen_states in fp32 to match
        # final_state and the canonical recurrent-state cache dtype.
        h_squeezed = h.view(B * NT, H, K, V) if cu_seqlens is None else h[0]
        varlen_states = torch.empty(
            h_squeezed.shape, dtype=torch.float32, device=h_squeezed.device
        )
        if h_squeezed.shape[0] > 1:
            varlen_states[:-1] = h_squeezed[1:].to(torch.float32)
        if cu_seqlens is None:
            for b in range(B):
                varlen_states[b * NT + NT - 1] = final_state[b]
        else:
            assert chunk_offsets is not None
            last_chunk_indices = (chunk_offsets[1:] - 1).long()
            varlen_states[last_chunk_indices] = final_state

    return h, v_new, final_state, varlen_states
