# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This file contains code copied from the flash-linear-attention project.
# The original source code was licensed under the MIT license and included
# the following copyright notice:
# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
# ruff: noqa: E501

from dataclasses import dataclass
from typing import Any

import torch

from vllm.model_executor.warmup.jit_warmup_triton_helper import (
    LaunchSpec,
    TritonWarmupTensor,
    VllmTritonJitKernel,
    kernel_launcher,
)
from vllm.third_party.flash_linear_attention.ops.chunk_delta_h import (
    chunk_gated_delta_rule_fwd_h,
)
from vllm.third_party.flash_linear_attention.ops.cumsum import chunk_local_cumsum
from vllm.third_party.flash_linear_attention.ops.index import prepare_chunk_indices
from vllm.third_party.flash_linear_attention.ops.l2norm import l2norm_fwd
from vllm.third_party.flash_linear_attention.ops.op import exp2, log
from vllm.third_party.flash_linear_attention.ops.utils import FLA_CHUNK_SIZE, is_amd
from vllm.triton_utils import tl, triton
from vllm.utils.math_utils import RCP_LN2, cdiv, next_power_of_2

from .chunk_intra import chunk_kda_fwd_intra

BT_LIST_AUTOTUNE = [32, 64, 128]
NUM_WARPS_AUTOTUNE = [2, 4, 8, 16] if is_amd else [4, 8, 16, 32]


@triton.heuristics(
    {
        "STORE_QG": lambda args: args["qg"] is not None,
        "STORE_KG": lambda args: args["kg"] is not None,
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
    }
)
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [2, 4, 8]
        for num_stages in [2, 3, 4]
    ],
    key=["H", "K", "V", "BT", "BK", "BV", "IS_VARLEN"],
)
@triton.jit(do_not_specialize=["T"])
def recompute_w_u_fwd_kernel(
    q,
    k,
    qg,
    kg,
    v,
    beta,
    w,
    u,
    A,
    gk,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    STORE_QG: tl.constexpr,
    STORE_KG: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    DOT_PRECISION: tl.constexpr,
):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        i_n, i_t = (
            tl.load(chunk_indices + i_t * 2).to(tl.int32),
            tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32),
        )
        bos, eos = (
            tl.load(cu_seqlens + i_n).to(tl.int32),
            tl.load(cu_seqlens + i_n + 1).to(tl.int32),
        )
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T
    p_b = tl.make_block_ptr(beta + bos * H + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,))
    b_b = tl.load(p_b, boundary_check=(0,)).to(tl.float32)

    p_A = tl.make_block_ptr(
        A + (bos * H + i_h) * BT, (T, BT), (H * BT, 1), (i_t * BT, 0), (BT, BT), (1, 0)
    )
    b_A = tl.load(p_A, boundary_check=(0, 1))

    for i_v in range(tl.cdiv(V, BV)):
        p_v = tl.make_block_ptr(
            v + (bos * H + i_h) * V,
            (T, V),
            (H * V, 1),
            (i_t * BT, i_v * BV),
            (BT, BV),
            (1, 0),
        )
        p_u = tl.make_block_ptr(
            u + (bos * H + i_h) * V,
            (T, V),
            (H * V, 1),
            (i_t * BT, i_v * BV),
            (BT, BV),
            (1, 0),
        )
        b_v = tl.load(p_v, boundary_check=(0, 1))
        b_vb = (b_v * b_b[:, None]).to(b_v.dtype)
        b_u = tl.dot(b_A, b_vb, input_precision=DOT_PRECISION)
        tl.store(p_u, b_u.to(p_u.dtype.element_ty), boundary_check=(0, 1))

    for i_k in range(tl.cdiv(K, BK)):
        p_w = tl.make_block_ptr(
            w + (bos * H + i_h) * K,
            (T, K),
            (H * K, 1),
            (i_t * BT, i_k * BK),
            (BT, BK),
            (1, 0),
        )
        p_k = tl.make_block_ptr(
            k + (bos * H + i_h) * K,
            (T, K),
            (H * K, 1),
            (i_t * BT, i_k * BK),
            (BT, BK),
            (1, 0),
        )
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_kb = b_k * b_b[:, None]

        p_gk = tl.make_block_ptr(
            gk + (bos * H + i_h) * K,
            (T, K),
            (H * K, 1),
            (i_t * BT, i_k * BK),
            (BT, BK),
            (1, 0),
        )
        b_gk = tl.load(p_gk, boundary_check=(0, 1))
        b_kb *= exp2(b_gk)
        if STORE_QG:
            p_q = tl.make_block_ptr(
                q + (bos * H + i_h) * K,
                (T, K),
                (H * K, 1),
                (i_t * BT, i_k * BK),
                (BT, BK),
                (1, 0),
            )
            p_qg = tl.make_block_ptr(
                qg + (bos * H + i_h) * K,
                (T, K),
                (H * K, 1),
                (i_t * BT, i_k * BK),
                (BT, BK),
                (1, 0),
            )
            b_q = tl.load(p_q, boundary_check=(0, 1))
            b_qg = b_q * exp2(b_gk)
            tl.store(p_qg, b_qg.to(p_qg.dtype.element_ty), boundary_check=(0, 1))
        if STORE_KG:
            last_idx = min(i_t * BT + BT, T) - 1

            o_k = i_k * BK + tl.arange(0, BK)
            m_k = o_k < K
            b_gn = tl.load(
                gk + ((bos + last_idx) * H + i_h) * K + o_k, mask=m_k, other=0.0
            )
            b_kg = b_k * exp2(b_gn - b_gk)

            p_kg = tl.make_block_ptr(
                kg + (bos * H + i_h) * K,
                (T, K),
                (H * K, 1),
                (i_t * BT, i_k * BK),
                (BT, BK),
                (1, 0),
            )
            tl.store(p_kg, b_kg.to(p_kg.dtype.element_ty), boundary_check=(0, 1))

        b_w = tl.dot(b_A, b_kb.to(b_k.dtype))
        tl.store(p_w, b_w.to(p_w.dtype.element_ty), boundary_check=(0, 1))


class KimiK3RecomputeWUKernel(
    VllmTritonJitKernel["KimiK3RecomputeWUKernel.CompileKey"]
):
    """JIT owner for the autotuned w/u recompute kernel."""

    # Triton's Autotuner owns num_warps/num_stages and compile-warms every
    # candidate for each logical key below. Only the compile is pre-warmed;
    # the autotune decision itself runs at the first real launch / profile_run.
    kernel = staticmethod(recompute_w_u_fwd_kernel)

    @dataclass(frozen=True)
    class CompileKey:
        k_dtype: torch.dtype
        kg_dtype: torch.dtype
        v_dtype: torch.dtype
        beta_dtype: torch.dtype
        w_dtype: torch.dtype
        u_dtype: torch.dtype
        a_dtype: torch.dtype
        gk_dtype: torch.dtype
        num_heads: int
        qk_head_dim: int
        v_head_dim: int
        block_t: int
        block_k: int
        block_v: int
        store_qg: bool
        store_kg: bool
        is_varlen: bool
        dot_precision: str

    def dispatch(  # type: ignore[override]
        self,
        *,
        k_dtype: torch.dtype,
        kg_dtype: torch.dtype,
        v_dtype: torch.dtype,
        beta_dtype: torch.dtype,
        w_dtype: torch.dtype,
        u_dtype: torch.dtype,
        a_dtype: torch.dtype,
        gk_dtype: torch.dtype,
        num_heads: int,
        qk_head_dim: int,
        v_head_dim: int,
        block_t: int = FLA_CHUNK_SIZE,
        block_k: int = 64,
        block_v: int = 64,
        store_qg: bool = False,
        store_kg: bool = True,
        is_varlen: bool = True,
        dot_precision: str = "ieee",
    ) -> CompileKey:
        return self.CompileKey(
            k_dtype=k_dtype,
            kg_dtype=kg_dtype,
            v_dtype=v_dtype,
            beta_dtype=beta_dtype,
            w_dtype=w_dtype,
            u_dtype=u_dtype,
            a_dtype=a_dtype,
            gk_dtype=gk_dtype,
            num_heads=num_heads,
            qk_head_dim=qk_head_dim,
            v_head_dim=v_head_dim,
            block_t=block_t,
            block_k=block_k,
            block_v=block_v,
            store_qg=store_qg,
            store_kg=store_kg,
            is_varlen=is_varlen,
            dot_precision=dot_precision,
        )

    def get_warmup_keys(
        self,
        *,
        k_dtype: torch.dtype,
        kg_dtype: torch.dtype,
        v_dtype: torch.dtype,
        beta_dtype: torch.dtype,
        w_dtype: torch.dtype,
        u_dtype: torch.dtype,
        a_dtype: torch.dtype,
        gk_dtype: torch.dtype,
        num_heads: int,
        qk_head_dim: int,
        v_head_dim: int,
        block_t: int = FLA_CHUNK_SIZE,
        block_k: int = 64,
        block_v: int = 64,
        store_qg: bool = False,
        store_kg: bool = True,
        is_varlen: bool = True,
        dot_precision: str = "ieee",
    ) -> list[CompileKey]:
        return self._trace_dispatch(self.dispatch)(
            k_dtype=k_dtype,
            kg_dtype=kg_dtype,
            v_dtype=v_dtype,
            beta_dtype=beta_dtype,
            w_dtype=w_dtype,
            u_dtype=u_dtype,
            a_dtype=a_dtype,
            gk_dtype=gk_dtype,
            num_heads=num_heads,
            qk_head_dim=qk_head_dim,
            v_head_dim=v_head_dim,
            block_t=block_t,
            block_k=block_k,
            block_v=block_v,
            store_qg=store_qg,
            store_kg=store_kg,
            is_varlen=is_varlen,
            dot_precision=dot_precision,
        )

    def warmup_inputs(self, compile_key: CompileKey) -> dict[str, Any]:
        b = 1
        t = compile_key.block_t
        h = compile_key.num_heads
        k = compile_key.qk_head_dim
        v = compile_key.v_head_dim
        nt = 1
        return {
            "q": None,
            "k": TritonWarmupTensor(compile_key.k_dtype, shape=(b, t, h, k)),
            "qg": (
                TritonWarmupTensor(compile_key.gk_dtype, shape=(b, t, h, k))
                if compile_key.store_qg
                else None
            ),
            "kg": (
                TritonWarmupTensor(compile_key.kg_dtype, shape=(b, t, h, k))
                if compile_key.store_kg
                else None
            ),
            "v": TritonWarmupTensor(compile_key.v_dtype, shape=(b, t, h, v)),
            "beta": TritonWarmupTensor(compile_key.beta_dtype, shape=(b, t, h)),
            "w": TritonWarmupTensor(compile_key.w_dtype, shape=(b, t, h, k)),
            "u": TritonWarmupTensor(compile_key.u_dtype, shape=(b, t, h, v)),
            "A": TritonWarmupTensor(
                compile_key.a_dtype, shape=(b, t, h, compile_key.block_t)
            ),
            "gk": TritonWarmupTensor(compile_key.gk_dtype, shape=(b, t, h, k)),
            "cu_seqlens": (
                TritonWarmupTensor(torch.int32, shape=(b + 1,))
                if compile_key.is_varlen
                else None
            ),
            "chunk_indices": (
                TritonWarmupTensor(torch.int32, shape=(nt, 2))
                if compile_key.is_varlen
                else None
            ),
            "dot_precision": compile_key.dot_precision,
        }

    @kernel_launcher
    def __call__(
        self,
        q: torch.Tensor | None,
        k: torch.Tensor,
        qg: torch.Tensor | None,
        kg: torch.Tensor | None,
        v: torch.Tensor,
        beta: torch.Tensor,
        w: torch.Tensor,
        u: torch.Tensor,
        A: torch.Tensor,
        gk: torch.Tensor,
        cu_seqlens: torch.Tensor | None = None,
        chunk_indices: torch.Tensor | None = None,
        dot_precision: str = "ieee",
    ) -> LaunchSpec:
        b, t, num_heads, qk_head_dim = k.shape
        v_head_dim = v.shape[-1]
        block_t = A.shape[-1]
        block_k = 64
        block_v = 64
        nt = cdiv(t, block_t)
        if chunk_indices is not None:
            nt = chunk_indices.shape[0]
        grid = (nt, b * num_heads)
        return (
            grid,
            {
                "T": t,
                "H": num_heads,
                "K": qk_head_dim,
                "V": v_head_dim,
                "BT": block_t,
                "BK": block_k,
                "BV": block_v,
                "DOT_PRECISION": dot_precision,
            },
        )


def recompute_w_u_fwd(
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    A: torch.Tensor,
    q: torch.Tensor | None = None,
    gk: torch.Tensor | None = None,
    cu_seqlens: torch.Tensor | None = None,
    chunk_indices: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    BT = A.shape[-1]
    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)

    w = torch.empty_like(k)
    u = torch.empty_like(v)
    kg = torch.empty_like(k) if gk is not None else None
    _RECOMPUTE_WU_KERNEL(
        q=q,
        k=k,
        qg=None,
        kg=kg,
        v=v,
        beta=beta,
        w=w,
        u=u,
        A=A,
        gk=gk,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        dot_precision="ieee",
    )
    return w, u, None, kg


@triton.heuristics({"IS_VARLEN": lambda args: args["cu_seqlens"] is not None})
@triton.autotune(
    configs=[
        triton.Config({"BK": BK, "BV": BV}, num_warps=num_warps, num_stages=num_stages)
        for BK in [32, 64]
        for BV in [64, 128]
        for num_warps in [2, 4, 8]
        for num_stages in [2, 3, 4]
    ],
    key=["BT"],
)
@triton.jit(do_not_specialize=["T"])
def chunk_gla_fwd_kernel_o(
    q,
    v,
    g,
    h,
    o,
    A,
    cu_seqlens,
    chunk_indices,
    scale,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_v, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        i_tg = i_t
        i_n, i_t = (
            tl.load(chunk_indices + i_t * 2).to(tl.int32),
            tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32),
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

    m_s = tl.arange(0, BT)[:, None] >= tl.arange(0, BT)[None, :]

    b_o = tl.zeros([BT, BV], dtype=tl.float32)
    for i_k in range(tl.cdiv(K, BK)):
        p_q = tl.make_block_ptr(
            q + (bos * H + i_h) * K,
            (T, K),
            (H * K, 1),
            (i_t * BT, i_k * BK),
            (BT, BK),
            (1, 0),
        )
        p_g = tl.make_block_ptr(
            g + (bos * H + i_h) * K,
            (T, K),
            (H * K, 1),
            (i_t * BT, i_k * BK),
            (BT, BK),
            (1, 0),
        )
        p_h = tl.make_block_ptr(
            h + (i_tg * H + i_h) * K * V,
            (V, K),
            (K, 1),
            (i_v * BV, i_k * BK),
            (BV, BK),
            (1, 0),
        )

        # [BT, BK]
        b_q = tl.load(p_q, boundary_check=(0, 1))
        b_q = (b_q * scale).to(b_q.dtype)
        # [BT, BK]
        b_g = tl.load(p_g, boundary_check=(0, 1))
        # [BT, BK]
        b_qg = (b_q * exp2(b_g)).to(b_q.dtype)
        # [BV, BK]
        b_h = tl.load(p_h, boundary_check=(0, 1))
        # [BT, BV]
        if i_k >= 0:
            b_o += tl.dot(b_qg, tl.trans(b_h).to(b_qg.dtype))
    p_v = tl.make_block_ptr(
        v + (bos * H + i_h) * V,
        (T, V),
        (H * V, 1),
        (i_t * BT, i_v * BV),
        (BT, BV),
        (1, 0),
    )
    p_o = tl.make_block_ptr(
        o + (bos * H + i_h) * V,
        (T, V),
        (H * V, 1),
        (i_t * BT, i_v * BV),
        (BT, BV),
        (1, 0),
    )
    p_A = tl.make_block_ptr(
        A + (bos * H + i_h) * BT, (T, BT), (H * BT, 1), (i_t * BT, 0), (BT, BT), (1, 0)
    )
    # [BT, BV]
    b_v = tl.load(p_v, boundary_check=(0, 1))
    # [BT, BT]
    b_A = tl.load(p_A, boundary_check=(0, 1))
    b_A = tl.where(m_s, b_A, 0.0).to(b_v.dtype)
    b_o += tl.dot(b_A, b_v, allow_tf32=False)
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))


class KimiK3ChunkGlaFwdOKernel(
    VllmTritonJitKernel["KimiK3ChunkGlaFwdOKernel.CompileKey"]
):
    """JIT owner for the autotuned chunked-prefill output kernel."""

    # Triton's Autotuner owns BK/BV/num_warps/num_stages and compile-warms
    # every candidate for each logical key below.
    kernel = staticmethod(chunk_gla_fwd_kernel_o)

    @dataclass(frozen=True)
    class CompileKey:
        q_dtype: torch.dtype
        v_dtype: torch.dtype
        g_dtype: torch.dtype
        h_dtype: torch.dtype
        out_dtype: torch.dtype
        a_dtype: torch.dtype
        num_heads: int
        qk_head_dim: int
        v_head_dim: int
        block_t: int
        is_varlen: bool

    def dispatch(  # type: ignore[override]
        self,
        *,
        q_dtype: torch.dtype,
        v_dtype: torch.dtype,
        g_dtype: torch.dtype,
        h_dtype: torch.dtype,
        out_dtype: torch.dtype,
        a_dtype: torch.dtype,
        num_heads: int,
        qk_head_dim: int,
        v_head_dim: int,
        block_t: int = FLA_CHUNK_SIZE,
        is_varlen: bool = True,
    ) -> CompileKey:
        return self.CompileKey(
            q_dtype=q_dtype,
            v_dtype=v_dtype,
            g_dtype=g_dtype,
            h_dtype=h_dtype,
            out_dtype=out_dtype,
            a_dtype=a_dtype,
            num_heads=num_heads,
            qk_head_dim=qk_head_dim,
            v_head_dim=v_head_dim,
            block_t=block_t,
            is_varlen=is_varlen,
        )

    def get_warmup_keys(
        self,
        *,
        q_dtype: torch.dtype,
        v_dtype: torch.dtype,
        g_dtype: torch.dtype,
        h_dtype: torch.dtype,
        out_dtype: torch.dtype,
        a_dtype: torch.dtype,
        num_heads: int,
        qk_head_dim: int,
        v_head_dim: int,
        block_t: int = FLA_CHUNK_SIZE,
        is_varlen: bool = True,
    ) -> list[CompileKey]:
        return self._trace_dispatch(self.dispatch)(
            q_dtype=q_dtype,
            v_dtype=v_dtype,
            g_dtype=g_dtype,
            h_dtype=h_dtype,
            out_dtype=out_dtype,
            a_dtype=a_dtype,
            num_heads=num_heads,
            qk_head_dim=qk_head_dim,
            v_head_dim=v_head_dim,
            block_t=block_t,
            is_varlen=is_varlen,
        )

    def warmup_inputs(self, compile_key: CompileKey) -> dict[str, Any]:
        b = 1
        t = compile_key.block_t
        h = compile_key.num_heads
        k = compile_key.qk_head_dim
        v = compile_key.v_head_dim
        nt = 1
        return {
            "q": TritonWarmupTensor(compile_key.q_dtype, shape=(b, t, h, k)),
            "v": TritonWarmupTensor(compile_key.v_dtype, shape=(b, t, h, v)),
            "g": TritonWarmupTensor(compile_key.g_dtype, shape=(b, t, h, k)),
            "h": TritonWarmupTensor(
                compile_key.h_dtype,
                shape=(b, nt, h, v, k),
            ),
            "o": TritonWarmupTensor(compile_key.out_dtype, shape=(b, t, h, v)),
            "A": TritonWarmupTensor(
                compile_key.a_dtype,
                shape=(b, t, h, compile_key.block_t),
            ),
            "cu_seqlens": (
                TritonWarmupTensor(torch.int32, shape=(b + 1,))
                if compile_key.is_varlen
                else None
            ),
            "chunk_indices": (
                TritonWarmupTensor(torch.int32, shape=(nt, 2))
                if compile_key.is_varlen
                else None
            ),
            "scale": 1.0,
            "chunk_size": compile_key.block_t,
        }

    @kernel_launcher
    def __call__(
        self,
        q: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        A: torch.Tensor,
        h: torch.Tensor,
        o: torch.Tensor,
        scale: float,
        cu_seqlens: torch.Tensor | None = None,
        chunk_indices: torch.Tensor | None = None,
        chunk_size: int = FLA_CHUNK_SIZE,
    ) -> LaunchSpec:
        b, t, num_heads, qk_head_dim = q.shape
        v_head_dim = v.shape[-1]
        nt = cdiv(t, chunk_size)
        if chunk_indices is not None:
            nt = chunk_indices.shape[0]

        def grid(meta: dict[str, Any]) -> tuple[int, int, int]:
            return (cdiv(v_head_dim, meta["BV"]), nt, b * num_heads)

        return (
            grid,
            {
                "T": t,
                "H": num_heads,
                "K": qk_head_dim,
                "V": v_head_dim,
                "BT": chunk_size,
            },
            o,
        )


def chunk_gla_fwd_o_gk(
    q: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    A: torch.Tensor,
    h: torch.Tensor,
    o: torch.Tensor,
    scale: float,
    cu_seqlens: torch.Tensor | None = None,
    chunk_indices: torch.Tensor | None = None,
    chunk_size: int = FLA_CHUNK_SIZE,
):
    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size)

    return _CHUNK_GLA_FWD_O_KERNEL(
        q,
        v,
        g,
        A,
        h,
        o,
        scale,
        cu_seqlens,
        chunk_indices,
        chunk_size,
    )


@triton.heuristics(
    {
        "HAS_BIAS": lambda args: args["g_bias"] is not None,
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
    }
)
@triton.autotune(
    configs=[
        triton.Config({"BS": BS}, num_warps=num_warps)
        for BS in [32, 64]
        for num_warps in [2, 4, 8]
    ],
    key=["H", "S", "BT", "IS_VARLEN"],
)
@triton.jit(do_not_specialize=["T"])
def kda_gate_chunk_cumsum_vector_kernel(
    s,
    raw_beta,
    A_log,
    g_bias,
    o,
    beta_out,
    cu_seqlens,
    chunk_indices,
    cumsum_scale,
    lower_bound,
    beta,
    threshold,
    T,
    stride_beta_batch,
    stride_beta_token,
    stride_beta_head,
    H: tl.constexpr,
    S: tl.constexpr,
    BT: tl.constexpr,
    BS: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    USE_LOWER_BOUND: tl.constexpr,
):
    i_s, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        i_n, i_t = (
            tl.load(chunk_indices + i_t * 2).to(tl.int32),
            tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32),
        )
        bos, eos = (
            tl.load(cu_seqlens + i_n).to(tl.int32),
            tl.load(cu_seqlens + i_n + 1).to(tl.int32),
        )
        T = eos - bos
    else:
        bos = i_b * T

    if i_s == 0:
        o_beta_t = tl.arange(0, BT)
        m_beta = i_t * BT + o_beta_t < T
        if IS_VARLEN:
            p_beta = (
                raw_beta
                + (bos + i_t * BT + o_beta_t) * stride_beta_token
                + i_h * stride_beta_head
            )
        else:
            p_beta = (
                raw_beta
                + i_b * stride_beta_batch
                + (i_t * BT + o_beta_t) * stride_beta_token
                + i_h * stride_beta_head
            )
        b_beta = tl.load(p_beta, mask=m_beta, other=0.0).to(tl.float32)
        p_beta_out = beta_out + (bos + i_t * BT + o_beta_t) * H + i_h
        tl.store(p_beta_out, tl.sigmoid(b_beta), mask=m_beta)
        return

    i_s -= 1

    p_s = tl.make_block_ptr(
        s + (bos * H + i_h) * S,
        (T, S),
        (H * S, 1),
        (i_t * BT, i_s * BS),
        (BT, BS),
        (1, 0),
    )
    p_o = tl.make_block_ptr(
        o + (bos * H + i_h) * S,
        (T, S),
        (H * S, 1),
        (i_t * BT, i_s * BS),
        (BT, BS),
        (1, 0),
    )

    b_s = tl.load(p_s, boundary_check=(0, 1)).to(tl.float32)
    if HAS_BIAS:
        p_bias = tl.make_block_ptr(
            g_bias + i_h * S,
            (S,),
            (1,),
            (i_s * BS,),
            (BS,),
            (0,),
        )
        b_bias = tl.load(p_bias, boundary_check=(0,)).to(tl.float32)
        b_s += b_bias[None, :]

    b_a = tl.exp(tl.load(A_log + i_h).to(tl.float32))
    if USE_LOWER_BOUND:
        b_gate = lower_bound * tl.sigmoid(b_a * b_s)
    else:
        b_g_scaled = b_s * beta
        b_softplus = tl.where(
            b_g_scaled > threshold,
            b_s,
            (1.0 / beta) * log(1.0 + tl.exp(b_g_scaled)),
        )
        b_gate = -b_a * b_softplus

    # Boundary loads return zero, but bias and gate activation can make padded
    # rows nonzero. Padding trails valid rows, so it only affects masked stores.
    b_o = tl.cumsum(b_gate, axis=0) * cumsum_scale
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))


class KimiK3GateChunkCumsumKernel(
    VllmTritonJitKernel["KimiK3GateChunkCumsumKernel.CompileKey"]
):
    """JIT owner for the autotuned fused gate/beta chunk-cumsum kernel."""

    # Triton's Autotuner owns BS/num_warps and compile-warms every candidate
    # for each logical key below. Only the compile is pre-warmed; the autotune
    # decision itself runs at the first real launch / profile_run.
    kernel = staticmethod(kda_gate_chunk_cumsum_vector_kernel)

    @dataclass(frozen=True)
    class CompileKey:
        s_dtype: torch.dtype
        raw_beta_dtype: torch.dtype
        a_log_dtype: torch.dtype
        g_bias_dtype: torch.dtype
        o_dtype: torch.dtype
        beta_out_dtype: torch.dtype
        num_heads: int
        gate_dim: int
        block_t: int
        has_bias: bool
        is_varlen: bool
        use_lower_bound: bool

    def dispatch(  # type: ignore[override]
        self,
        *,
        s_dtype: torch.dtype,
        raw_beta_dtype: torch.dtype,
        a_log_dtype: torch.dtype,
        g_bias_dtype: torch.dtype,
        o_dtype: torch.dtype,
        beta_out_dtype: torch.dtype,
        num_heads: int,
        gate_dim: int,
        block_t: int = FLA_CHUNK_SIZE,
        has_bias: bool = True,
        is_varlen: bool = True,
        use_lower_bound: bool = False,
    ) -> CompileKey:
        return self.CompileKey(
            s_dtype=s_dtype,
            raw_beta_dtype=raw_beta_dtype,
            a_log_dtype=a_log_dtype,
            g_bias_dtype=g_bias_dtype,
            o_dtype=o_dtype,
            beta_out_dtype=beta_out_dtype,
            num_heads=num_heads,
            gate_dim=gate_dim,
            block_t=block_t,
            has_bias=has_bias,
            is_varlen=is_varlen,
            use_lower_bound=use_lower_bound,
        )

    def get_warmup_keys(
        self,
        *,
        s_dtype: torch.dtype,
        raw_beta_dtype: torch.dtype,
        a_log_dtype: torch.dtype,
        g_bias_dtype: torch.dtype,
        o_dtype: torch.dtype,
        beta_out_dtype: torch.dtype,
        num_heads: int,
        gate_dim: int,
        block_t: int = FLA_CHUNK_SIZE,
        has_bias: bool = True,
        is_varlen: bool = True,
        use_lower_bound: bool = False,
    ) -> list[CompileKey]:
        return self._trace_dispatch(self.dispatch)(
            s_dtype=s_dtype,
            raw_beta_dtype=raw_beta_dtype,
            a_log_dtype=a_log_dtype,
            g_bias_dtype=g_bias_dtype,
            o_dtype=o_dtype,
            beta_out_dtype=beta_out_dtype,
            num_heads=num_heads,
            gate_dim=gate_dim,
            block_t=block_t,
            has_bias=has_bias,
            is_varlen=is_varlen,
            use_lower_bound=use_lower_bound,
        )

    def warmup_inputs(self, compile_key: CompileKey) -> dict[str, Any]:
        b = 1
        t = compile_key.block_t
        h = compile_key.num_heads
        d = compile_key.gate_dim
        nt = 1
        return {
            "s": TritonWarmupTensor(compile_key.s_dtype, shape=(b, t, h, d)),
            "raw_beta": TritonWarmupTensor(
                compile_key.raw_beta_dtype, shape=(b, t, h)
            ),
            "A_log": TritonWarmupTensor(compile_key.a_log_dtype, shape=(h,)),
            "g_bias": (
                TritonWarmupTensor(compile_key.g_bias_dtype, shape=(h * d,))
                if compile_key.has_bias
                else None
            ),
            "o": TritonWarmupTensor(compile_key.o_dtype, shape=(b, t, h, d)),
            "beta_out": TritonWarmupTensor(
                compile_key.beta_out_dtype, shape=(b, t, h)
            ),
            "cu_seqlens": (
                TritonWarmupTensor(torch.int32, shape=(b + 1,))
                if compile_key.is_varlen
                else None
            ),
            "chunk_indices": (
                TritonWarmupTensor(torch.int32, shape=(nt, 2))
                if compile_key.is_varlen
                else None
            ),
            "cumsum_scale": RCP_LN2,
            "lower_bound": 0.0,
            "beta": 1.0,
            "threshold": 20.0,
            "chunk_size": compile_key.block_t,
            "use_lower_bound": compile_key.use_lower_bound,
        }

    @kernel_launcher
    def __call__(
        self,
        s: torch.Tensor,
        raw_beta: torch.Tensor,
        A_log: torch.Tensor,
        g_bias: torch.Tensor | None,
        o: torch.Tensor,
        beta_out: torch.Tensor,
        cu_seqlens: torch.Tensor | None = None,
        chunk_indices: torch.Tensor | None = None,
        cumsum_scale: float = RCP_LN2,
        lower_bound: float = 0.0,
        beta: float = 1.0,
        threshold: float = 20.0,
        chunk_size: int = FLA_CHUNK_SIZE,
        use_lower_bound: bool = False,
    ) -> LaunchSpec:
        b, t, num_heads, gate_dim = s.shape
        nt = cdiv(t, chunk_size)
        if chunk_indices is not None:
            nt = chunk_indices.shape[0]
        stride_beta_batch = raw_beta.stride(0)
        stride_beta_token = raw_beta.stride(1)
        stride_beta_head = raw_beta.stride(2)

        def grid(meta: dict[str, Any]) -> tuple[int, int, int]:
            # program 0 (per chunk/head) computes beta; the rest cover the gate.
            return (cdiv(gate_dim, meta["BS"]) + 1, nt, b * num_heads)

        return (
            grid,
            {
                "T": t,
                "stride_beta_batch": stride_beta_batch,
                "stride_beta_token": stride_beta_token,
                "stride_beta_head": stride_beta_head,
                "H": num_heads,
                "S": gate_dim,
                "BT": chunk_size,
                "USE_LOWER_BOUND": use_lower_bound,
            },
        )


def fused_kda_gate_chunk_cumsum(
    raw_g: torch.Tensor,
    raw_beta: torch.Tensor,
    A_log: torch.Tensor,
    g_bias: torch.Tensor | None = None,
    beta: float = 1.0,
    threshold: float = 20.0,
    lower_bound: float | None = None,
    cu_seqlens: torch.Tensor | None = None,
    chunk_indices: torch.Tensor | None = None,
    chunk_size: int = FLA_CHUNK_SIZE,
    output_dtype: torch.dtype | None = torch.float,
) -> tuple[torch.Tensor, torch.Tensor]:
    if cu_seqlens is not None:
        assert raw_g.shape[0] == 1, (
            "Only batch size 1 is supported when cu_seqlens are provided"
        )
    B, T, H, D = raw_g.shape
    if raw_beta.shape != (B, T, H):
        raise ValueError(
            f"Expected raw_beta shape {(B, T, H)}, got {raw_beta.shape}"
        )
    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size)

    A_log = A_log.reshape(-1)
    if g_bias is not None:
        g_bias = g_bias.reshape(-1)
    y = torch.empty_like(raw_g, dtype=output_dtype or raw_g.dtype)
    beta_out = torch.empty(raw_beta.shape, device=raw_beta.device, dtype=torch.float32)

    _GATE_CHUNK_CUMSUM_KERNEL(
        s=raw_g,
        raw_beta=raw_beta,
        A_log=A_log,
        g_bias=g_bias,
        o=y,
        beta_out=beta_out,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        # RCP_LN2 folds in the natural-log -> log2 conversion so downstream
        # exp2-based kernels reproduce exp(g). Keep this in sync with the
        # `use_exp2=True` path in `_chunk_kda_fwd_with_cumulative_g`.
        cumsum_scale=RCP_LN2,
        lower_bound=lower_bound or 0.0,
        beta=beta,
        threshold=threshold,
        chunk_size=chunk_size,
        use_lower_bound=lower_bound is not None,
    )
    return y, beta_out


def _chunk_kda_fwd_with_cumulative_g(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor,
    output_final_state: bool,
    cu_seqlens: torch.Tensor | None = None,
    chunk_indices: torch.Tensor | None = None,
    chunk_size: int = FLA_CHUNK_SIZE,
    safe_gate: bool = False,
):
    # `g` must already be chunk-local cumulatively-summed AND scaled by
    # RCP_LN2 (so the downstream exp2-based kernels reproduce exp(g)).
    # Use `chunk_kda_fwd` or `chunk_kda_with_fused_gate_fwd` instead of
    # calling this helper directly unless that invariant is upheld.
    Aqk, A = chunk_kda_fwd_intra(
        q=q,
        k=k,
        gk=g,
        beta=beta,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        chunk_size=chunk_size,
        safe_gate=safe_gate,
    )
    w, u, _, kg = recompute_w_u_fwd(
        k=k,
        v=v,
        beta=beta,
        A=A,
        gk=g,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
    )
    del A
    h, v_new, final_state = chunk_gated_delta_rule_fwd_h(
        k=kg,
        w=w,
        u=u,
        gk=g,
        initial_state=initial_state,
        output_final_state=output_final_state,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        use_exp2=True,
    )
    del w, u, kg
    o = chunk_gla_fwd_o_gk(
        q=q,
        v=v_new,
        g=g,
        A=Aqk,
        h=h,
        o=v,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        chunk_size=chunk_size,
    )
    del Aqk, v_new, h
    return o, final_state


def chunk_kda_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor,
    output_final_state: bool,
    cu_seqlens: torch.Tensor | None = None,
):
    chunk_size = FLA_CHUNK_SIZE
    chunk_indices = (
        prepare_chunk_indices(cu_seqlens, chunk_size)
        if cu_seqlens is not None
        else None
    )
    g = chunk_local_cumsum(
        g,
        chunk_size=chunk_size,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
    )
    # KDA evaluates cumulative gate decays with exp2. Convert from natural-log
    # space so exp(x) is preserved as exp2(x / ln(2)).
    g = g * RCP_LN2
    return _chunk_kda_fwd_with_cumulative_g(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        scale=scale,
        initial_state=initial_state,
        output_final_state=output_final_state,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        chunk_size=chunk_size,
    )


def chunk_kda_with_fused_gate_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    raw_g: torch.Tensor,
    raw_beta: torch.Tensor,
    A_log: torch.Tensor,
    g_bias: torch.Tensor | None,
    scale: float,
    initial_state: torch.Tensor,
    output_final_state: bool,
    lower_bound: float | None = None,
    cu_seqlens: torch.Tensor | None = None,
):
    chunk_size = FLA_CHUNK_SIZE
    chunk_indices = (
        prepare_chunk_indices(cu_seqlens, chunk_size)
        if cu_seqlens is not None
        else None
    )
    g, beta = fused_kda_gate_chunk_cumsum(
        raw_g,
        raw_beta=raw_beta,
        A_log=A_log,
        g_bias=g_bias,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        chunk_size=chunk_size,
        lower_bound=lower_bound,
    )
    return _chunk_kda_fwd_with_cumulative_g(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        scale=scale,
        initial_state=initial_state,
        output_final_state=output_final_state,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        chunk_size=chunk_size,
        safe_gate=lower_bound is not None,
    )


def chunk_kda(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float = None,
    initial_state: torch.Tensor = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
    cu_seqlens: torch.Tensor | None = None,
    **kwargs,
):
    if scale is None:
        scale = k.shape[-1] ** -0.5

    if use_qk_l2norm_in_kernel:
        q = l2norm_fwd(q.contiguous())
        k = l2norm_fwd(k.contiguous())

    o, final_state = chunk_kda_fwd(
        q=q,
        k=k,
        v=v.contiguous(),
        g=g.contiguous(),
        beta=beta.contiguous(),
        scale=scale,
        initial_state=initial_state.contiguous(),
        output_final_state=output_final_state,
        cu_seqlens=cu_seqlens,
    )
    return o, final_state


def chunk_kda_with_fused_gate(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    raw_g: torch.Tensor,
    raw_beta: torch.Tensor,
    A_log: torch.Tensor,
    g_bias: torch.Tensor | None,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    lower_bound: float | None = None,
    use_qk_l2norm_in_kernel: bool = False,
    cu_seqlens: torch.Tensor | None = None,
    **kwargs,
):
    """Run chunk KDA from raw gate and beta projections."""
    if scale is None:
        scale = k.shape[-1] ** -0.5

    if use_qk_l2norm_in_kernel:
        q = l2norm_fwd(q.contiguous())
        k = l2norm_fwd(k.contiguous())

    o, final_state = chunk_kda_with_fused_gate_fwd(
        q=q,
        k=k,
        v=v.contiguous(),
        raw_g=raw_g.contiguous(),
        raw_beta=raw_beta,
        A_log=A_log,
        g_bias=g_bias,
        scale=scale,
        initial_state=initial_state.contiguous() if initial_state is not None else None,
        output_final_state=output_final_state,
        lower_bound=lower_bound,
        cu_seqlens=cu_seqlens,
    )
    return o, final_state


@triton.autotune(
    configs=[
        triton.Config({"BT": bt}, num_warps=nw, num_stages=ns)
        for bt in BT_LIST_AUTOTUNE
        for nw in NUM_WARPS_AUTOTUNE
        for ns in [2, 3]
    ],
    key=["H", "D"],
)
@triton.jit
def kda_gate_fwd_kernel(
    g,
    A,
    y,
    g_bias,
    lower_bound,
    beta: tl.constexpr,
    threshold: tl.constexpr,
    T,
    H,
    D: tl.constexpr,
    BT: tl.constexpr,
    BD: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    USE_LOWER_BOUND: tl.constexpr,
):
    i_t, i_h = tl.program_id(0), tl.program_id(1)
    n_t = i_t * BT

    b_a = tl.exp(tl.load(A + i_h).to(tl.float32))

    stride_row = H * D
    stride_col = 1

    g_ptr = tl.make_block_ptr(
        base=g + i_h * D,
        shape=(T, D),
        strides=(stride_row, stride_col),
        offsets=(n_t, 0),
        block_shape=(BT, BD),
        order=(1, 0),
    )

    y_ptr = tl.make_block_ptr(
        base=y + i_h * D,
        shape=(T, D),
        strides=(stride_row, stride_col),
        offsets=(n_t, 0),
        block_shape=(BT, BD),
        order=(1, 0),
    )

    b_g = tl.load(g_ptr, boundary_check=(0, 1)).to(tl.float32)

    if HAS_BIAS:
        n_d = tl.arange(0, BD)
        bias_mask = n_d < D
        b_bias = tl.load(g_bias + i_h * D + n_d, mask=bias_mask, other=0.0).to(
            tl.float32
        )
        b_g = b_g + b_bias[None, :]

    if USE_LOWER_BOUND:
        b_y = lower_bound * tl.sigmoid(b_a * b_g)
    else:
        g_scaled = b_g * beta
        use_linear = g_scaled > threshold
        sp = tl.where(use_linear, b_g, (1.0 / beta) * log(1.0 + tl.exp(g_scaled)))
        b_y = -b_a * sp

    tl.store(y_ptr, b_y.to(y.dtype.element_ty), boundary_check=(0, 1))


def fused_kda_gate(
    g: torch.Tensor,
    A: torch.Tensor,
    head_k_dim: int,
    g_bias: torch.Tensor | None = None,
    beta: float = 1.0,
    threshold: float = 20.0,
    lower_bound: float | None = None,
) -> torch.Tensor:
    """
    Forward pass for KDA gate:
      input g: [..., H*D]
      param A: [H] or [1, 1, H, 1]
      beta: softplus beta parameter
      threshold: softplus threshold parameter
      return  : [..., H, D]
    """
    orig_shape = g.shape[:-1]

    g = g.view(-1, g.shape[-1])
    T = g.shape[0]
    HD = g.shape[1]
    H = A.numel()
    assert H * head_k_dim == HD
    assert g.stride() == (HD, 1)

    y = torch.empty_like(g, dtype=torch.float32)

    def grid(meta):
        return (cdiv(T, meta["BT"]), H)

    kda_gate_fwd_kernel[grid](
        g,
        A,
        y,
        g_bias,
        lower_bound or 0.0,
        beta,
        threshold,
        T,
        H,
        head_k_dim,
        BD=next_power_of_2(head_k_dim),
        HAS_BIAS=g_bias is not None,
        USE_LOWER_BOUND=lower_bound is not None,
    )

    y = y.view(*orig_shape, H, head_k_dim)
    return y


_CHUNK_GLA_FWD_O_KERNEL = KimiK3ChunkGlaFwdOKernel()
_RECOMPUTE_WU_KERNEL = KimiK3RecomputeWUKernel()
_GATE_CHUNK_CUMSUM_KERNEL = KimiK3GateChunkCumsumKernel()
