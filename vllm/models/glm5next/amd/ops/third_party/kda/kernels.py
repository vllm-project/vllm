# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Songlin Yang, Yu Zhang
# mypy: ignore-errors
#
# This file contains code copied from the flash-linear-attention project.
# The original source code was licensed under the MIT license and included
# the following copyright notice:
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang
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
from vllm.third_party.flash_linear_attention.ops.solve_tril import solve_tril
from vllm.third_party.flash_linear_attention.ops.utils import FLA_CHUNK_SIZE, is_amd
from vllm.triton_utils import tl, triton
from vllm.utils.math_utils import RCP_LN2, cdiv, next_power_of_2

from .fused_recurrent import (
    _FUSED_RECURRENT_GATED_DELTA_RULE_FWD_KERNEL,
)

BT_LIST_AUTOTUNE = [32, 64, 128]
NUM_WARPS_AUTOTUNE = [2, 4, 8, 16] if is_amd else [4, 8, 16, 32]


def fused_recurrent_kda_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor,
    inplace_final_state: bool = True,
    cu_seqlens: torch.Tensor | None = None,
    ssm_state_indices: torch.Tensor | None = None,
    num_accepted_tokens: torch.Tensor | None = None,
    use_qk_l2norm_in_kernel: bool = False,
    out: torch.Tensor | None = None,
    sigmoid_beta: bool = False,
    a_log: torch.Tensor | None = None,
    g_bias: torch.Tensor | None = None,
    compute_gate: bool = False,
    lower_bound: float | None = -5.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    _, T, _, K, V = *k.shape, v.shape[-1]
    HV = v.shape[2]

    if compute_gate:
        assert a_log is not None and g_bias is not None, (
            "compute_gate requires a_log and g_bias"
        )
        assert lower_bound is not None, (
            "compute_gate implements the bounded (safe_gate) branch only"
        )
        a_log = a_log.reshape(-1).contiguous()
        g_bias = g_bias.reshape(-1).contiguous()

    if out is None:
        o = torch.empty_like(k)
    else:
        # Caller-provided output buffer; must be layout-compatible with the
        # tensor the kernel indexes (contiguous, same shape/dtype as k).
        assert out.shape == k.shape and out.dtype == k.dtype
        assert out.is_contiguous()
        o = out
    if inplace_final_state:
        final_state = initial_state
    else:
        final_state = q.new_empty(T, HV, V, K, dtype=initial_state.dtype)

    _FUSED_RECURRENT_GATED_DELTA_RULE_FWD_KERNEL(
        q,
        k,
        v,
        g,
        beta,
        o,
        initial_state,
        final_state,
        cu_seqlens,
        ssm_state_indices,
        num_accepted_tokens,
        a_log,
        g_bias,
        scale=scale,
        inplace_final_state=inplace_final_state,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        is_kda=True,
        sigmoid_beta=sigmoid_beta,
        compute_gate=compute_gate,
        lower_bound=lower_bound if lower_bound is not None else -5.0,
    )

    return o, final_state


def fused_recurrent_kda(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor = None,
    scale: float = None,
    initial_state: torch.Tensor = None,
    inplace_final_state: bool = True,
    use_qk_l2norm_in_kernel: bool = True,
    cu_seqlens: torch.Tensor | None = None,
    ssm_state_indices: torch.LongTensor | None = None,
    num_accepted_tokens: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
    sigmoid_beta: bool = False,
    a_log: torch.Tensor | None = None,
    g_bias: torch.Tensor | None = None,
    compute_gate: bool = False,
    lower_bound: float | None = -5.0,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    if cu_seqlens is not None and q.shape[0] != 1:
        raise ValueError(
            f"The batch size is expected to be 1 rather than {q.shape[0]} when using `cu_seqlens`."
            f"Please flatten variable-length inputs before processing."
        )
    if scale is None:
        scale = k.shape[-1] ** -0.5

    o, final_state = fused_recurrent_kda_fwd(
        q=q.contiguous(),
        k=k.contiguous(),
        v=v.contiguous(),
        g=g.contiguous(),
        beta=beta.contiguous(),
        scale=scale,
        initial_state=initial_state,
        inplace_final_state=inplace_final_state,
        cu_seqlens=cu_seqlens,
        ssm_state_indices=ssm_state_indices,
        num_accepted_tokens=num_accepted_tokens,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        out=out,
        sigmoid_beta=sigmoid_beta,
        a_log=a_log,
        g_bias=g_bias,
        compute_gate=compute_gate,
        lower_bound=lower_bound,
    )
    return o, final_state


@triton.heuristics({"IS_VARLEN": lambda args: args["cu_seqlens"] is not None})
@triton.autotune(
    configs=[
        triton.Config({"BK": BK}, num_warps=num_warps, num_stages=num_stages)
        for BK in [32, 64]
        for num_warps in [1, 2, 4, 8]
        for num_stages in [2, 3, 4]
    ],
    key=["BC"],
)
@triton.jit(do_not_specialize=["T"])
def chunk_kda_scaled_dot_kkt_fwd_kernel_intra_sub_inter(
    q,
    k,
    g,
    beta,
    A,
    Aqk,
    scale,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
    BK: tl.constexpr,
    NC: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_t, i_c, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_h = i_bh // H, i_bh % H
    i_i, i_j = i_c // NC, i_c % NC
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

    if i_t * BT + i_i * BC >= T:
        return
    if i_i <= i_j:
        return

    q += (bos * H + i_h) * K
    k += (bos * H + i_h) * K
    g += (bos * H + i_h) * K
    A += (bos * H + i_h) * BT
    Aqk += (bos * H + i_h) * BT

    p_b = tl.make_block_ptr(
        beta + bos * H + i_h, (T,), (H,), (i_t * BT + i_i * BC,), (BC,), (0,)
    )
    b_b = tl.load(p_b, boundary_check=(0,))

    b_A = tl.zeros([BC, BC], dtype=tl.float32)
    b_Aqk = tl.zeros([BC, BC], dtype=tl.float32)
    for i_k in range(tl.cdiv(K, BK)):
        p_q = tl.make_block_ptr(
            q, (T, K), (H * K, 1), (i_t * BT + i_i * BC, i_k * BK), (BC, BK), (1, 0)
        )
        p_k = tl.make_block_ptr(
            k, (T, K), (H * K, 1), (i_t * BT + i_i * BC, i_k * BK), (BC, BK), (1, 0)
        )
        p_g = tl.make_block_ptr(
            g, (T, K), (H * K, 1), (i_t * BT + i_i * BC, i_k * BK), (BC, BK), (1, 0)
        )
        b_kt = tl.make_block_ptr(
            k, (K, T), (1, H * K), (i_k * BK, i_t * BT + i_j * BC), (BK, BC), (0, 1)
        )
        p_gk = tl.make_block_ptr(
            g, (K, T), (1, H * K), (i_k * BK, i_t * BT + i_j * BC), (BK, BC), (0, 1)
        )

        o_k = i_k * BK + tl.arange(0, BK)
        m_k = o_k < K
        # [BK,]
        b_gn = tl.load(g + (i_t * BT + i_i * BC) * H * K + o_k, mask=m_k, other=0)
        # [BC, BK]
        b_g = tl.load(p_g, boundary_check=(0, 1))
        b_k = tl.load(p_k, boundary_check=(0, 1)) * exp2(b_g - b_gn[None, :])
        # [BK, BC]
        b_gk = tl.load(p_gk, boundary_check=(0, 1))
        b_kt = tl.load(b_kt, boundary_check=(0, 1))
        # [BC, BC]
        b_ktg = b_kt * exp2(b_gn[:, None] - b_gk)
        b_A += tl.dot(b_k, b_ktg)

        b_q = tl.load(p_q, boundary_check=(0, 1))
        b_qg = b_q * exp2(b_g - b_gn[None, :]) * scale
        b_Aqk += tl.dot(b_qg, b_ktg)

    b_A *= b_b[:, None]

    p_A = tl.make_block_ptr(
        A, (T, BT), (H * BT, 1), (i_t * BT + i_i * BC, i_j * BC), (BC, BC), (1, 0)
    )
    tl.store(p_A, b_A.to(A.dtype.element_ty), boundary_check=(0, 1))
    p_Aqk = tl.make_block_ptr(
        Aqk, (T, BT), (H * BT, 1), (i_t * BT + i_i * BC, i_j * BC), (BC, BC), (1, 0)
    )
    tl.store(p_Aqk, b_Aqk.to(Aqk.dtype.element_ty), boundary_check=(0, 1))


@triton.heuristics({"IS_VARLEN": lambda args: args["cu_seqlens"] is not None})
@triton.autotune(
    configs=[triton.Config({}, num_warps=num_warps) for num_warps in [1, 2, 4, 8]],
    key=["BK", "BT"],
)
@triton.jit(do_not_specialize=["T"])
def chunk_kda_scaled_dot_kkt_fwd_kernel_intra_sub_intra(
    q,
    k,
    g,
    beta,
    A,
    Aqk,
    scale,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
    BK: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_t, i_i, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
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

    if i_t * BT + i_i * BC >= T:
        return

    o_i = tl.arange(0, BC)
    o_k = tl.arange(0, BK)
    m_k = o_k < K
    m_A = (i_t * BT + i_i * BC + o_i) < T
    o_A = (bos + i_t * BT + i_i * BC + o_i) * H * BT + i_h * BT + i_i * BC

    p_q = tl.make_block_ptr(
        q + (bos * H + i_h) * K,
        (T, K),
        (H * K, 1),
        (i_t * BT + i_i * BC, 0),
        (BC, BK),
        (1, 0),
    )
    p_k = tl.make_block_ptr(
        k + (bos * H + i_h) * K,
        (T, K),
        (H * K, 1),
        (i_t * BT + i_i * BC, 0),
        (BC, BK),
        (1, 0),
    )
    p_g = tl.make_block_ptr(
        g + (bos * H + i_h) * K,
        (T, K),
        (H * K, 1),
        (i_t * BT + i_i * BC, 0),
        (BC, BK),
        (1, 0),
    )
    b_q = tl.load(p_q, boundary_check=(0, 1))
    b_k = tl.load(p_k, boundary_check=(0, 1))
    b_g = tl.load(p_g, boundary_check=(0, 1))

    p_b = beta + (bos + i_t * BT + i_i * BC + o_i) * H + i_h
    b_k = b_k * tl.load(p_b, mask=m_A, other=0)[:, None]

    p_kt = k + (bos + i_t * BT + i_i * BC) * H * K + i_h * K + o_k
    p_gk = g + (bos + i_t * BT + i_i * BC) * H * K + i_h * K + o_k

    for j in range(0, min(BC, T - i_t * BT - i_i * BC)):
        b_kt = tl.load(p_kt, mask=m_k, other=0).to(tl.float32)
        b_gk = tl.load(p_gk, mask=m_k, other=0).to(tl.float32)
        b_ktg = b_kt[None, :] * exp2(b_g - b_gk[None, :])
        b_A = tl.sum(b_k * b_ktg, 1)
        b_A = tl.where(o_i > j, b_A, 0.0)
        b_Aqk = tl.sum(b_q * b_ktg, 1)
        b_Aqk = tl.where(o_i >= j, b_Aqk * scale, 0.0)
        tl.store(A + o_A + j, b_A, mask=m_A)
        tl.store(Aqk + o_A + j, b_Aqk, mask=m_A)
        p_kt += H * K
        p_gk += H * K


class Glm5NextKdaInterChunkKernel(
    VllmTritonJitKernel["Glm5NextKdaInterChunkKernel.CompileKey"]
):
    kernel = staticmethod(chunk_kda_scaled_dot_kkt_fwd_kernel_intra_sub_inter)

    @dataclass(frozen=True)
    class CompileKey:
        q_dtype: torch.dtype
        k_dtype: torch.dtype
        g_dtype: torch.dtype
        beta_dtype: torch.dtype
        a_dtype: torch.dtype
        aqk_dtype: torch.dtype
        num_heads: int
        head_dim: int
        block_t: int
        block_c: int
        num_chunks: int
        is_varlen: bool

    def dispatch(self, **kwargs: Any) -> CompileKey:
        return self.CompileKey(**kwargs)

    def get_warmup_keys(self, **kwargs: Any) -> list[CompileKey]:
        return self._trace_dispatch(self.dispatch)(**kwargs)

    def warmup_inputs(self, compile_key: CompileKey) -> dict[str, Any]:
        b, t, h, k = 1, compile_key.block_t, compile_key.num_heads, compile_key.head_dim
        return {
            "q": TritonWarmupTensor(compile_key.q_dtype, shape=(b, t, h, k)),
            "k": TritonWarmupTensor(compile_key.k_dtype, shape=(b, t, h, k)),
            "g": TritonWarmupTensor(compile_key.g_dtype, shape=(b, t, h, k)),
            "beta": TritonWarmupTensor(compile_key.beta_dtype, shape=(b, t, h)),
            "A": TritonWarmupTensor(compile_key.a_dtype, shape=(b, t, h, t)),
            "Aqk": TritonWarmupTensor(compile_key.aqk_dtype, shape=(b, t, h, t)),
            "scale": 1.0,
            "cu_seqlens": TritonWarmupTensor(torch.int32, shape=(b + 1,)) if compile_key.is_varlen else None,
            "chunk_indices": TritonWarmupTensor(torch.int32, shape=(1, 2)) if compile_key.is_varlen else None,
            "chunk_size": compile_key.block_t,
        }

    @kernel_launcher
    def __call__(self, q: torch.Tensor, k: torch.Tensor, g: torch.Tensor,
                 beta: torch.Tensor, A: torch.Tensor, Aqk: torch.Tensor,
                 scale: float, cu_seqlens: torch.Tensor | None = None,
                 chunk_indices: torch.Tensor | None = None,
                 chunk_size: int = FLA_CHUNK_SIZE) -> LaunchSpec:
        b, t, h, k_dim = k.shape
        block_c = min(16, chunk_size)
        num_chunks = cdiv(chunk_size, block_c)
        nt = cdiv(t, chunk_size) if cu_seqlens is None else len(chunk_indices)
        return (nt, num_chunks * num_chunks, b * h), {
            "T": t, "H": h, "K": k_dim, "BT": chunk_size,
            "BC": block_c, "NC": num_chunks,
        }


class Glm5NextKdaIntraChunkKernel(
    VllmTritonJitKernel["Glm5NextKdaIntraChunkKernel.CompileKey"]
):
    kernel = staticmethod(chunk_kda_scaled_dot_kkt_fwd_kernel_intra_sub_intra)

    @dataclass(frozen=True)
    class CompileKey:
        q_dtype: torch.dtype
        k_dtype: torch.dtype
        g_dtype: torch.dtype
        beta_dtype: torch.dtype
        a_dtype: torch.dtype
        aqk_dtype: torch.dtype
        num_heads: int
        head_dim: int
        block_t: int
        block_c: int
        block_k: int
        is_varlen: bool

    def dispatch(self, **kwargs: Any) -> CompileKey:
        return self.CompileKey(**kwargs)

    def get_warmup_keys(self, **kwargs: Any) -> list[CompileKey]:
        return self._trace_dispatch(self.dispatch)(**kwargs)

    def warmup_inputs(self, compile_key: CompileKey) -> dict[str, Any]:
        b, t, h, k = 1, compile_key.block_t, compile_key.num_heads, compile_key.head_dim
        return {
            "q": TritonWarmupTensor(compile_key.q_dtype, shape=(b, t, h, k)),
            "k": TritonWarmupTensor(compile_key.k_dtype, shape=(b, t, h, k)),
            "g": TritonWarmupTensor(compile_key.g_dtype, shape=(b, t, h, k)),
            "beta": TritonWarmupTensor(compile_key.beta_dtype, shape=(b, t, h)),
            "A": TritonWarmupTensor(compile_key.a_dtype, shape=(b, t, h, t)),
            "Aqk": TritonWarmupTensor(compile_key.aqk_dtype, shape=(b, t, h, t)),
            "scale": 1.0,
            "cu_seqlens": TritonWarmupTensor(torch.int32, shape=(b + 1,)) if compile_key.is_varlen else None,
            "chunk_indices": TritonWarmupTensor(torch.int32, shape=(1, 2)) if compile_key.is_varlen else None,
            "chunk_size": compile_key.block_t,
        }

    @kernel_launcher
    def __call__(self, q: torch.Tensor, k: torch.Tensor, g: torch.Tensor,
                 beta: torch.Tensor, A: torch.Tensor, Aqk: torch.Tensor,
                 scale: float, cu_seqlens: torch.Tensor | None = None,
                 chunk_indices: torch.Tensor | None = None,
                 chunk_size: int = FLA_CHUNK_SIZE) -> LaunchSpec:
        b, t, h, k_dim = k.shape
        block_c = min(16, chunk_size)
        block_k = max(next_power_of_2(k_dim), 16)
        nt = cdiv(t, chunk_size) if cu_seqlens is None else len(chunk_indices)
        return (nt, cdiv(chunk_size, block_c), b * h), {
            "T": t, "H": h, "K": k_dim, "BT": chunk_size,
            "BC": block_c, "BK": block_k,
        }


def chunk_kda_scaled_dot_kkt_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    gk: torch.Tensor | None = None,
    beta: torch.Tensor | None = None,
    scale: float | None = None,
    cu_seqlens: torch.Tensor | None = None,
    chunk_indices: torch.Tensor | None = None,
    chunk_size: int = FLA_CHUNK_SIZE,
    output_dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""
    Compute beta * K * K^T.

    Args:
        k (torch.Tensor):
            The key tensor of shape `[B, T, H, K]`.
        beta (torch.Tensor):
            The beta tensor of shape `[B, T, H]`.
        gk (torch.Tensor):
            The cumulative sum of the gate tensor of shape `[B, T, H, K]` applied to the key tensor. Default: `None`.
        cu_seqlens (torch.Tensor):
            The cumulative sequence lengths of the input tensor.
            Default: None
        chunk_size (int):
            The chunk size. Default: 64.
        output_dtype (torch.dtype):
            The dtype of the output tensor. Default: `torch.float32`

    Returns:
        beta * K * K^T of shape `[B, T, H, BT]` where `BT` is the chunk size.
    """
    B, T, H, K = k.shape
    assert K <= 256
    BT = chunk_size
    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
    NT = cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)

    BC = min(16, BT)
    NC = cdiv(BT, BC)
    BK = max(next_power_of_2(K), 16)
    A = torch.zeros(B, T, H, BT, device=k.device, dtype=output_dtype)
    Aqk = torch.zeros(B, T, H, BT, device=k.device, dtype=output_dtype)
    _KDA_INTER_CHUNK_KERNEL(
        q, k, gk, beta, A, Aqk, scale, cu_seqlens, chunk_indices, chunk_size
    )
    _KDA_INTRA_CHUNK_KERNEL(
        q, k, gk, beta, A, Aqk, scale, cu_seqlens, chunk_indices, chunk_size
    )
    return A, Aqk


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
    b_b = tl.load(p_b, boundary_check=(0,))

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


class Glm5NextRecomputeWUKernel(
    VllmTritonJitKernel["Glm5NextRecomputeWUKernel.CompileKey"]
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


class Glm5NextChunkGlaFwdOKernel(
    VllmTritonJitKernel["Glm5NextChunkGlaFwdOKernel.CompileKey"]
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
        triton.Config({"BD": BD}, num_warps=num_warps)
        for BD in [32, 64]
        for num_warps in [2, 4, 8]
    ],
    key=["H", "D", "BT", "IS_VARLEN"],
)
@triton.jit(do_not_specialize=["T"])
def kda_gate_cumsum_fwd_kernel(
    g,
    A,
    y,
    g_bias,
    cu_seqlens,
    chunk_indices,
    cumsum_scale,
    beta,
    threshold,
    SAFE_GATE: tl.constexpr,
    LOWER_BOUND: tl.constexpr,
    T,
    H: tl.constexpr,
    D: tl.constexpr,
    BT: tl.constexpr,
    BD: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_d, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
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

    p_g = tl.make_block_ptr(
        g + (bos * H + i_h) * D,
        (T, D),
        (H * D, 1),
        (i_t * BT, i_d * BD),
        (BT, BD),
        (1, 0),
    )
    p_y = tl.make_block_ptr(
        y + (bos * H + i_h) * D,
        (T, D),
        (H * D, 1),
        (i_t * BT, i_d * BD),
        (BT, BD),
        (1, 0),
    )

    b_g = tl.load(p_g, boundary_check=(0, 1)).to(tl.float32)
    if HAS_BIAS:
        o_d = i_d * BD + tl.arange(0, BD)
        b_bias = tl.load(g_bias + i_h * D + o_d, mask=o_d < D, other=0.0).to(tl.float32)
        b_g = b_g + b_bias[None, :]

    b_a = tl.load(A + i_h).to(tl.float32)
    b_a = tl.exp(b_a) if SAFE_GATE else -tl.exp(b_a)
    if SAFE_GATE:
        # y = lower_bound * sigmoid(exp(A) * (g + g_bias)), bounded to
        # (lower_bound, 0) for safe-gate checkpoints.
        b_gate = LOWER_BOUND / (1.0 + tl.exp(-(b_a * b_g)))
    else:
        b_g_scaled = b_g * beta
        b_softplus = tl.where(
            b_g_scaled > threshold,
            b_g,
            (1.0 / beta) * log(1.0 + tl.exp(b_g_scaled)),
        )
        b_gate = b_a * b_softplus

    # Out-of-bounds rows (load returns 0, but softplus/bias can still make
    # b_gate non-zero) participate in the dot product. They only contribute to
    # out-of-bounds output rows, which are masked away by `boundary_check` on
    # the store, so visible output matches unfused gate + chunk-local cumsum.
    o_t = tl.arange(0, BT)
    m_cumsum = tl.where(o_t[:, None] >= o_t[None, :], 1.0, 0.0)
    b_y = tl.dot(m_cumsum, b_gate, allow_tf32=False) * cumsum_scale
    tl.store(p_y, b_y.to(p_y.dtype.element_ty), boundary_check=(0, 1))


class Glm5NextKdaGateCumsumKernel(
    VllmTritonJitKernel["Glm5NextKdaGateCumsumKernel.CompileKey"]
):
    kernel = staticmethod(kda_gate_cumsum_fwd_kernel)

    @dataclass(frozen=True)
    class CompileKey:
        g_dtype: torch.dtype
        a_dtype: torch.dtype
        y_dtype: torch.dtype
        bias_dtype: torch.dtype
        num_heads: int
        gate_dim: int
        block_t: int
        safe_gate: bool
        lower_bound: float
        has_bias: bool
        is_varlen: bool

    def dispatch(
        self,
        *,
        g_dtype: torch.dtype,
        a_dtype: torch.dtype,
        y_dtype: torch.dtype,
        bias_dtype: torch.dtype,
        num_heads: int,
        gate_dim: int,
        block_t: int,
        safe_gate: bool,
        lower_bound: float,
        has_bias: bool,
        is_varlen: bool,
    ) -> CompileKey:
        return self.CompileKey(
            g_dtype=g_dtype,
            a_dtype=a_dtype,
            y_dtype=y_dtype,
            bias_dtype=bias_dtype,
            num_heads=num_heads,
            gate_dim=gate_dim,
            block_t=block_t,
            safe_gate=safe_gate,
            lower_bound=lower_bound,
            has_bias=has_bias,
            is_varlen=is_varlen,
        )

    def get_warmup_keys(self, **kwargs: Any) -> list[CompileKey]:
        return self._trace_dispatch(self.dispatch)(**kwargs)

    def warmup_inputs(self, compile_key: CompileKey) -> dict[str, Any]:
        b = 1
        t = compile_key.block_t
        h = compile_key.num_heads
        d = compile_key.gate_dim
        return {
            "raw_g": TritonWarmupTensor(compile_key.g_dtype, shape=(b, t, h, d)),
            "A_log": TritonWarmupTensor(compile_key.a_dtype, shape=(h,)),
            "g_bias": (
                TritonWarmupTensor(compile_key.bias_dtype, shape=(h * d,))
                if compile_key.has_bias
                else None
            ),
            "y": TritonWarmupTensor(compile_key.y_dtype, shape=(b, t, h, d)),
            "cu_seqlens": (
                TritonWarmupTensor(torch.int32, shape=(b + 1,))
                if compile_key.is_varlen
                else None
            ),
            "chunk_indices": (
                TritonWarmupTensor(torch.int32, shape=(1, 2))
                if compile_key.is_varlen
                else None
            ),
            "beta": 1.0,
            "threshold": 20.0,
            "chunk_size": compile_key.block_t,
            "safe_gate": compile_key.safe_gate,
            "lower_bound": compile_key.lower_bound,
        }

    @kernel_launcher
    def __call__(
        self,
        raw_g: torch.Tensor,
        A_log: torch.Tensor,
        y: torch.Tensor,
        g_bias: torch.Tensor | None = None,
        beta: float = 1.0,
        threshold: float = 20.0,
        cu_seqlens: torch.Tensor | None = None,
        chunk_indices: torch.Tensor | None = None,
        chunk_size: int = FLA_CHUNK_SIZE,
        safe_gate: bool = False,
        lower_bound: float = -5.0,
    ) -> LaunchSpec:
        b, t, h, d = raw_g.shape
        nt = cdiv(t, chunk_size) if cu_seqlens is None else len(chunk_indices)

        def grid(meta: dict[str, Any]) -> tuple[int, int, int]:
            return (cdiv(meta["D"], meta["BD"]), nt, b * h)

        return grid, {
            "cumsum_scale": RCP_LN2,
            "SAFE_GATE": safe_gate,
            "LOWER_BOUND": lower_bound,
            "T": t,
            "H": h,
            "D": d,
            "BT": chunk_size,
        }


def fused_kda_gate_chunk_cumsum(
    raw_g: torch.Tensor,
    A_log: torch.Tensor,
    g_bias: torch.Tensor | None = None,
    beta: float = 1.0,
    threshold: float = 20.0,
    cu_seqlens: torch.Tensor | None = None,
    chunk_indices: torch.Tensor | None = None,
    chunk_size: int = FLA_CHUNK_SIZE,
    output_dtype: torch.dtype | None = torch.float,
    safe_gate: bool = False,
    lower_bound: float = -5.0,
) -> torch.Tensor:
    if cu_seqlens is not None:
        assert raw_g.shape[0] == 1, (
            "Only batch size 1 is supported when cu_seqlens are provided"
        )
    _, _, _, _ = raw_g.shape
    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size)
    A_log = A_log.reshape(-1)
    if g_bias is not None:
        g_bias = g_bias.reshape(-1)
    y = torch.empty_like(raw_g, dtype=output_dtype or raw_g.dtype)
    _KDA_GATE_CUMSUM_KERNEL(
        raw_g,
        A_log,
        y,
        g_bias,
        beta,
        threshold,
        cu_seqlens,
        chunk_indices,
        chunk_size,
        safe_gate,
        lower_bound,
    )
    return y


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
):
    # `g` must already be chunk-local cumulatively-summed AND scaled by
    # RCP_LN2 (so the downstream exp2-based kernels reproduce exp(g)).
    # Use `chunk_kda_fwd` or `chunk_kda_with_fused_gate_fwd` instead of
    # calling this helper directly unless that invariant is upheld.
    # the intra Aqk is kept in fp32
    # the computation has very marginal effect on the entire throughput
    A, Aqk = chunk_kda_scaled_dot_kkt_fwd(
        q=q,
        k=k,
        gk=g,
        beta=beta,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        chunk_size=chunk_size,
        output_dtype=torch.float32,
    )
    A = solve_tril(A=A, cu_seqlens=cu_seqlens, output_dtype=k.dtype)
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
    beta: torch.Tensor,
    A_log: torch.Tensor,
    g_bias: torch.Tensor | None,
    scale: float,
    initial_state: torch.Tensor,
    output_final_state: bool,
    cu_seqlens: torch.Tensor | None = None,
    safe_gate: bool = False,
    lower_bound: float = -5.0,
):
    chunk_size = FLA_CHUNK_SIZE
    chunk_indices = (
        prepare_chunk_indices(cu_seqlens, chunk_size)
        if cu_seqlens is not None
        else None
    )
    g = fused_kda_gate_chunk_cumsum(
        raw_g,
        A_log=A_log,
        g_bias=g_bias,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        chunk_size=chunk_size,
        safe_gate=safe_gate,
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
    beta: torch.Tensor,
    A_log: torch.Tensor,
    g_bias: torch.Tensor | None,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
    cu_seqlens: torch.Tensor | None = None,
    safe_gate: bool = False,
    lower_bound: float = -5.0,
    **kwargs,
):
    """Run chunk KDA from raw gate projection using fused gate+cumsum."""
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
        beta=beta.contiguous(),
        A_log=A_log,
        g_bias=g_bias,
        scale=scale,
        initial_state=initial_state.contiguous() if initial_state is not None else None,
        output_final_state=output_final_state,
        cu_seqlens=cu_seqlens,
        safe_gate=safe_gate,
        lower_bound=lower_bound,
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
    beta: tl.constexpr,
    threshold: tl.constexpr,
    SAFE_GATE: tl.constexpr,
    LOWER_BOUND: tl.constexpr,
    T,
    H,
    D: tl.constexpr,
    BT: tl.constexpr,
    BD: tl.constexpr,
    HAS_BIAS: tl.constexpr,
):
    i_t, i_h = tl.program_id(0), tl.program_id(1)
    n_t = i_t * BT

    b_a = tl.load(A + i_h).to(tl.float32)
    b_a = tl.exp(b_a) if SAFE_GATE else -tl.exp(b_a)

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

    if SAFE_GATE:
        # y = lower_bound * sigmoid(exp(A) * (g + g_bias)), bounded to
        # (lower_bound, 0) for safe-gate checkpoints.
        b_y = LOWER_BOUND / (1.0 + tl.exp(-(b_a * b_g)))
    else:
        # softplus(x, beta) = (1/beta) * log(1 + exp(beta * x))
        # When beta * x > threshold, use linear approximation x
        # Use threshold to switch to linear when beta*x > threshold
        g_scaled = b_g * beta
        use_linear = g_scaled > threshold
        sp = tl.where(use_linear, b_g, (1.0 / beta) * log(1.0 + tl.exp(g_scaled)))
        b_y = b_a * sp

    tl.store(y_ptr, b_y.to(y.dtype.element_ty), boundary_check=(0, 1))


def fused_kda_gate(
    g: torch.Tensor,
    A: torch.Tensor,
    head_k_dim: int,
    g_bias: torch.Tensor | None = None,
    beta: float = 1.0,
    threshold: float = 20.0,
    safe_gate: bool = False,
    lower_bound: float | None = -5.0,
) -> torch.Tensor:
    """
    Forward pass for KDA gate:
      input g: [..., H*D]
      param A: [H] or [1, 1, H, 1]
      beta: softplus beta parameter (softplus branch only)
      threshold: softplus threshold parameter (softplus branch only)
      safe_gate: when False (default) compute y = -exp(A)*softplus(g+g_bias);
        when True compute the bounded y = lower_bound*sigmoid(exp(A)*(g+g_bias))
      lower_bound: floor for the safe_gate branch (default -5.0)
      return  : [..., H, D]
    """
    orig_shape = g.shape[:-1]

    g = g.view(-1, g.shape[-1])
    T = g.shape[0]
    HD = g.shape[1]
    H = A.numel()
    assert H * head_k_dim == HD

    y = torch.empty_like(g, dtype=torch.float32)

    def grid(meta):
        return (cdiv(T, meta["BT"]), H)

    kda_gate_fwd_kernel[grid](
        g,
        A,
        y,
        g_bias,
        beta,
        threshold,
        safe_gate,
        lower_bound if lower_bound is not None else -5.0,
        T,
        H,
        head_k_dim,
        BD=next_power_of_2(head_k_dim),
        HAS_BIAS=g_bias is not None,
    )

    y = y.view(*orig_shape, H, head_k_dim)
    return y


_CHUNK_GLA_FWD_O_KERNEL = Glm5NextChunkGlaFwdOKernel()
_RECOMPUTE_WU_KERNEL = Glm5NextRecomputeWUKernel()
_KDA_GATE_CUMSUM_KERNEL = Glm5NextKdaGateCumsumKernel()
_KDA_INTER_CHUNK_KERNEL = Glm5NextKdaInterChunkKernel()
_KDA_INTRA_CHUNK_KERNEL = Glm5NextKdaIntraChunkKernel()
