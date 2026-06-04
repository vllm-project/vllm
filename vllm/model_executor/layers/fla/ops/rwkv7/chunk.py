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
# https://github.com/fla-org/flash-linear-attention/blob/main/fla/ops/rwkv7/chunk.py
# https://github.com/fla-org/flash-linear-attention/blob/main/fla/ops/generalized_delta_rule/dplr/chunk.py
# Forward path only.
# ruff: noqa: E501

import torch

from ..index import prepare_chunk_indices
from .chunk_a import chunk_dplr_fwd_intra
from .chunk_h import chunk_dplr_fwd_h
from .chunk_o import chunk_dplr_fwd_o
from .cumsum import chunk_rwkv6_fwd_cumsum
from .wy_fast import prepare_wy_repr_fwd


def chunk_dplr_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    gk: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor | None,
    output_final_state: bool,
    cu_seqlens: torch.Tensor | None = None,
    chunk_size: int = 16,
    safe_gate: bool = False,
    chunk_indices: torch.Tensor | None = None,
    return_varlen_states: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    """Forward chunked DPLR delta rule, used by RWKV-7.

    Computes ``o`` and the final recurrent state via chunk-parallel kernels:
        gi, ge = cumsum(gk)              # cumulative decay
        intra: A_*, projected qg/kg/ag/bg
        wy:    w, u, A_ab_inv            # WY representation for inter-chunk recurrence
        h:     chunk states + v_new      # recurrent state evolution
        o:     final outputs
    """
    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size)

    gi, ge = chunk_rwkv6_fwd_cumsum(
        gk, chunk_size, cu_seqlens=cu_seqlens, chunk_indices=chunk_indices
    )

    A_ab, A_qk, A_ak, A_qb, qg, kg, ag, bg = chunk_dplr_fwd_intra(
        q=q,
        k=k,
        a=a,
        b=b,
        gi=gi,
        ge=ge,
        scale=scale,
        cu_seqlens=cu_seqlens,
        safe_gate=safe_gate,
        chunk_size=chunk_size,
        chunk_indices=chunk_indices,
    )

    w, u, _A_ab_inv = prepare_wy_repr_fwd(
        ag=ag,
        A_ab=A_ab,
        A_ak=A_ak,
        v=v,
        cu_seqlens=cu_seqlens,
        chunk_size=chunk_size,
        chunk_indices=chunk_indices,
    )

    h, v_new, final_state, varlen_states = chunk_dplr_fwd_h(
        kg=kg,
        bg=bg,
        v=v,
        w=w,
        u=u,
        gk=gi,
        initial_state=initial_state,
        output_final_state=output_final_state,
        cu_seqlens=cu_seqlens,
        chunk_size=chunk_size,
        chunk_indices=chunk_indices,
        return_varlen_states=return_varlen_states,
    )

    o = chunk_dplr_fwd_o(
        qg=qg,
        v=v,
        v_new=v_new,
        A_qk=A_qk,
        A_qb=A_qb,
        h=h,
        cu_seqlens=cu_seqlens,
        chunk_size=chunk_size,
        chunk_indices=chunk_indices,
    )

    return o.to(q.dtype), final_state, varlen_states


def chunk_dplr_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    gk: torch.Tensor,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    cu_seqlens: torch.Tensor | None = None,
    safe_gate: bool = False,
    chunk_size: int | None = None,
    return_varlen_states: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    if chunk_size is None:
        chunk_size = 16
    if cu_seqlens is not None:
        if q.shape[0] != 1:
            raise ValueError(
                f"Batch size must be 1 when using cu_seqlens, got {q.shape[0]}"
            )
        if initial_state is not None and initial_state.shape[0] != len(cu_seqlens) - 1:
            raise ValueError(
                f"initial_state must have {len(cu_seqlens) - 1} sequences, "
                f"got {initial_state.shape[0]}"
            )
    if scale is None:
        scale = k.shape[-1] ** -0.5
    return chunk_dplr_fwd(
        q=q,
        k=k,
        v=v,
        a=a,
        b=b,
        gk=gk,
        scale=scale,
        initial_state=initial_state,
        output_final_state=output_final_state,
        cu_seqlens=cu_seqlens,
        chunk_size=chunk_size,
        safe_gate=safe_gate,
        return_varlen_states=return_varlen_states,
    )


def chunk_rwkv7(
    r: torch.Tensor,
    w: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    scale: float = 1.0,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    cu_seqlens: torch.Tensor | None = None,
    safe_gate: bool = False,
    chunk_size: int | None = None,
    return_varlen_states: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    """Chunked-prefill DPLR delta rule for RWKV-7.

    Args:
        r: ``[B, T, H, K]`` (receptance / queries)
        w: ``[B, T, H, K]`` log-decay
        k: ``[B, T, H, K]`` keys
        v: ``[B, T, H, V]`` values
        a: ``[B, T, H, K]`` activations (== ``-kk`` for RWKV-7)
        b: ``[B, T, H, K]`` betas (== ``kk * a_gate`` for RWKV-7)
        initial_state: ``[N, H, K, V]`` per-sequence prior recurrent state
        cu_seqlens: ``[N+1]`` cumulative sequence lengths (FlashAttention API)
        safe_gate: if True, use the TensorCore-accelerated intra kernel
            (requires gates in roughly ``[-5, 0)``)
        chunk_size: chunk parallelism granularity. Default 16 (only 16 and 64
            are tuned in the underlying kernels).

    Returns ``(o, final_state, varlen_states)``. ``varlen_states`` is
    ``None`` unless ``return_varlen_states=True``, in which case it has
    shape ``[NT_total, H, K, V]`` carrying the recurrent state at the END
    of each chunk (state after consuming chunk k, indexed in the same
    flat-across-sequences layout the chunk kernels use).
    """
    return chunk_dplr_delta_rule(
        q=r,
        k=k,
        v=v,
        a=a,
        b=b,
        gk=w,
        scale=scale,
        initial_state=initial_state,
        output_final_state=output_final_state,
        cu_seqlens=cu_seqlens,
        safe_gate=safe_gate,
        chunk_size=chunk_size,
        return_varlen_states=return_varlen_states,
    )
