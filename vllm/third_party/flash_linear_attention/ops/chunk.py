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

from .chunk_delta_h import chunk_gated_delta_rule_fwd_h
from .chunk_o import chunk_fwd_o
from .chunk_scaled_dot_kkt import chunk_scaled_dot_kkt_fwd
from .cumsum import chunk_local_cumsum
from .l2norm import l2norm_fwd
from .solve_tril import solve_tril
from .utils import FLA_CHUNK_SIZE, SUPPRESS_LEVEL, input_guard
from .wy_fast import recompute_w_u_fwd


def _chunk_gated_delta_rule_fwd(
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
    chunk_offsets: torch.Tensor | None = None,
    core_attn_out: torch.Tensor | None = None,
    state_cache: torch.Tensor | None = None,
    state_indices: torch.Tensor | None = None,
    has_initial_state: torch.Tensor | None = None,
    checkpoint_state: torch.Tensor | None = None,
    checkpoint_offsets: torch.Tensor | None = None,
    checkpoint_state_indices: torch.Tensor | None = None,
):
    g = chunk_local_cumsum(
        g, chunk_size=FLA_CHUNK_SIZE, cu_seqlens=cu_seqlens, chunk_indices=chunk_indices
    )
    # obtain WY representation. u is actually the new v.
    A = chunk_scaled_dot_kkt_fwd(
        k=k,
        beta=beta,
        g=g,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        output_dtype=torch.float32,
    )
    A = solve_tril(
        A=A, cu_seqlens=cu_seqlens, chunk_indices=chunk_indices, output_dtype=k.dtype
    )
    w, u = recompute_w_u_fwd(
        k=k,
        v=v,
        beta=beta,
        A=A,
        g_cumsum=g,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
    )
    h, v_new, final_state = chunk_gated_delta_rule_fwd_h(
        k=k,
        w=w,
        u=u,
        g=g,
        initial_state=initial_state,
        output_final_state=output_final_state,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        chunk_offsets=chunk_offsets,
        state_cache=state_cache,
        state_indices=state_indices,
        has_initial_state=has_initial_state,
        checkpoint_state=checkpoint_state,
        checkpoint_offsets=checkpoint_offsets,
        checkpoint_state_indices=checkpoint_state_indices,
    )
    o = chunk_fwd_o(
        q=q,
        k=k,
        v=v_new,
        h=h,
        g=g,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        core_attn_out=core_attn_out,
    )
    if SUPPRESS_LEVEL < 3:
        return g, o, A, final_state, None, None, None
    elif SUPPRESS_LEVEL >= 3:
        return g, o, A, final_state, w, h, v_new


class ChunkGatedDeltaRuleFunction(torch.autograd.Function):
    @staticmethod
    @input_guard
    @torch.amp.custom_fwd(device_type="cuda")
    def forward(
        ctx,
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
        chunk_offsets: torch.Tensor | None = None,
        use_qk_l2norm_in_kernel: bool = False,
        core_attn_out: torch.Tensor | None = None,
        checkpoint_state: torch.Tensor | None = None,
        checkpoint_offsets: torch.Tensor | None = None,
    ):
        if use_qk_l2norm_in_kernel:
            q = l2norm_fwd(q)
            k = l2norm_fwd(k)

        g, o, A, final_state, w, h, v_new = _chunk_gated_delta_rule_fwd(
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
            chunk_offsets=chunk_offsets,
            core_attn_out=core_attn_out,
            checkpoint_state=checkpoint_state,
            checkpoint_offsets=checkpoint_offsets,
        )
        ctx.scale = scale
        ctx.use_qk_l2norm_in_kernel = use_qk_l2norm_in_kernel
        if core_attn_out is not None:
            assert not torch.is_grad_enabled(), (
                "core_attn_out buffer reuse is only supported for inference"
            )
            assert q.dtype == o.dtype, "Incompatible dtype for inplace computation"
        return o.to(q.dtype), final_state


@torch.compiler.disable
def chunk_gated_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float = None,
    initial_state: torch.Tensor = None,
    output_final_state: bool = False,
    cu_seqlens: torch.Tensor | None = None,
    chunk_indices: torch.Tensor | None = None,
    chunk_offsets: torch.Tensor | None = None,
    use_qk_l2norm_in_kernel: bool = False,
    core_attn_out: torch.Tensor | None = None,
    checkpoint_state: torch.Tensor | None = None,
    checkpoint_offsets: torch.Tensor | None = None,
):
    r"""
    Args:
        q (torch.Tensor):
            Queries of shape `[B, T, H, K]`.
        k (torch.Tensor):
            Keys of shape `[B, T, H, K]`.
        v (torch.Tensor):
            Values of shape `[B, T, H, V]`.
        g (torch.Tensor):
            (forget) Gating tensor (in log space!) of shape `[B, T, H]`.
        beta (torch.Tensor):
            Betas of shape `[B, T, H]`.
        scale (Optional[int]):
            Scale factor for the RetNet attention scores.
            If not provided, it will default to `1 / sqrt(K)`. Default: `None`.
        initial_state (Optional[torch.Tensor]):
            Initial state of shape `[N, H, V, K]` for `N` input sequences.
            For equal-length input sequences, `N` equals the batch size `B`.
            Default: `None`.
        output_final_state (Optional[bool]):
            Whether to output the final state of shape `[N, H, V, K]`. Default: `False`.
        cu_seqlens (torch.Tensor):
            Cumulative sequence lengths of shape `[N+1]` used for variable-length training,
            consistent with the FlashAttention API.
        checkpoint_state (Optional[torch.Tensor]):
            Contiguous `[N, H, V, K]` buffer; row `n` receives sequence `n`'s state.
        checkpoint_offsets (Optional[torch.Tensor]):
            `[N]` token offsets, with `0` for none. Only multiples of 64 strictly
            inside each sequence are stored.
    Returns:
        o (torch.Tensor):
            Outputs of shape `[B, T, H, V]`.
        final_state (torch.Tensor):
            Final state of shape `[N, H, V, K]` if `output_final_state=True` else `None`.

    Examples::
        >>> import torch
        >>> import torch.nn.functional as F
        >>> from einops import rearrange
        >>> from fla.ops.gated_delta_rule import chunk_gated_delta_rule
        # inputs with equal lengths
        >>> B, T, H, K, V = 4, 2048, 4, 512, 512
        >>> q = torch.randn(B, T, H, K, dtype=torch.bfloat16, device='cuda')
        >>> k = F.normalize(torch.randn(B, T, H, K, dtype=torch.bfloat16, device='cuda'), p=2, dim=-1)
        >>> v = torch.randn(B, T, H, V, dtype=torch.bfloat16, device='cuda')
        >>> beta = torch.rand(B, T, H, dtype=torch.bfloat16, device='cuda').sigmoid()
        >>> g = F.logsigmoid(torch.rand(B, T, H, dtype=torch.bfloat16, device='cuda'))
        >>> h0 = torch.randn(B, H, V, K, dtype=torch.bfloat16, device='cuda')
        >>> o, ht = chunk_gated_delta_rule(
            q, k, v, g, beta,
            initial_state=h0,
            output_final_state=True
        )
        # for variable-length inputs, the batch size `B` is expected to be 1 and `cu_seqlens` is required
        >>> q, k, v, beta, g = map(lambda x: rearrange(x, 'b t ... -> 1 (b t) ...'), (q, k, v, beta, g))
        # for a batch with 4 sequences, `cu_seqlens` with 5 start/end positions are expected
        >>> cu_seqlens = q.new_tensor([0, 2048, 4096, 6144, 8192], dtype=torch.int32)
        >>> o_var, ht_var = chunk_gated_delta_rule(
            q, k, v, g, beta,
            initial_state=h0,
            output_final_state=True,
            cu_seqlens=cu_seqlens
        )
    """
    _check_chunk_inputs(q, k, v, beta, cu_seqlens)
    if cu_seqlens is not None:
        if initial_state is not None and initial_state.shape[0] != len(cu_seqlens) - 1:
            raise ValueError(
                f"The number of initial states is expected to be equal to the number of input sequences, "
                f"i.e., {len(cu_seqlens) - 1} rather than {initial_state.shape[0]}."
            )
    if (checkpoint_state is None) != (checkpoint_offsets is None):
        raise ValueError("checkpoint_state and checkpoint_offsets go together")
    if checkpoint_state is not None:
        num_seqs = q.shape[0] if cu_seqlens is None else len(cu_seqlens) - 1
        # input_guard must not replace a writable buffer with a copy.
        expected_shape = (num_seqs, *v.shape[-2:], k.shape[-1])
        if (
            checkpoint_state.shape != expected_shape
            or not checkpoint_state.is_contiguous()
        ):
            raise ValueError(
                "checkpoint_state must be a contiguous [N, H, V, K] buffer"
            )
        if checkpoint_offsets.shape != (num_seqs,):
            raise ValueError(f"checkpoint_offsets must have shape [{num_seqs}]")
    if scale is None:
        scale = k.shape[-1] ** -0.5
    o, final_state = ChunkGatedDeltaRuleFunction.apply(
        q,
        k,
        v,
        g,
        beta,
        scale,
        initial_state,
        output_final_state,
        cu_seqlens,
        chunk_indices,
        chunk_offsets,
        use_qk_l2norm_in_kernel,
        core_attn_out,
        checkpoint_state,
        checkpoint_offsets,
    )
    return o, final_state


@torch.compiler.disable
def chunk_gated_delta_rule_paged(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    *,
    state_cache: torch.Tensor,
    state_indices: torch.Tensor,
    has_initial_state: torch.Tensor,
    checkpoint_offsets: torch.Tensor | None = None,
    checkpoint_state_indices: torch.Tensor | None = None,
    scale: float | None = None,
    cu_seqlens: torch.Tensor | None = None,
    chunk_indices: torch.Tensor | None = None,
    chunk_offsets: torch.Tensor | None = None,
    use_qk_l2norm_in_kernel: bool = False,
    core_attn_out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run inference with recurrent states read and written in the same pool.

    Args:
        state_cache: `[slots, H, V, K]` pool with contiguous rows. Rows may be
            strided apart. The original storage is updated in place.
        state_indices: `[N]` initial and final state row per sequence. Rows
            <= 0 are skipped. Valid final-state rows must be unique.
        has_initial_state: `[N]` bool; false starts from zero.
        checkpoint_offsets: `[N]` token offsets; 0 skips the checkpoint.
            Only multiples of 64 strictly inside each sequence are stored.
        checkpoint_state_indices: `[N]` checkpoint destinations in state_cache;
            rows <= 0 are skipped. Valid destinations must be unique and must
            not overlap any sequence's initial or final state row.

    Returns:
        Attention output. Final states and checkpoints are written in place.
    """
    _check_chunk_inputs(q, k, v, beta, cu_seqlens)
    if (checkpoint_offsets is None) != (checkpoint_state_indices is None):
        raise ValueError("checkpoint_offsets and checkpoint_state_indices go together")
    if core_attn_out is not None and (
        not core_attn_out.is_contiguous() or core_attn_out.dtype != q.dtype
    ):
        raise ValueError("core_attn_out must be contiguous and have the query dtype")
    if scale is None:
        scale = k.shape[-1] ** -0.5
    # Normalize read-only inputs explicitly; never copy the writable pool.
    with torch.accelerator.device_index(q.device.index):
        q, k, v, g, beta = (x.contiguous() for x in (q, k, v, g, beta))
        if use_qk_l2norm_in_kernel:
            q, k = l2norm_fwd(q), l2norm_fwd(k)
        _, o, _, _, _, _, _ = _chunk_gated_delta_rule_fwd(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            scale=scale,
            initial_state=None,
            output_final_state=False,
            cu_seqlens=cu_seqlens.contiguous() if cu_seqlens is not None else None,
            chunk_indices=(
                chunk_indices.contiguous() if chunk_indices is not None else None
            ),
            chunk_offsets=(
                chunk_offsets.contiguous() if chunk_offsets is not None else None
            ),
            core_attn_out=core_attn_out,
            state_cache=state_cache,
            state_indices=state_indices,
            has_initial_state=has_initial_state,
            checkpoint_state=state_cache if checkpoint_offsets is not None else None,
            checkpoint_offsets=checkpoint_offsets,
            checkpoint_state_indices=checkpoint_state_indices,
        )
    return o.to(q.dtype)


def _check_chunk_inputs(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    cu_seqlens: torch.Tensor | None,
) -> None:
    assert q.dtype == k.dtype == v.dtype
    assert q.dtype != torch.float32, "Use bfloat16 inputs for chunked GDN."
    assert len(beta.shape) == 3, "beta must be of shape [B, T, H]."
    if cu_seqlens is not None and q.shape[0] != 1:
        raise ValueError("Flatten variable-length inputs to batch size 1")
