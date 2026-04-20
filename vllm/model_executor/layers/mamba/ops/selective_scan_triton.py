# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Triton implementation of the Mamba selective scan forward pass.

This provides a platform-portable alternative to the CUDA-only
selective_scan_fwd kernel. It supports both varlen and non-varlen modes,
with optional z-gating, D bias, delta bias, and delta softplus.

The kernel uses a sequential scan approach: each program handles one
(batch, dim) pair and iterates over the sequence length, maintaining
the SSM state vector across positions. Parallelism comes from launching
batch * dim programs concurrently.
"""

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _softplus(x):
    return tl.where(x <= 20.0, tl.math.log(tl.math.exp(x) + 1.0), x)


@triton.jit
def _selective_scan_fwd_kernel(
    # Pointers to input tensors
    u_ptr,
    delta_ptr,
    A_ptr,
    B_ptr,
    C_ptr,
    D_ptr,
    z_ptr,
    delta_bias_ptr,
    # Pointers to output tensors (out aliases delta, out_z aliases z)
    out_ptr,
    out_z_ptr,
    # SSM states
    ssm_states_ptr,
    # Optional pointers
    query_start_loc_ptr,
    cache_indices_ptr,
    has_initial_state_ptr,
    # APC pointers
    block_idx_first_ptr,
    block_idx_last_ptr,
    initial_state_idx_ptr,
    cu_chunk_seqlen_ptr,
    last_chunk_indices_ptr,
    # Dimensions
    batch: tl.int32,
    dim: tl.int32,
    seqlen: tl.int32,
    dstate: tl.int32,
    n_groups: tl.int32,
    dim_ngroups_ratio: tl.int32,
    # Strides for u (and out, since out = delta which has same layout)
    u_batch_stride: tl.int64,
    u_d_stride: tl.int64,
    # Strides for delta
    delta_batch_stride: tl.int64,
    delta_d_stride: tl.int64,
    # Strides for A
    A_d_stride: tl.int64,
    A_dstate_stride: tl.int64,
    # Strides for B
    B_batch_stride: tl.int64,
    B_group_stride: tl.int64,
    B_dstate_stride: tl.int64,
    # Strides for C
    C_batch_stride: tl.int64,
    C_group_stride: tl.int64,
    C_dstate_stride: tl.int64,
    # Strides for z
    z_batch_stride: tl.int64,
    z_d_stride: tl.int64,
    # Strides for out
    out_batch_stride: tl.int64,
    out_d_stride: tl.int64,
    # Strides for out_z
    out_z_batch_stride: tl.int64,
    out_z_d_stride: tl.int64,
    # Strides for ssm_states
    ssm_batch_stride: tl.int64,
    ssm_dim_stride: tl.int64,
    ssm_dstate_stride: tl.int64,
    # Cache strides
    cache_indices_stride: tl.int64,
    # Scalar params
    null_block_id: tl.int64,
    block_size: tl.int32,
    # Compile-time constants
    delta_softplus: tl.constexpr,
    HAS_D: tl.constexpr,
    HAS_Z: tl.constexpr,
    HAS_DELTA_BIAS: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    HAS_CACHE_INDICES: tl.constexpr,
    CACHE_ENABLED: tl.constexpr,
    BLOCK_DSTATE: tl.constexpr,
):
    batch_idx = tl.program_id(0)
    dim_idx = tl.program_id(1)
    group_idx = dim_idx // dim_ngroups_ratio

    # Determine sequence boundaries
    if IS_VARLEN:
        seq_start = tl.load(query_start_loc_ptr + batch_idx).to(tl.int32)
        seq_end = tl.load(query_start_loc_ptr + batch_idx + 1).to(tl.int32)
        actual_seqlen = seq_end - seq_start
    else:
        seq_start = 0
        actual_seqlen = seqlen

    # Determine cache index for ssm_states
    if CACHE_ENABLED:
        init_state_idx = tl.load(initial_state_idx_ptr + batch_idx).to(tl.int32)
        load_cache_slot = tl.load(
            cache_indices_ptr + batch_idx * cache_indices_stride + init_state_idx
        ).to(tl.int64)
        if load_cache_slot == null_block_id:
            return
    elif HAS_CACHE_INDICES:
        cache_index = tl.load(cache_indices_ptr + batch_idx).to(tl.int64)
        if cache_index == null_block_id:
            return
        load_cache_slot = cache_index
    else:
        load_cache_slot = batch_idx.to(tl.int64)

    # Load D value
    D_val = 0.0
    if HAS_D:
        D_val = tl.load(D_ptr + dim_idx).to(tl.float32)

    # Load delta_bias value
    delta_bias_val = 0.0
    if HAS_DELTA_BIAS:
        delta_bias_val = tl.load(delta_bias_ptr + dim_idx).to(tl.float32)

    # Load A values for this dim - shape (dstate,)
    dstate_offs = tl.arange(0, BLOCK_DSTATE)
    dstate_mask = dstate_offs < dstate
    A_vals = tl.load(
        A_ptr + dim_idx * A_d_stride + dstate_offs * A_dstate_stride,
        mask=dstate_mask,
        other=0.0,
    ).to(tl.float32)

    # Initialize state vector
    state = tl.zeros((BLOCK_DSTATE,), dtype=tl.float32)

    # Load initial state if available
    has_init = False
    if has_initial_state_ptr is not None:
        has_init = tl.load(has_initial_state_ptr + batch_idx)
    if has_init:
        state = tl.load(
            ssm_states_ptr
            + load_cache_slot * ssm_batch_stride
            + dim_idx * ssm_dim_stride
            + dstate_offs * ssm_dstate_stride,
            mask=dstate_mask,
            other=0.0,
        ).to(tl.float32)

    # Compute base addresses for u and delta
    if IS_VARLEN:
        u_base = u_ptr + dim_idx * u_d_stride + seq_start * u_batch_stride
        delta_base = (
            delta_ptr + dim_idx * delta_d_stride + seq_start * delta_batch_stride
        )
        out_base = (
            out_ptr + dim_idx * out_d_stride + seq_start * out_batch_stride
        )
        B_base = B_ptr + group_idx * B_group_stride + seq_start * B_batch_stride
        C_base = C_ptr + group_idx * C_group_stride + seq_start * C_batch_stride
    else:
        u_base = u_ptr + batch_idx * u_batch_stride + dim_idx * u_d_stride
        delta_base = (
            delta_ptr + batch_idx * delta_batch_stride + dim_idx * delta_d_stride
        )
        out_base = (
            out_ptr + batch_idx * out_batch_stride + dim_idx * out_d_stride
        )
        B_base = B_ptr + batch_idx * B_batch_stride + group_idx * B_group_stride
        C_base = C_ptr + batch_idx * C_batch_stride + group_idx * C_group_stride

    if HAS_Z:
        if IS_VARLEN:
            z_base = z_ptr + dim_idx * z_d_stride + seq_start * z_batch_stride
            out_z_base = (
                out_z_ptr
                + dim_idx * out_z_d_stride
                + seq_start * out_z_batch_stride
            )
        else:
            z_base = z_ptr + batch_idx * z_batch_stride + dim_idx * z_d_stride
            out_z_base = (
                out_z_ptr
                + batch_idx * out_z_batch_stride
                + dim_idx * out_z_d_stride
            )

    # Determine chunk boundaries for APC mode
    if CACHE_ENABLED:
        last_chunk_idx = tl.load(last_chunk_indices_ptr + batch_idx).to(tl.int32)
        if batch_idx == 0:
            first_chunk_idx = 0
        else:
            first_chunk_idx = (
                tl.load(last_chunk_indices_ptr + batch_idx - 1).to(tl.int32) + 1
            )
        n_chunks = last_chunk_idx - first_chunk_idx + 1
        first_chunk_tokens = (
            tl.load(cu_chunk_seqlen_ptr + first_chunk_idx + 1).to(tl.int32)
            - tl.load(cu_chunk_seqlen_ptr + first_chunk_idx).to(tl.int32)
        )
        block_idx_first = tl.load(block_idx_first_ptr + batch_idx).to(tl.int32)
        chunk_start_offset = 0
        if n_chunks > 1 and first_chunk_tokens < block_size:
            chunk_start_offset = block_size - first_chunk_tokens
        current_position = block_idx_first * block_size + chunk_start_offset
    else:
        n_chunks = 1
        first_chunk_idx = 0

    # Sequential scan over the sequence
    tokens_processed = 0
    for chunk in range(0, n_chunks if CACHE_ENABLED else 1):
        if CACHE_ENABLED:
            chunk_tokens = (
                tl.load(
                    cu_chunk_seqlen_ptr + first_chunk_idx + chunk + 1
                ).to(tl.int32)
                - tl.load(
                    cu_chunk_seqlen_ptr + first_chunk_idx + chunk
                ).to(tl.int32)
            )
        else:
            chunk_tokens = actual_seqlen

        for local_pos in range(chunk_tokens):
            pos = tokens_processed + local_pos
            # Load u value
            u_val = tl.load(u_base + pos).to(tl.float32)

            # Load delta value
            delta_val = tl.load(delta_base + pos).to(tl.float32)

            # Apply delta bias
            if HAS_DELTA_BIAS:
                delta_val = delta_val + delta_bias_val

            # Apply softplus
            if delta_softplus:
                delta_val = _softplus(delta_val)

            delta_u = delta_val * u_val

            # Compute dA = exp(delta * A) for all dstate elements
            dA = tl.exp(delta_val * A_vals)

            # Load B values for this position
            B_vals = tl.load(
                B_base + dstate_offs * B_dstate_stride + pos,
                mask=dstate_mask,
                other=0.0,
            ).to(tl.float32)

            # Load C values for this position
            C_vals = tl.load(
                C_base + dstate_offs * C_dstate_stride + pos,
                mask=dstate_mask,
                other=0.0,
            ).to(tl.float32)

            # Update state: state = dA * state + delta * u * B
            state = dA * state + delta_u * B_vals

            # Compute output: out = sum(state * C) + D * u
            out_val = tl.sum(state * C_vals, axis=0)
            if HAS_D:
                out_val = out_val + D_val * u_val

            # Store output
            tl.store(out_base + pos, out_val.to(out_ptr.dtype.element_ty))

            if HAS_Z:
                z_val = tl.load(z_base + pos).to(tl.float32)
                out_z_val = out_val * z_val / (1.0 + tl.exp(-z_val))
                tl.store(
                    out_z_base + pos,
                    out_z_val.to(out_z_ptr.dtype.element_ty),
                )

        tokens_processed += chunk_tokens

        # Store intermediate state for APC mode
        if CACHE_ENABLED:
            if chunk == n_chunks - 1:
                store_slot = tl.load(
                    cache_indices_ptr
                    + batch_idx * cache_indices_stride
                    + tl.load(block_idx_last_ptr + batch_idx).to(tl.int32)
                ).to(tl.int64)
            else:
                block_idx_done = (
                    current_position + chunk_tokens - 1
                ) // block_size
                store_slot = tl.load(
                    cache_indices_ptr
                    + batch_idx * cache_indices_stride
                    + block_idx_done
                ).to(tl.int64)

            tl.store(
                ssm_states_ptr
                + store_slot * ssm_batch_stride
                + dim_idx * ssm_dim_stride
                + dstate_offs * ssm_dstate_stride,
                state.to(ssm_states_ptr.dtype.element_ty),
                mask=dstate_mask,
            )
            current_position += chunk_tokens

    # Store final state for non-APC mode
    if not CACHE_ENABLED:
        tl.store(
            ssm_states_ptr
            + load_cache_slot * ssm_batch_stride
            + dim_idx * ssm_dim_stride
            + dstate_offs * ssm_dstate_stride,
            state.to(ssm_states_ptr.dtype.element_ty),
            mask=dstate_mask,
        )


def selective_scan_fwd_triton(
    u: torch.Tensor,
    delta: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D_: torch.Tensor | None,
    z_: torch.Tensor | None,
    delta_bias_: torch.Tensor | None,
    delta_softplus: bool,
    query_start_loc: torch.Tensor | None,
    cache_indices: torch.Tensor | None,
    has_initial_state: torch.Tensor | None,
    ssm_states: torch.Tensor,
    null_block_id: int,
    block_size: int = 1024,
    block_idx_first_scheduled_token: torch.Tensor | None = None,
    block_idx_last_scheduled_token: torch.Tensor | None = None,
    initial_state_idx: torch.Tensor | None = None,
    cu_chunk_seqlen: torch.Tensor | None = None,
    last_chunk_indices: torch.Tensor | None = None,
):
    """
    Triton implementation of selective scan forward pass.

    This writes output in-place to delta (when z is None) or z (when z is
    provided), matching the CUDA kernel's behavior.

    See selective_scan_fn() in mamba_ssm.py for parameter documentation.
    """
    varlen = query_start_loc is not None
    batch_size = (
        (query_start_loc.shape[0] - 1) if varlen else u.shape[0]
    )
    dim = u.shape[0] if varlen else u.shape[1]
    total_seqlen = u.shape[1] if varlen else u.shape[2]
    dstate = A.size(1)
    n_groups = B.size(0) if varlen else B.size(1)
    dim_ngroups_ratio = dim // n_groups

    has_z = z_ is not None
    has_D = D_ is not None
    has_delta_bias = delta_bias_ is not None
    has_cache_indices = cache_indices is not None
    cache_enabled = block_idx_first_scheduled_token is not None

    # out and out_z alias delta and z respectively
    out = delta
    out_z = z_ if has_z else delta  # dummy, won't be used if not has_z

    BLOCK_DSTATE = triton.next_power_of_2(dstate)

    # Compute strides
    if varlen:
        u_batch_stride = u.stride(1)
        u_d_stride = u.stride(0)
        delta_batch_stride = delta.stride(1)
        delta_d_stride = delta.stride(0)
        B_batch_stride = B.stride(2)
        B_group_stride = B.stride(0)
        B_dstate_stride = B.stride(1)
        C_batch_stride = C.stride(2)
        C_group_stride = C.stride(0)
        C_dstate_stride = C.stride(1)
        out_batch_stride = out.stride(1)
        out_d_stride = out.stride(0)
        if has_z:
            z_batch_stride = z_.stride(1)
            z_d_stride = z_.stride(0)
            out_z_batch_stride = out_z.stride(1)
            out_z_d_stride = out_z.stride(0)
        else:
            z_batch_stride = 0
            z_d_stride = 0
            out_z_batch_stride = 0
            out_z_d_stride = 0
    else:
        u_batch_stride = u.stride(0)
        u_d_stride = u.stride(1)
        delta_batch_stride = delta.stride(0)
        delta_d_stride = delta.stride(1)
        B_batch_stride = B.stride(0)
        B_group_stride = B.stride(1)
        B_dstate_stride = B.stride(2)
        C_batch_stride = C.stride(0)
        C_group_stride = C.stride(1)
        C_dstate_stride = C.stride(2)
        out_batch_stride = out.stride(0)
        out_d_stride = out.stride(1)
        if has_z:
            z_batch_stride = z_.stride(0)
            z_d_stride = z_.stride(1)
            out_z_batch_stride = out_z.stride(0)
            out_z_d_stride = out_z.stride(1)
        else:
            z_batch_stride = 0
            z_d_stride = 0
            out_z_batch_stride = 0
            out_z_d_stride = 0

    ssm_batch_stride = ssm_states.stride(0)
    ssm_dim_stride = ssm_states.stride(1)
    ssm_dstate_stride = ssm_states.stride(2)
    cache_indices_stride = cache_indices.stride(0) if has_cache_indices else 0

    grid = (batch_size, dim)
    _selective_scan_fwd_kernel[grid](
        u,
        delta,
        A,
        B,
        C,
        D_ if has_D else u,  # dummy, won't be dereferenced
        z_ if has_z else u,  # dummy
        delta_bias_ if has_delta_bias else u,  # dummy
        out,
        out_z,
        ssm_states,
        query_start_loc if varlen else u,  # dummy
        cache_indices if has_cache_indices else u,  # dummy
        has_initial_state,
        # APC pointers
        block_idx_first_scheduled_token if cache_enabled else u,
        block_idx_last_scheduled_token if cache_enabled else u,
        initial_state_idx if cache_enabled else u,
        cu_chunk_seqlen if cache_enabled else u,
        last_chunk_indices if cache_enabled else u,
        # Dimensions
        batch_size,
        dim,
        total_seqlen,
        dstate,
        n_groups,
        dim_ngroups_ratio,
        # Strides
        u_batch_stride,
        u_d_stride,
        delta_batch_stride,
        delta_d_stride,
        A.stride(0),
        A.stride(1),
        B_batch_stride,
        B_group_stride,
        B_dstate_stride,
        C_batch_stride,
        C_group_stride,
        C_dstate_stride,
        z_batch_stride,
        z_d_stride,
        out_batch_stride,
        out_d_stride,
        out_z_batch_stride,
        out_z_d_stride,
        ssm_batch_stride,
        ssm_dim_stride,
        ssm_dstate_stride,
        cache_indices_stride,
        null_block_id,
        block_size,
        # Compile-time constants
        delta_softplus=delta_softplus,
        HAS_D=has_D,
        HAS_Z=has_z,
        HAS_DELTA_BIAS=has_delta_bias,
        IS_VARLEN=varlen,
        HAS_CACHE_INDICES=has_cache_indices,
        CACHE_ENABLED=cache_enabled,
        BLOCK_DSTATE=BLOCK_DSTATE,
    )
