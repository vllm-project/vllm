# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa: E501

import torch

from vllm.model_executor.layers.mamba.ops.mamba_ssm import softplus
from vllm.model_executor.layers.mamba.ops.replayssm_config import get_replayssm_config
from vllm.triton_utils import tl, triton
from vllm.v1.attention.backends.utils import NULL_BLOCK_ID


@triton.heuristics({"HAS_DT_BIAS": lambda args: args["dt_bias_ptr"] is not None})
@triton.heuristics({"HAS_D": lambda args: args["D_ptr"] is not None})
@triton.heuristics({"HAS_Z": lambda args: args["z_ptr"] is not None})
@triton.heuristics(
    {
        "HAS_STATE_BATCH_INDICES": lambda args: args["state_batch_indices_ptr"]
        is not None
    }
)
@triton.heuristics(
    {"BLOCK_SIZE_DSTATE": lambda args: triton.next_power_of_2(args["dstate"])}
)
@triton.jit
def _replayssm_state_and_output_kernel(
    # Pointers to matrices
    state_ptr,
    x_ptr,
    dt_ptr,
    dt_bias_ptr,
    A_ptr,
    B_ptr,
    C_ptr,
    D_ptr,
    z_ptr,
    out_ptr,
    x_cache_ptr,
    dt_cache_ptr,
    B_cache_ptr,
    write_pos_ptr,
    is_flush_ptr,
    state_batch_indices_ptr,
    null_block_id,
    # Matrix dimensions
    batch,
    nheads,
    dim,
    dstate,
    nheads_ngroups_ratio,
    # State strides
    stride_state_batch,
    stride_state_head,
    stride_state_dim,
    stride_state_dstate,
    # Input strides
    stride_x_batch,
    stride_x_head,
    stride_x_dim,
    stride_dt_batch,
    stride_dt_head,
    stride_dt_bias_head,
    stride_A_head,
    stride_B_batch,
    stride_B_group,
    stride_B_dstate,
    stride_C_batch,
    stride_C_group,
    stride_C_dstate,
    stride_D_head,
    stride_D_dim,
    stride_z_batch,
    stride_z_head,
    stride_z_dim,
    stride_out_batch,
    stride_out_head,
    stride_out_dim,
    # Cache strides
    stride_x_cache_batch,
    stride_x_cache_head,
    stride_x_cache_dim,
    stride_x_cache_pos,
    stride_dt_cache_batch,
    stride_dt_cache_head,
    stride_dt_cache_pos,
    stride_B_cache_batch,
    stride_B_cache_group,
    stride_B_cache_pos,
    stride_B_cache_dstate,
    stride_state_indices_batch,
    stride_state_indices_T,
    # Meta-parameters
    DT_SOFTPLUS: tl.constexpr,
    MAX_CACHE_LEN: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    # heuristic-computed
    BLOCK_SIZE_DSTATE: tl.constexpr,
    HAS_DT_BIAS: tl.constexpr,
    HAS_D: tl.constexpr,
    HAS_Z: tl.constexpr,
    HAS_STATE_BATCH_INDICES: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_b = tl.program_id(axis=1)
    pid_h = tl.program_id(axis=2)

    # Resolve the physical state slot for this decode row; skip padded rows.
    if HAS_STATE_BATCH_INDICES:
        state_batch_idx = tl.load(state_batch_indices_ptr + pid_b * stride_state_indices_batch + 0 * stride_state_indices_T).to(tl.int64)
        if state_batch_idx == null_block_id:
            return
    else:
        state_batch_idx = pid_b

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = tl.arange(0, BLOCK_SIZE_DSTATE)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    # Buffer cursor (number of cached tokens so far) and the flush flag.
    write_pos = tl.load(write_pos_ptr + pid_b).to(tl.int64)
    is_flush = tl.load(is_flush_ptr + pid_b) != 0

    # Advance every pointer to this (row, head, group).
    state_ptr += state_batch_idx * stride_state_batch + pid_h * stride_state_head
    x_ptr += pid_b * stride_x_batch + pid_h * stride_x_head
    dt_ptr += pid_b * stride_dt_batch + pid_h * stride_dt_head
    B_ptr += pid_b * stride_B_batch + (pid_h // nheads_ngroups_ratio) * stride_B_group
    C_ptr += pid_b * stride_C_batch + (pid_h // nheads_ngroups_ratio) * stride_C_group
    out_ptr += pid_b * stride_out_batch + pid_h * stride_out_head
    x_cache_ptr += state_batch_idx * stride_x_cache_batch + pid_h * stride_x_cache_head
    dt_cache_ptr += state_batch_idx * stride_dt_cache_batch + pid_h * stride_dt_cache_head
    B_cache_ptr += state_batch_idx * stride_B_cache_batch + (pid_h // nheads_ngroups_ratio) * stride_B_cache_group

    # Current-token dt (+ bias, softplus) and the per-head scalar A.
    dt_cur = tl.load(dt_ptr).to(tl.float32)
    if HAS_DT_BIAS:
        dt_cur += tl.load(dt_bias_ptr + pid_h * stride_dt_bias_head).to(tl.float32)
    if DT_SOFTPLUS:
        dt_cur = tl.where(dt_cur <= 20.0, softplus(dt_cur), dt_cur)
    A = tl.load(A_ptr + pid_h * stride_A_head).to(tl.float32)

    # dt over the window (cached history + current token at write_pos), then the
    # decay weights: total decay exp(A*sum dt) and the per-position scale s_j.
    dt_all = tl.load(dt_cache_ptr + offs_k * stride_dt_cache_pos, mask=offs_k < write_pos, other=0.0).to(tl.float32)
    dt_all = tl.where(offs_k == write_pos, dt_cur, dt_all)
    dA_cumsum = A * tl.cumsum(dt_all, axis=0)
    dA_total = A * tl.sum(dt_all, axis=0)
    total_decay = tl.exp(dA_total)
    scale = dt_all * tl.exp(dA_total - dA_cumsum)
    scale = tl.where(offs_k <= write_pos, scale, 0.0)

    # Current-token x / C, the checkpoint state S_0, and current-token B.
    x_cur = tl.load(x_ptr + offs_m * stride_x_dim, mask=offs_m < dim, other=0.0)
    C = tl.load(C_ptr + offs_n * stride_C_dstate, mask=offs_n < dstate, other=0.0).to(tl.float32)
    state_ptrs = state_ptr + offs_m[:, None] * stride_state_dim + offs_n[None, :] * stride_state_dstate
    state = tl.load(state_ptrs, mask=(offs_m[:, None] < dim) & (offs_n[None, :] < dstate), other=0.0)
    B_cur = tl.load(B_ptr + offs_n * stride_B_dstate, mask=offs_n < dstate, other=0.0)

    # Gather buffered x and B over the window (history + current token).
    x_all_ptrs = x_cache_ptr + offs_m[:, None] * stride_x_cache_dim + offs_k[None, :] * stride_x_cache_pos
    x_all = tl.load(x_all_ptrs, mask=(offs_m[:, None] < dim) & (offs_k[None, :] < write_pos), other=0.0)
    x_all = tl.where(offs_k[None, :] == write_pos, x_cur[:, None], x_all)
    B_all_ptrs = B_cache_ptr + offs_k[:, None] * stride_B_cache_pos + offs_n[None, :] * stride_B_cache_dstate
    B_all = tl.load(B_all_ptrs, mask=(offs_k[:, None] < write_pos) & (offs_n[None, :] < dstate), other=0.0)
    B_all = tl.where(offs_k[:, None] == write_pos, B_cur[None, :], B_all)

    # Reconstruct the state from cached inputs (outer-product / state route):
    # S_t = total_decay * S_0 + sum_j s_j (v_j k_j^T). Store it back on a flush.
    B_scaled = (B_all.to(tl.float32) * scale[:, None]).to(x_ptr.dtype.element_ty)
    # tf32x3 keeps fp32 parity with the elementwise baseline (plain tf32 on
    # fp32 inputs drifts ~1e-2); bf16/fp16 inputs are unaffected by this flag.
    delta_state = tl.dot(
        x_all.to(x_ptr.dtype.element_ty), B_scaled, input_precision="tf32x3"
    )
    state_new = state.to(tl.float32) * total_decay + delta_state.to(tl.float32)
    if is_flush:
        tl.store(state_ptrs, state_new.to(state.dtype), mask=(offs_m[:, None] < dim) & (offs_n[None, :] < dstate))

    # Read the output from the reconstructed state: y = S_t q.
    out = tl.sum(state_new * C[None, :], axis=1)

    # Skip connection (D) and output gate (z).
    if HAS_D:
        D_ptr += pid_h * stride_D_head
        D = tl.load(D_ptr + offs_m * stride_D_dim, mask=offs_m < dim, other=0.0).to(tl.float32)
        out += x_cur.to(tl.float32) * D
    if HAS_Z:
        z_ptr += pid_b * stride_z_batch + pid_h * stride_z_head
        z = tl.load(z_ptr + offs_m * stride_z_dim, mask=offs_m < dim, other=0.0).to(tl.float32)
        out *= z * tl.sigmoid(z)

    tl.store(out_ptr + offs_m * stride_out_dim, out, mask=offs_m < dim)

    # Non-flush step: append the current token (x, dt, B) into the buffer at write_pos.
    if not is_flush:
        tl.store(x_cache_ptr + offs_m * stride_x_cache_dim + write_pos * stride_x_cache_pos, x_cur, mask=offs_m < dim)
        if pid_m == 0:
            tl.store(dt_cache_ptr + write_pos * stride_dt_cache_pos, dt_cur)
            tl.store(B_cache_ptr + write_pos * stride_B_cache_pos + offs_n * stride_B_cache_dstate, B_cur, mask=offs_n < dstate)


def selective_state_update_replayssm_state_and_output(
    state: torch.Tensor,
    x: torch.Tensor,
    dt: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: torch.Tensor | None = None,
    dt_bias: torch.Tensor | None = None,
    z: torch.Tensor | None = None,
    dt_softplus: bool = False,
    x_cache: torch.Tensor | None = None,
    dt_cache: torch.Tensor | None = None,
    B_cache: torch.Tensor | None = None,
    write_pos: torch.Tensor | None = None,
    is_flush: torch.Tensor | None = None,
    max_cache_len: int = 16,
    state_batch_indices: torch.Tensor | None = None,
    null_block_id: int = NULL_BLOCK_ID,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Cached-dot SSM update for vLLM's autoregressive Mamba2 decode path."""
    has_heads = state.dim() > 3
    if state.dim() == 3:
        state = state.unsqueeze(1)
    if x.dim() == 2:
        x = x.unsqueeze(1)
    if dt.dim() == 2:
        dt = dt.unsqueeze(1)
    if A.dim() == 2:
        A = A.unsqueeze(0)
    if B.dim() == 2:
        B = B.unsqueeze(1)
    if C.dim() == 2:
        C = C.unsqueeze(1)
    if D is not None and D.dim() == 1:
        D = D.unsqueeze(0)
    if z is not None and z.dim() == 2:
        z = z.unsqueeze(1)
    if dt_bias is not None and dt_bias.dim() == 1:
        dt_bias = dt_bias.unsqueeze(0)
    if out is not None and out.dim() == 2:
        out = out.unsqueeze(1)
    if state_batch_indices is not None and state_batch_indices.dim() == 1:
        state_batch_indices = state_batch_indices.unsqueeze(1)

    _, nheads, dim, dstate = state.shape
    batch = x.shape[0]
    assert x.shape == (batch, nheads, dim)
    assert dt.shape == x.shape
    assert A.shape == (nheads, dim, dstate)
    ngroups = B.shape[1]
    assert nheads % ngroups == 0, "nheads must be divisible by ngroups"
    assert B.shape == (batch, ngroups, dstate)
    assert C.shape == B.shape
    if D is not None:
        assert D.shape == (nheads, dim)
    if z is not None:
        assert z.shape == x.shape
    if dt_bias is not None:
        assert dt_bias.shape == (nheads, dim)
    assert out is not None and out.shape == x.shape

    assert A.stride(-1) == 0 and A.stride(-2) == 0, (
        "Cached kernel requires TIE_HDIM (A scalar per head)"
    )
    assert dt.stride(-1) == 0, "Cached kernel requires TIE_HDIM (dt scalar per head)"
    if dt_bias is not None:
        assert dt_bias.stride(-1) == 0, (
            "Cached kernel requires TIE_HDIM (dt_bias scalar per head)"
        )

    assert x_cache is not None
    assert dt_cache is not None
    assert B_cache is not None
    assert x_cache.shape[1:] == (nheads, max_cache_len, dim)
    assert dt_cache.shape[1:] == (nheads, max_cache_len)
    assert B_cache.shape[1:] == (ngroups, max_cache_len, dstate)
    assert write_pos is not None and write_pos.shape[0] >= batch
    assert write_pos.dtype == torch.int32
    assert is_flush is not None and is_flush.shape[0] >= batch
    assert is_flush.dtype in (torch.bool, torch.int8)
    if state_batch_indices is not None:
        assert state_batch_indices.shape[0] >= batch
        assert state_batch_indices.shape[1] >= 1

    block_size_k = max(16, triton.next_power_of_2(max_cache_len))
    block_size_m, num_warps = get_replayssm_config(
        "mamba2_state_and_output", dstate=dstate, L=max_cache_len
    )

    grid = lambda META: (triton.cdiv(dim, META["BLOCK_SIZE_M"]), batch, nheads)
    z_strides = (z.stride(0), z.stride(1), z.stride(2)) if z is not None else (0, 0, 0)
    state_indices_strides = (
        (state_batch_indices.stride(0), state_batch_indices.stride(1))
        if state_batch_indices is not None
        else (0, 0)
    )

    with torch.accelerator.device_index(x.device.index):
        _replayssm_state_and_output_kernel[grid](
            state,
            x,
            dt,
            dt_bias,
            A,
            B,
            C,
            D,
            z,
            out,
            x_cache,
            dt_cache,
            B_cache,
            write_pos,
            is_flush,
            state_batch_indices,
            null_block_id,
            batch,
            nheads,
            dim,
            dstate,
            nheads // ngroups,
            state.stride(0),
            state.stride(1),
            state.stride(2),
            state.stride(3),
            x.stride(0),
            x.stride(1),
            x.stride(2),
            dt.stride(0),
            dt.stride(1),
            dt_bias.stride(0) if dt_bias is not None else 0,
            A.stride(0),
            B.stride(0),
            B.stride(1),
            B.stride(2),
            C.stride(0),
            C.stride(1),
            C.stride(2),
            D.stride(0) if D is not None else 0,
            D.stride(1) if D is not None else 0,
            z_strides[0],
            z_strides[1],
            z_strides[2],
            out.stride(0),
            out.stride(1),
            out.stride(2),
            x_cache.stride(0),
            x_cache.stride(1),
            x_cache.stride(3),
            x_cache.stride(2),
            dt_cache.stride(0),
            dt_cache.stride(1),
            dt_cache.stride(2),
            B_cache.stride(0),
            B_cache.stride(1),
            B_cache.stride(2),
            B_cache.stride(3),
            state_indices_strides[0],
            state_indices_strides[1],
            dt_softplus,
            max_cache_len,
            block_size_m,
            block_size_k,
            num_warps=num_warps,
        )

    if not has_heads:
        out = out.squeeze(1)
    return out
