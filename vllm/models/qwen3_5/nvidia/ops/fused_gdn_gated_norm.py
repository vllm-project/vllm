# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch

from vllm.third_party.flash_linear_attention.ops.layernorm_guard import (
    calc_rows_per_block,
)
from vllm.triton_utils import tl, triton


@triton.jit(do_not_specialize=["M"])
def _fused_gdn_gated_rms_norm_kernel(
    x_ptr,
    z_ptr,
    w_ptr,
    out_ptr,
    scale_ptr,
    stride_x_tok,
    stride_z_tok,
    stride_out_tok,
    M,  # number of tokens
    eps,
    N: tl.constexpr,  # head_v_dim
    BLOCK_N: tl.constexpr,
    ROWS_PER_BLOCK: tl.constexpr,
    NORM_BEFORE_GATE: tl.constexpr,
    ACTIVATION: tl.constexpr,
    QUANT_FP8: tl.constexpr,
):
    """FLA-equivalent per-head gated RMS norm with token-strided z."""
    row_start = tl.program_id(0) * ROWS_PER_BLOCK
    head = tl.program_id(1)

    rows = row_start + tl.arange(0, ROWS_PER_BLOCK)
    cols = tl.arange(0, BLOCK_N)
    row_mask = rows[:, None] < M
    col_mask = cols[None, :] < N
    mask = row_mask & col_mask

    x_ptrs = x_ptr + rows[:, None] * stride_x_tok + head * N + cols[None, :]
    z_ptrs = z_ptr + rows[:, None] * stride_z_tok + head * N + cols[None, :]

    x = tl.load(x_ptrs, mask=mask, other=0.0).to(tl.float32)

    if not NORM_BEFORE_GATE:
        z = tl.load(z_ptrs, mask=mask, other=0.0).to(tl.float32)
        if ACTIVATION == "swish" or ACTIVATION == "silu":
            x *= z * tl.sigmoid(z)
        elif ACTIVATION == "sigmoid":
            x *= tl.sigmoid(z)

    xbar = tl.where(mask, x, 0.0)
    var = tl.sum(xbar * xbar, axis=1) / N
    rstd = tl.rsqrt(var + eps)

    w = tl.load(w_ptr + cols, mask=cols < N, other=0.0).to(tl.float32)
    y = x * rstd[:, None] * w[None, :]

    if NORM_BEFORE_GATE:
        z = tl.load(z_ptrs, mask=mask, other=0.0).to(tl.float32)
        if ACTIVATION == "swish" or ACTIVATION == "silu":
            y *= z * tl.sigmoid(z)
        elif ACTIVATION == "sigmoid":
            y *= tl.sigmoid(z)

    out_ptrs = out_ptr + rows[:, None] * stride_out_tok + head * N + cols[None, :]
    if QUANT_FP8:
        # Static FP8 from the bf16-rounded output, like scaled_fp8_quant.
        y = y.to(x_ptr.dtype.element_ty).to(tl.float32)
        y = y * tl.math.div_rn(1.0, tl.load(scale_ptr))
        y = tl.clamp(y, -448.0, 448.0)
    tl.store(out_ptrs, y.to(out_ptr.dtype.element_ty), mask=mask)


def fused_gdn_gated_rms_norm(
    x: torch.Tensor,
    z: torch.Tensor,
    weight: torch.Tensor,
    out: torch.Tensor,
    eps: float,
    num_v_heads: int,
    head_v_dim: int,
    norm_before_gate: bool,
    activation: str,
    scale: torch.Tensor | None = None,
) -> None:
    """Gated RMS norm ``out = rms_norm(x) * act(z)`` for Qwen3.5 GDN.

    Args:
        x: (num_tokens, num_v_heads * head_v_dim), contiguous. May alias out.
        z: (num_tokens, num_v_heads * head_v_dim) gate values, typically a
            strided view into the packed in_proj_qkvz output. Only the last
            dim must be contiguous.
        weight: (head_v_dim,) shared per-head norm weight.
        out: (num_tokens, num_v_heads * head_v_dim), contiguous.
        eps: RMS norm epsilon.
        num_v_heads: number of V heads per TP rank.
        head_v_dim: head_v_dim (per-row norm size).
        norm_before_gate: if True, out = norm(x) * act(z), else
            out = norm(x * act(z)).
        activation: "swish"/"silu" or "sigmoid".
        scale: optional static per-tensor FP8 scale; when given, out is
            float8_e4m3fn and must not alias x.

    """
    assert x.shape == z.shape == out.shape and x.dim() == 2
    assert x.stride(-1) == 1 and z.stride(-1) == 1 and out.stride(-1) == 1
    assert weight.shape == (head_v_dim,) and weight.stride(0) == 1
    assert x.shape[1] == num_v_heads * head_v_dim
    quant_fp8 = scale is not None
    if scale is not None:
        assert out.dtype == torch.float8_e4m3fn and scale.numel() == 1

    num_tokens = x.shape[0]
    if num_tokens == 0:
        return
    block_n = triton.next_power_of_2(head_v_dim)
    rows_per_block = calc_rows_per_block(num_tokens, x.device)
    grid = (triton.cdiv(num_tokens, rows_per_block), num_v_heads)
    _fused_gdn_gated_rms_norm_kernel[grid](
        x,
        z,
        weight,
        out,
        scale if quant_fp8 else weight,
        x.stride(0),
        z.stride(0),
        out.stride(0),
        num_tokens,
        eps,
        N=head_v_dim,
        BLOCK_N=block_n,
        ROWS_PER_BLOCK=rows_per_block,
        NORM_BEFORE_GATE=norm_before_gate,
        ACTIVATION=activation,
        QUANT_FP8=quant_fp8,
        num_warps=min(max(block_n // 256, 1), 8),
    )
