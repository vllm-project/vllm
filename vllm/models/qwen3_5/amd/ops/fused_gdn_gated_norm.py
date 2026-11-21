# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch

from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod
from vllm.third_party.flash_linear_attention.ops.layernorm_guard import (
    calc_rows_per_block,
)
from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import direct_register_custom_op

_SUPPORTED_ACTIVATIONS = ("swish", "silu", "sigmoid")


@triton.jit(do_not_specialize=["M"])
def _fused_gdn_gated_rms_norm_kernel(
    x_ptr,
    z_ptr,
    w_ptr,
    out_ptr,
    stride_x_tok,
    stride_z_tok,
    stride_out_tok,
    M,  # number of tokens
    eps,
    N: tl.constexpr,  # head_v_dim
    BLOCK_N: tl.constexpr,
    ROWS_PER_BLOCK: tl.constexpr,
    ACTIVATION: tl.constexpr,
):
    """FLA-equivalent per-head gated RMS norm with token-strided z."""
    # int64: rows * stride_z_tok wraps int32 at ~131K tokens (stride ~16K).
    row_start = tl.program_id(0).to(tl.int64) * ROWS_PER_BLOCK
    head = tl.program_id(1)

    rows = row_start + tl.arange(0, ROWS_PER_BLOCK)
    cols = tl.arange(0, BLOCK_N)
    row_mask = rows[:, None] < M
    col_mask = cols[None, :] < N
    mask = row_mask & col_mask

    x_ptrs = x_ptr + rows[:, None] * stride_x_tok + head * N + cols[None, :]
    z_ptrs = z_ptr + rows[:, None] * stride_z_tok + head * N + cols[None, :]

    x = tl.load(x_ptrs, mask=mask, other=0.0).to(tl.float32)

    xbar = tl.where(mask, x, 0.0)
    var = tl.sum(xbar * xbar, axis=1) / N
    rstd = tl.rsqrt(var + eps)

    w = tl.load(w_ptr + cols, mask=cols < N, other=0.0).to(tl.float32)
    y = x * rstd[:, None] * w[None, :]

    z = tl.load(z_ptrs, mask=mask, other=0.0).to(tl.float32)
    if ACTIVATION == "swish" or ACTIVATION == "silu":
        y *= z * tl.sigmoid(z)
    elif ACTIVATION == "sigmoid":
        y *= tl.sigmoid(z)

    out_ptrs = out_ptr + rows[:, None] * stride_out_tok + head * N + cols[None, :]
    tl.store(out_ptrs, y.to(out_ptr.dtype.element_ty), mask=mask)


def fused_gdn_gated_rms_norm(
    x: torch.Tensor,
    z: torch.Tensor,
    weight: torch.Tensor,
    out: torch.Tensor,
    eps: float,
    num_v_heads: int,
    head_v_dim: int,
    activation: str,
) -> None:
    """Gated RMS norm `out = rms_norm(x) * act(z)` for Qwen3.5 GDN.

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
        activation: "swish"/"silu" or "sigmoid".

    """
    assert x.shape == z.shape == out.shape and x.dim() == 2
    assert x.stride(-1) == 1 and z.stride(-1) == 1 and out.stride(-1) == 1
    assert weight.shape == (head_v_dim,) and weight.stride(0) == 1
    assert x.shape[1] == num_v_heads * head_v_dim

    num_tokens = x.shape[0]
    if num_tokens == 0:
        return
    block_n = triton.next_power_of_2(head_v_dim)
    # Same tile as rmsnorm_fn, which sees one row per (token, head), so the
    # output is bit-identical to it.
    rows_per_block = calc_rows_per_block(num_tokens * num_v_heads, x.device)
    grid = (triton.cdiv(num_tokens, rows_per_block), num_v_heads)
    _fused_gdn_gated_rms_norm_kernel[grid](
        x,
        z,
        weight,
        out,
        x.stride(0),
        z.stride(0),
        out.stride(0),
        num_tokens,
        eps,
        N=head_v_dim,
        BLOCK_N=block_n,
        ROWS_PER_BLOCK=rows_per_block,
        ACTIVATION=activation,
        num_warps=min(max(block_n // 256, 1), 8),
    )


def _rocm_gdn_gated_rms_norm(
    x: torch.Tensor,
    z: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    num_v_heads: int,
    head_v_dim: int,
    activation: str,
) -> torch.Tensor:
    out = torch.empty_like(x)
    fused_gdn_gated_rms_norm(
        x, z, weight, out, eps, num_v_heads, head_v_dim, activation
    )
    return out


def _rocm_gdn_gated_rms_norm_fake(
    x: torch.Tensor,
    z: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    num_v_heads: int,
    head_v_dim: int,
    activation: str,
) -> torch.Tensor:
    return torch.empty_like(x)


direct_register_custom_op(
    op_name="rocm_gdn_gated_rms_norm",
    op_func=_rocm_gdn_gated_rms_norm,
    fake_impl=_rocm_gdn_gated_rms_norm_fake,
)


def rocm_fused_gdn_gated_rms_norm(
    *,
    x: torch.Tensor,
    z: torch.Tensor,
    weight: torch.Tensor,
    out: torch.Tensor,
    out_proj: LinearBase,
    eps: float,
    num_v_heads: int,
    head_v_dim: int,
    norm_before_gate: bool,
    activation: str,
) -> torch.Tensor:
    """`fused_gdn_gated_rms_norm` hook for `QwenGatedDeltaNetAttention`.

    `z` is the strided gate slice of the packed qkvz projection. Quantized
    out_proj layers keep the `fla_rms_norm_gated` op so the compile-time
    norm+quant fusion patterns still match.
    """
    del out
    if not norm_before_gate:
        raise ValueError("Qwen GDN gated norm always applies the norm before the gate")
    if (
        isinstance(out_proj.quant_method, UnquantizedLinearMethod)
        and activation in _SUPPORTED_ACTIVATIONS
    ):
        return torch.ops.vllm.rocm_gdn_gated_rms_norm(
            x, z, weight, eps, num_v_heads, head_v_dim, activation
        )
    normed = torch.ops.vllm.fla_rms_norm_gated(
        x.view(-1, num_v_heads, head_v_dim),
        z.reshape(-1, num_v_heads, head_v_dim),
        weight,
        eps,
        None,
        True,
        activation,
    )
    return normed.flatten(-2)


def warmup_fused_gdn_gated_rms_norm(
    *,
    max_num_tokens: int,
    num_v_heads: int,
    head_v_dim: int,
    z_token_stride: int,
    z_offset: int,
    x_dtype: torch.dtype,
    weight_dtype: torch.dtype,
    device: torch.device,
    activation: str,
) -> None:
    """Compile the kernel for every `ROWS_PER_BLOCK` the runtime can pick.

    `z` is laid out like the real gate slice: rows `z_token_stride` apart,
    starting `z_offset` elements into the packed row.
    """
    width = num_v_heads * head_v_dim
    weight = torch.ones(head_v_dim, dtype=weight_dtype, device=device)
    seen: set[int] = set()
    for num_tokens in range(1, max_num_tokens + 1):
        rows_per_block = calc_rows_per_block(num_tokens * num_v_heads, device)
        if rows_per_block in seen:
            continue
        seen.add(rows_per_block)
        x = torch.zeros(num_tokens, width, dtype=x_dtype, device=device)
        packed = torch.zeros(num_tokens, z_token_stride, dtype=x_dtype, device=device)
        z = packed[:, z_offset : z_offset + width]
        fused_gdn_gated_rms_norm(
            x, z, weight, torch.empty_like(x), 1e-6, num_v_heads, head_v_dim, activation
        )
        if rows_per_block >= 4:
            break
