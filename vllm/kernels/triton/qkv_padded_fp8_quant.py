# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Stride-aware FP8 quantization with head_dim padding for ViT attention.

Reads directly from non-contiguous QKV views using 3D strides and pads
head_dim to a multiple of ``pad_to``: 16 for cuDNN, 128 for AITER's asm FP8
kernel on ROCm.
"""

import torch

from vllm.model_executor.layers.quantization.input_quant_fp8 import QuantFP8
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    get_fp8_min_max,
)
from vllm.platforms import current_platform
from vllm.triton_utils import HAS_TRITON, tl, triton
from vllm.utils.math_utils import round_up

_FP8_MIN, _FP8_MAX = get_fp8_min_max()


@triton.jit
def _quantize_pad_fp8_kernel(
    x_ptr,
    y_ptr,
    scale_ptr,
    stride_xs,
    stride_xh,
    stride_xd,
    stride_ys,
    stride_yh,
    stride_yd,
    num_heads,
    n_rows,
    n_cols,
    n_cols_padded,
    fp8_min: tl.constexpr,
    fp8_max: tl.constexpr,
    SKIP_SCALE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_m = offs_m < n_rows
    mask_out = mask_m[:, None] & (offs_n[None, :] < n_cols_padded)
    mask_in = mask_m[:, None] & (offs_n[None, :] < n_cols)

    # Decompose flattened row into (token, head) for 3D stride indexing.
    s = offs_m // num_heads
    h = offs_m % num_heads

    x_ptrs = (
        x_ptr
        + s[:, None] * stride_xs
        + h[:, None] * stride_xh
        + offs_n[None, :] * stride_xd
    )
    x = tl.load(x_ptrs, mask=mask_in, other=0.0).to(tl.float32)
    if SKIP_SCALE:
        x_q = x
    else:
        scale = tl.load(scale_ptr)
        x_q = x / scale
    x_q = tl.clamp(x_q, fp8_min, fp8_max).to(y_ptr.dtype.element_ty)

    y_ptrs = (
        y_ptr
        + s[:, None] * stride_ys
        + h[:, None] * stride_yh
        + offs_n[None, :] * stride_yd
    )
    tl.store(y_ptrs, x_q, mask=mask_out)


def _get_fp8_pad_quant_config(padded_head_dim: int) -> tuple[int, int, int]:
    block_n = triton.next_power_of_2(padded_head_dim)
    block_n = max(16, min(block_n, 128))
    block_m = 16
    num_warps = 4
    return block_m, block_n, num_warps


def quantize_fp8_pad_head_dim_triton(
    tensor: torch.Tensor,
    scale: torch.Tensor,
    skip_scale: bool = False,
    block_m: int | None = None,
    block_n: int | None = None,
    num_warps: int | None = None,
    pad_to: int = 16,
) -> torch.Tensor:
    """Quantize a 3D/4D tensor to FP8, padding head_dim to a multiple of
    ``pad_to`` (16 by default, for cuDNN).

    Reads directly from the input using its 3D strides, so non-contiguous
    views (e.g. Q/K/V slices from an interleaved QKV buffer) are handled
    without an extra copy.  Output is always a fresh contiguous tensor
    with shape (S, H, padded_D).
    """
    if not HAS_TRITON:
        raise RuntimeError("Triton is required to quantize with head_dim padding.")

    original_shape = tensor.shape
    if tensor.dim() == 4:
        tensor = tensor.view(-1, tensor.shape[-2], tensor.shape[-1])
    assert tensor.dim() == 3, f"Expected 3D input (S, H, D), got {tensor.dim()}D"
    S, H, D = tensor.shape
    padded_head_dim = round_up(D, pad_to)
    out_dtype = current_platform.fp8_dtype()
    output = torch.empty(
        (S, H, padded_head_dim),
        device=tensor.device,
        dtype=out_dtype,
    )

    scale_1d = scale.reshape(-1)
    n_rows = S * H

    if block_m is None or block_n is None or num_warps is None:
        block_m, block_n, num_warps = _get_fp8_pad_quant_config(padded_head_dim)

    grid = (
        triton.cdiv(n_rows, block_m),
        triton.cdiv(padded_head_dim, block_n),
    )

    _quantize_pad_fp8_kernel[grid](
        tensor,
        output,
        scale_1d,
        tensor.stride(0),
        tensor.stride(1),
        tensor.stride(2),
        output.stride(0),
        output.stride(1),
        output.stride(2),
        H,
        n_rows,
        D,
        padded_head_dim,
        _FP8_MIN,
        _FP8_MAX,
        SKIP_SCALE=skip_scale,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        num_warps=num_warps,
    )

    return output.view((*original_shape[:-1], padded_head_dim))


def quantize_fp8_maybe_pad_head_dim(
    tensor: torch.Tensor,
    scale: torch.Tensor,
    fp8_quant: QuantFP8,
    skip_scale: bool = False,
    pad_to: int = 16,
) -> torch.Tensor:
    """Quantize a 3D/4D tensor to FP8, padding head_dim to a multiple of
    ``pad_to`` (16 by default) only when needed.

    Accepts (S, H, D) or (B, S, H, D) input. Uses `fp8_quant` (a
    `QuantFP8` CustomOp) when head_dim is already a multiple of
    `pad_to` (no padding); otherwise falls back to a stride-aware Triton
    kernel that pads head_dim to a multiple of `pad_to`.
    """
    head_dim = tensor.shape[-1]
    if head_dim % pad_to != 0:
        return quantize_fp8_pad_head_dim_triton(
            tensor, scale, skip_scale=skip_scale, pad_to=pad_to
        )

    if skip_scale:
        return tensor.to(current_platform.fp8_dtype())

    # QuantFP8 expects 2D: flatten all dims except (H, D).
    orig_shape = tensor.shape
    total_tokens = tensor.numel() // (orig_shape[-1] * orig_shape[-2])
    tensor_2d = tensor.reshape(total_tokens, -1)
    fp8_tensor, _ = fp8_quant(tensor_2d, scale=scale)
    return fp8_tensor.reshape(orig_shape)


# Dynamic per-tensor scales for Q, K and V in two launches instead of ~7 small
# torch ops per tensor plus three quantize launches. Same math as
# ``MMEncoderAttention._record_amax_and_update_scales`` followed by
# ``quantize_fp8_pad_head_dim_triton``: the amax is the signed max
# (``Tensor.amax()``), the history max includes the new entry, and the scale is
# ``clamp(max, tiny) * fp32(1 / fp8_max)`` (PyTorch divides by a Python scalar
# as a multiply by its fp32 reciprocal), so outputs and scales are
# bit-identical to the unfused path.
_QKV_MAX_PARTIALS = 256
_INV_FP8_MAX = float(
    torch.tensor(1.0, dtype=torch.float32) / torch.tensor(_FP8_MAX, dtype=torch.float32)
)


@triton.jit
def _qkv_partial_max_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    partial_ptr,
    stride_qs,
    stride_qh,
    stride_qd,
    stride_ks,
    stride_kh,
    stride_kd,
    stride_vs,
    stride_vh,
    stride_vd,
    num_heads,
    n_rows,
    n_cols,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # Each program reduces a strided set of (token, head) rows of Q, K and V
    # to one max per tensor: partial[pid, 0:3].
    pid = tl.program_id(0)
    offs_n = tl.arange(0, BLOCK_N)
    d = offs_n[None, :]
    mask_n = d < n_cols
    acc_q = tl.full((BLOCK_M, BLOCK_N), float("-inf"), tl.float32)
    acc_k = tl.full((BLOCK_M, BLOCK_N), float("-inf"), tl.float32)
    acc_v = tl.full((BLOCK_M, BLOCK_N), float("-inf"), tl.float32)
    for start in range(pid * BLOCK_M, n_rows, tl.num_programs(0) * BLOCK_M):
        offs_m = start + tl.arange(0, BLOCK_M)
        mask = (offs_m < n_rows)[:, None] & mask_n
        s = (offs_m // num_heads)[:, None]
        h = (offs_m % num_heads)[:, None]
        q = tl.load(
            q_ptr + s * stride_qs + h * stride_qh + d * stride_qd,
            mask=mask,
            other=float("-inf"),
        )
        k = tl.load(
            k_ptr + s * stride_ks + h * stride_kh + d * stride_kd,
            mask=mask,
            other=float("-inf"),
        )
        v = tl.load(
            v_ptr + s * stride_vs + h * stride_vh + d * stride_vd,
            mask=mask,
            other=float("-inf"),
        )
        acc_q = tl.maximum(acc_q, q.to(tl.float32))
        acc_k = tl.maximum(acc_k, k.to(tl.float32))
        acc_v = tl.maximum(acc_v, v.to(tl.float32))
    tl.store(partial_ptr + pid * 3, tl.max(acc_q))
    tl.store(partial_ptr + pid * 3 + 1, tl.max(acc_k))
    tl.store(partial_ptr + pid * 3 + 2, tl.max(acc_v))


@triton.jit
def _amax_and_scale(
    partial_ptr,
    n_partials,
    IDX: tl.constexpr,
    hist_ptr,
    pos,
    tiny,
    inv_fp8_max,
    P_BLOCK: tl.constexpr,
    HIST_LEN: tl.constexpr,
):
    # This call's amax and the scale from the history with it at ``pos``.
    offs_p = tl.arange(0, P_BLOCK)
    amax = tl.max(
        tl.load(
            partial_ptr + offs_p * 3 + IDX,
            mask=offs_p < n_partials,
            other=float("-inf"),
        )
    )
    offs_h = tl.arange(0, HIST_LEN)
    hist = tl.load(hist_ptr + offs_h, mask=offs_h != pos, other=float("-inf"))
    scale = tl.maximum(tl.maximum(amax, tl.max(hist)), tiny) * inv_fp8_max
    return amax, scale


@triton.jit
def _quantize_pad_tile(
    x_ptr,
    y_ptr,
    scale,
    s,
    h,
    d,
    mask_in,
    mask_out,
    stride_xs,
    stride_xh,
    stride_xd,
    stride_ys,
    stride_yh,
    stride_yd,
    fp8_min: tl.constexpr,
    fp8_max: tl.constexpr,
):
    x = tl.load(
        x_ptr + s * stride_xs + h * stride_xh + d * stride_xd,
        mask=mask_in,
        other=0.0,
    ).to(tl.float32)
    x_q = tl.clamp(x / scale, fp8_min, fp8_max).to(y_ptr.dtype.element_ty)
    tl.store(y_ptr + s * stride_ys + h * stride_yh + d * stride_yd, x_q, mask=mask_out)


@triton.jit(do_not_specialize=["n_partials", "pos"])
def _qkv_dynamic_quant_pad_fp8_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    yq_ptr,
    yk_ptr,
    yv_ptr,
    partial_ptr,
    n_partials,
    q_hist_ptr,
    k_hist_ptr,
    v_hist_ptr,
    q_scale_ptr,
    k_scale_ptr,
    v_scale_ptr,
    pos,
    stride_qs,
    stride_qh,
    stride_qd,
    stride_ks,
    stride_kh,
    stride_kd,
    stride_vs,
    stride_vh,
    stride_vd,
    stride_ys,
    stride_yh,
    stride_yd,
    num_heads,
    n_rows,
    n_cols,
    n_cols_padded,
    tiny,
    inv_fp8_max,
    fp8_min: tl.constexpr,
    fp8_max: tl.constexpr,
    P_BLOCK: tl.constexpr,
    HIST_LEN: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    amax_q, scale_q = _amax_and_scale(
        partial_ptr, n_partials, 0, q_hist_ptr, pos,
        tiny, inv_fp8_max, P_BLOCK, HIST_LEN,
    )  # fmt: skip
    amax_k, scale_k = _amax_and_scale(
        partial_ptr, n_partials, 1, k_hist_ptr, pos,
        tiny, inv_fp8_max, P_BLOCK, HIST_LEN,
    )  # fmt: skip
    amax_v, scale_v = _amax_and_scale(
        partial_ptr, n_partials, 2, v_hist_ptr, pos,
        tiny, inv_fp8_max, P_BLOCK, HIST_LEN,
    )  # fmt: skip
    # Other programs read every history slot but ``pos``, so one program can
    # record this call's amax and publish the scales without a race.
    if (pid_m == 0) & (pid_n == 0):
        tl.store(q_hist_ptr + pos, amax_q)
        tl.store(k_hist_ptr + pos, amax_k)
        tl.store(v_hist_ptr + pos, amax_v)
        tl.store(q_scale_ptr, scale_q)
        tl.store(k_scale_ptr, scale_k)
        tl.store(v_scale_ptr, scale_v)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_m = offs_m < n_rows
    mask_out = mask_m[:, None] & (offs_n[None, :] < n_cols_padded)
    mask_in = mask_m[:, None] & (offs_n[None, :] < n_cols)
    s = (offs_m // num_heads)[:, None]
    h = (offs_m % num_heads)[:, None]
    d = offs_n[None, :]
    _quantize_pad_tile(
        q_ptr, yq_ptr, scale_q, s, h, d, mask_in, mask_out,
        stride_qs, stride_qh, stride_qd, stride_ys, stride_yh, stride_yd,
        fp8_min, fp8_max,
    )  # fmt: skip
    _quantize_pad_tile(
        k_ptr, yk_ptr, scale_k, s, h, d, mask_in, mask_out,
        stride_ks, stride_kh, stride_kd, stride_ys, stride_yh, stride_yd,
        fp8_min, fp8_max,
    )  # fmt: skip
    _quantize_pad_tile(
        v_ptr, yv_ptr, scale_v, s, h, d, mask_in, mask_out,
        stride_vs, stride_vh, stride_vd, stride_ys, stride_yh, stride_yd,
        fp8_min, fp8_max,
    )  # fmt: skip


def dynamic_quantize_pad_qkv_fp8_triton(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    amax_history: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    scales: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    pos: int,
    pad_to: int = 16,
    max_partials: int = _QKV_MAX_PARTIALS,
    block_m: int = 64,
    partial_block_m: int = 64,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Dynamic per-tensor FP8 quantization of Q, K and V with head_dim padding.

    Records each tensor's amax at ``amax_history[i][pos]``, writes
    ``scales[i]`` from the history max and returns the quantized, padded
    tensors, like the unfused path, in two kernel launches. Q, K and V must
    have the same 3D/4D shape; they may be strided views of one QKV buffer.
    """
    if not HAS_TRITON:
        raise RuntimeError("Triton is required to quantize with head_dim padding.")
    shape = query.shape
    assert key.shape == shape and value.shape == shape, (shape, key.shape, value.shape)
    xs = [
        t.view(-1, shape[-2], shape[-1]) if t.dim() == 4 else t
        for t in (query, key, value)
    ]
    assert xs[0].dim() == 3, f"Expected 3D input (S, H, D), got {query.dim()}D"
    S, H, D = xs[0].shape
    hist_len = amax_history[0].numel()
    assert hist_len == triton.next_power_of_2(hist_len)
    padded_head_dim = round_up(D, pad_to)
    out_dtype = current_platform.fp8_dtype()
    ys = [
        torch.empty((S, H, padded_head_dim), device=query.device, dtype=out_dtype)
        for _ in range(3)
    ]
    n_rows = S * H
    strides = [st for x in xs for st in x.stride()]

    n_partials = max(1, min(max_partials, triton.cdiv(n_rows, partial_block_m)))
    partial = torch.empty((n_partials, 3), device=query.device, dtype=torch.float32)
    _qkv_partial_max_kernel[(n_partials,)](
        *xs,
        partial,
        *strides,
        H,
        n_rows,
        D,
        BLOCK_M=partial_block_m,
        BLOCK_N=max(16, triton.next_power_of_2(D)),
        num_warps=4,
    )

    # 64-row tiles of all three tensors per program: 4.3x less GPU time than
    # the unfused path at 4144 tokens x 16 heads on MI350X (16 rows: 3.7x).
    _, bn, num_warps = _get_fp8_pad_quant_config(padded_head_dim)
    grid = (triton.cdiv(n_rows, block_m), triton.cdiv(padded_head_dim, bn))
    _qkv_dynamic_quant_pad_fp8_kernel[grid](
        *xs,
        *ys,
        partial,
        n_partials,
        *amax_history,
        *(sc.reshape(-1) for sc in scales),
        pos,
        *strides,
        *ys[0].stride(),
        H,
        n_rows,
        D,
        padded_head_dim,
        torch.finfo(torch.float32).tiny,
        _INV_FP8_MAX,
        _FP8_MIN,
        _FP8_MAX,
        P_BLOCK=max(16, triton.next_power_of_2(max_partials)),
        HIST_LEN=hist_len,
        BLOCK_M=block_m,
        BLOCK_N=bn,
        num_warps=num_warps,
    )
    return tuple(y.view((*shape[:-1], padded_head_dim)) for y in ys)
