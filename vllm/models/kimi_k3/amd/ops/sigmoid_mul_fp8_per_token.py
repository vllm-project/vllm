# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fused ``attn_out * sigmoid(gate)`` + per-token FP8 for Kimi-K3 MLA ``o_proj``.

One Triton kernel writes the PTPC pair ``y [T, K] fp8``, ``scale [T, 1] float32``.
Fusion is a no-op unless ``o_proj.input_quant_key`` is ``kFp8DynamicTokenSym``.
Do not use this for group-128 or MXFP4; those GEMMs need a different scale layout.
``g_proj`` is not fused: it produces the gate, not the tensor ``o_proj`` quantizes.
"""

from __future__ import annotations

import torch

from vllm.model_executor.layers.fusion.quant_activation import QuantizedActivation
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    get_fp8_min_max,
    kFp8DynamicTokenSym,
)
from vllm.platforms import current_platform
from vllm.triton_utils import HAS_TRITON

# Triton BLOCK is next_power_of_2(K); above this, fall back to the torch path.
_MAX_K = 8192

_sigmoid_mul_fp8_per_token_kernel = None
_next_power_of_2 = None
if HAS_TRITON:
    from vllm.triton_utils import tl, triton

    _next_power_of_2 = triton.next_power_of_2

    @triton.jit
    def _kernel(
        x_ptr,
        g_ptr,
        y_ptr,
        s_ptr,
        K,
        fp8_max,
        stride_x,
        stride_g,
        stride_y,
        BLOCK: tl.constexpr,
    ):
        tok = tl.program_id(0)
        cols = tl.arange(0, BLOCK)
        mask = cols < K
        x = tl.load(x_ptr + tok * stride_x + cols, mask=mask, other=0.0).to(tl.float32)
        g = tl.load(g_ptr + tok * stride_g + cols, mask=mask, other=0.0).to(tl.float32)
        y = x * tl.sigmoid(g)
        amax = tl.max(tl.abs(y))
        scale = amax / fp8_max
        inv = tl.where(scale > 0.0, 1.0 / scale, 0.0)
        q = y * inv
        q = tl.minimum(tl.maximum(q, -fp8_max), fp8_max)
        tl.store(y_ptr + tok * stride_y + cols, q.to(y_ptr.dtype.element_ty), mask=mask)
        tl.store(s_ptr + tok, scale)

    _sigmoid_mul_fp8_per_token_kernel = _kernel


def _sigmoid_mul_fp8_torch(
    x: torch.Tensor,
    gate: torch.Tensor,
    quant_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference: fp32 sigmoid-mul, then per-token amax FP8.

    Matches the Triton kernel: scale is ``amax / fp8_max``, and a zero amax
    writes a zero scale with a zero quantized row (no 1e-12 floor).
    """
    fp8_min, fp8_max = get_fp8_min_max()
    gated = x.float() * torch.sigmoid(gate.float())
    amax = gated.abs().amax(dim=-1, keepdim=True)
    scale = amax / float(fp8_max)
    inv = torch.where(scale > 0, 1.0 / scale, torch.zeros_like(scale))
    q = (gated * inv).clamp(fp8_min, fp8_max).to(quant_dtype)
    return q, scale


def sigmoid_mul_fp8_per_token(
    x: torch.Tensor,
    gate: torch.Tensor,
    quant_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``x * sigmoid(gate)`` fused with per-token FP8.

    ``x`` and ``gate`` are matching ``[T, K]``. Returns
    ``(y [T, K] fp8, scale [T, 1] float32)``.
    """
    if x.ndim != 2 or gate.shape != x.shape:
        raise ValueError(
            "expected matching 2D (T, K), "
            f"got x={tuple(x.shape)} gate={tuple(gate.shape)}"
        )
    t, k = x.shape
    _, fp8_max = get_fp8_min_max()
    if not HAS_TRITON or t == 0 or k > _MAX_K:
        if t == 0:
            return (
                torch.empty((0, k), dtype=quant_dtype, device=x.device),
                torch.empty((0, 1), dtype=torch.float32, device=x.device),
            )
        return _sigmoid_mul_fp8_torch(x, gate, quant_dtype)

    x = x.contiguous()
    gate = gate.contiguous()
    out = torch.empty((t, k), dtype=quant_dtype, device=x.device)
    scale = torch.empty((t, 1), dtype=torch.float32, device=x.device)
    assert _sigmoid_mul_fp8_per_token_kernel is not None
    assert _next_power_of_2 is not None

    _sigmoid_mul_fp8_per_token_kernel[(t,)](
        x,
        gate,
        out,
        scale,
        k,
        float(fp8_max),
        x.stride(0),
        gate.stride(0),
        out.stride(0),
        BLOCK=_next_power_of_2(k),
    )
    return out, scale


def o_proj_is_ptpc_fp8(o_proj: torch.nn.Module) -> bool:
    """True when o_proj advertised the per-token FP8 consumer ABI.
    """
    return getattr(o_proj, "input_quant_key", None) == kFp8DynamicTokenSym


def wrap_ptpc_activation(
    data: torch.Tensor,
    scale: torch.Tensor,
    orig_dtype: torch.dtype,
    orig_shape: torch.Size,
) -> QuantizedActivation:
    return QuantizedActivation(
        data=data,
        scale=scale,
        orig_dtype=orig_dtype,
        orig_shape=orig_shape,
        quant_key=kFp8DynamicTokenSym,
    )


def maybe_fused_mla_oproj_ptpc(
    attn_out: torch.Tensor,
    gate: torch.Tensor,
    o_proj: torch.nn.Module,
) -> QuantizedActivation | None:
    """MLA sigmoid gate + PTPC FP8, or None to keep the BF16 expression.
    """
    if not o_proj_is_ptpc_fp8(o_proj):
        return None
    if not HAS_TRITON:
        return None
    if attn_out.ndim != 2 or gate.shape != attn_out.shape:
        return None

    data, scale = sigmoid_mul_fp8_per_token(
        attn_out, gate, current_platform.fp8_dtype()
    )
    return wrap_ptpc_activation(data, scale, attn_out.dtype, attn_out.shape)
