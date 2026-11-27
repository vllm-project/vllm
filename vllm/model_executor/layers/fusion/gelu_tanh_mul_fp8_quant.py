# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GeluAndMul(approximate="tanh") fused with dynamic per-token FP8 quantization.

One pass over the [T, 2d] gate/up tensor: gelu_tanh(gate) * up is computed with
the same two bf16 roundings as the ``gelu_tanh_and_mul`` kernel, the row's
absolute maximum gives the scale (amax / FP8_MAX, as AITER's
``dynamic_per_token_scaled_quant``), and the row is written as FP8. The bf16
intermediate never goes to memory.
"""

import torch

from vllm.model_executor.layers.fusion.quant_activation import QuantizedActivation
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    get_fp8_min_max,
    kFp8DynamicTokenSym,
)
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import direct_register_custom_op

_FP8_MIN, _FP8_MAX = get_fp8_min_max()
_BETA = tl.constexpr(0.7978845608028654)  # sqrt(2 / pi)
_KAPPA = tl.constexpr(0.044715)
_LOG2E = tl.constexpr(1.4426950408889634)


@triton.jit
def _rcp(x):
    return tl.inline_asm_elementwise(
        "v_rcp_f32 $0, $1",
        "=v,v",
        [x],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _gelu_tanh_mul(g, u):
    # 0.5 * g * (1 + tanh(z)) == g * sigmoid(2z) == g / (1 + exp(-2z)): one exp2
    # and one reciprocal instead of libdevice tanh (which costs ~230 us of a
    # 330 us kernel at 16384 rows). Results differ from the CUDA-style kernel
    # by a few fp32 ulp before the bf16 rounding.
    z = (-2.0 * _LOG2E * _BETA) * (g + _KAPPA * g * g * g)
    act = g * _rcp(1.0 + tl.exp2(tl.minimum(z, 126.0)))
    act = act.to(tl.bfloat16).to(tl.float32)
    return (act * u).to(tl.bfloat16).to(tl.float32)


@triton.jit
def _gelu_tanh_mul_fp8_quant_kernel(
    x_ptr,
    out_ptr,
    scale_ptr,
    d,
    stride_x,
    stride_o,
    FP8_MAX: tl.constexpr,
    B0: tl.constexpr,
    B1: tl.constexpr,
):
    # One program per row; the row is held in registers as a power-of-two tile
    # B0 plus an optional tail tile B1 (10240 = 8192 + 2048) so nothing is padded.
    row = tl.program_id(0).to(tl.int64)
    x_row = x_ptr + row * stride_x
    o_row = out_ptr + row * stride_o
    o0 = tl.arange(0, B0)
    m0 = o0 < d
    v0 = _gelu_tanh_mul(
        tl.load(x_row + o0, mask=m0, other=0.0).to(tl.float32),
        tl.load(x_row + d + o0, mask=m0, other=0.0).to(tl.float32),
    )
    amax = tl.max(tl.abs(v0), axis=0)
    if B1 > 0:
        o1 = B0 + tl.arange(0, B1)
        m1 = o1 < d
        v1 = _gelu_tanh_mul(
            tl.load(x_row + o1, mask=m1, other=0.0).to(tl.float32),
            tl.load(x_row + d + o1, mask=m1, other=0.0).to(tl.float32),
        )
        amax = tl.maximum(amax, tl.max(tl.abs(v1), axis=0))
    scale = amax * (1.0 / FP8_MAX)
    inv = tl.where(scale > 0.0, _rcp(scale), 0.0)
    q0 = tl.minimum(tl.maximum(v0 * inv, -FP8_MAX), FP8_MAX)
    tl.store(o_row + o0, q0.to(out_ptr.dtype.element_ty), mask=m0)
    if B1 > 0:
        q1 = tl.minimum(tl.maximum(v1 * inv, -FP8_MAX), FP8_MAX)
        tl.store(o_row + o1, q1.to(out_ptr.dtype.element_ty), mask=m1)
    tl.store(scale_ptr + row, scale)


def _gelu_tanh_mul_fp8_per_token_quant_impl(
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    assert x.is_contiguous() and x.dtype == torch.bfloat16
    num_tokens = x.shape[0]
    d = x.shape[-1] // 2
    out = torch.empty(
        (num_tokens, d), dtype=current_platform.fp8_dtype(), device=x.device
    )
    scale = torch.empty((num_tokens, 1), dtype=torch.float32, device=x.device)
    if num_tokens == 0:
        return out, scale
    b0 = 1 << (d.bit_length() - 1)
    b1 = triton.next_power_of_2(d - b0) if d > b0 else 0
    _gelu_tanh_mul_fp8_quant_kernel[(num_tokens,)](
        x,
        out,
        scale,
        d,
        x.stride(0),
        out.stride(0),
        FP8_MAX=_FP8_MAX,
        B0=b0,
        B1=b1,
        num_warps=4,
    )
    return out, scale


def _gelu_tanh_mul_fp8_per_token_quant_fake(
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    d = x.shape[-1] // 2
    return (
        torch.empty((x.shape[0], d), dtype=current_platform.fp8_dtype(), device=x.device),
        torch.empty((x.shape[0], 1), dtype=torch.float32, device=x.device),
    )


direct_register_custom_op(
    op_name="gelu_tanh_mul_fp8_per_token_quant",
    op_func=_gelu_tanh_mul_fp8_per_token_quant_impl,
    mutates_args=[],
    fake_impl=_gelu_tanh_mul_fp8_per_token_quant_fake,
)


def gelu_tanh_mul_fp8_per_token_quant(x: torch.Tensor, linear) -> QuantizedActivation:
    """gelu_tanh(x[..., :d]) * x[..., d:] quantized to FP8 with per-token scales."""
    orig_shape = x.shape[:-1] + (x.shape[-1] // 2,)
    q, scale = torch.ops.vllm.gelu_tanh_mul_fp8_per_token_quant(x.view(-1, x.shape[-1]))
    return QuantizedActivation(
        data=q.view(orig_shape),
        scale=scale,
        orig_dtype=x.dtype,
        orig_shape=torch.Size(orig_shape),
        quant_key=kFp8DynamicTokenSym,
    )
