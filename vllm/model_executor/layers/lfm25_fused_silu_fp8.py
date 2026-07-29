# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""LFM2.5 SwiGLU + per-token FP8 quantization fusion.

Stock vLLM: silu_and_mul writes BF16 activation, then dynamic per-token FP8
quant, then CUTLASS FP8 down-projection.  This module fuses all three into one
Triton launch, saving ~4 MiB intermediate traffic per token per layer.
"""

import logging
import os

import torch

from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton

FUSED_SILU_FP8_ENABLED = os.getenv("VLLM_LFM25_FUSED_SILU_FP8", "0") == "1"
if FUSED_SILU_FP8_ENABLED:
    logging.getLogger(__name__).info("[LFM2.5] SiLU+FP8 quant fusion ENABLED")


@triton.jit
def _lfm25_fused_silu_fp8_kernel(
    input_ptr,
    output_ptr,
    scale_ptr,
    input_stride_token: tl.constexpr,
    output_stride_token: tl.constexpr,
    intermediate_size: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    INPUT_DTYPE: tl.constexpr,
    TEST_MATH_ONLY: tl.constexpr,
):
    pid = tl.program_id(0)
    off = tl.arange(0, BLOCK_SIZE)
    m = off < intermediate_size
    ib = input_ptr + pid * input_stride_token

    gate = tl.load(ib + off, mask=m, other=0.0).to(tl.float32)
    up = tl.load(ib + intermediate_size + off, mask=m, other=0.0).to(tl.float32)

    sg = (gate / (1.0 + tl.exp(-gate))).to(INPUT_DTYPE).to(tl.float32)
    act = (sg * up).to(INPUT_DTYPE).to(tl.float32)
    ax = tl.max(tl.where(m, tl.abs(act), 0.0), axis=0)
    sc = tl.maximum(ax * (1.0 / 448.0), 1.0 / (448.0 * 512.0))
    val = act / sc
    rnd = (
        tl.math.llrint(val)
        if hasattr(tl.math, "llrint")
        else (
            tl.where(val >= 0.0, (val + 0.5).to(tl.int32), (val - 0.5).to(tl.int32)).to(
                tl.float32
            )
        )
    )
    q = tl.maximum(tl.minimum(rnd, 448.0), -448.0)

    tl.store(
        output_ptr + pid * output_stride_token + off,
        act if TEST_MATH_ONLY else q,
        mask=m,
    )
    tl.store(scale_ptr + pid, sc)


def fused_lfm25_silu_fp8_quant(gate_up):
    if not gate_up.is_cuda or gate_up.ndim != 2 or gate_up.shape[1] % 2:
        raise ValueError("gate_up must be a contiguous CUDA [n, 2*d] tensor")
    if gate_up.dtype not in (torch.bfloat16, torch.float16):
        raise ValueError("gate_up must use BF16/FP16")

    n = gate_up.shape[0]
    d = gate_up.shape[1] // 2
    out = torch.empty((n, d), dtype=current_platform.fp8_dtype(), device=gate_up.device)
    sc = torch.empty((n, 1), dtype=torch.float32, device=gate_up.device)
    if n == 0:
        return out, sc

    bs = triton.next_power_of_2(d)
    _lfm25_fused_silu_fp8_kernel[(n,)](
        gate_up,
        out,
        sc,
        gate_up.stride(0),
        out.stride(0),
        d,
        BLOCK_SIZE=bs,
        INPUT_DTYPE=tl.bfloat16 if gate_up.dtype == torch.bfloat16 else tl.float16,
        TEST_MATH_ONLY=False,
        num_warps=8,
        num_stages=2,
    )
    return out, sc


def _fused_lfm25_silu_math_for_test(gate_up):
    """Kernel math-only path (no FP8 store). Used as reference in tests."""
    if not gate_up.is_cuda or gate_up.ndim != 2 or gate_up.shape[1] % 2:
        raise ValueError("gate_up must be a CUDA [n, 2*d] tensor")
    n, d = gate_up.shape[0], gate_up.shape[1] // 2
    act = torch.empty((n, d), dtype=gate_up.dtype, device=gate_up.device)
    sc = torch.empty((n, 1), dtype=torch.float32, device=gate_up.device)
    if n == 0:
        return act, sc
    bs = triton.next_power_of_2(d)
    _lfm25_fused_silu_fp8_kernel[(n,)](
        gate_up,
        act,
        sc,
        gate_up.stride(0),
        act.stride(0),
        d,
        BLOCK_SIZE=bs,
        INPUT_DTYPE=tl.bfloat16 if gate_up.dtype == torch.bfloat16 else tl.float16,
        TEST_MATH_ONLY=True,
        num_warps=8,
        num_stages=2,
    )
    return act, sc


def supports_fused_lfm25_silu_fp8_linear(linear):
    qm = getattr(linear, "quant_method", None)
    fl = getattr(qm, "fp8_linear", None) if qm else None
    return bool(
        FUSED_SILU_FP8_ENABLED
        and getattr(linear, "tp_size", None) == 1
        and getattr(linear, "bias", None) is None
        and qm is not None
        and qm.__class__.__name__ == "Fp8PerTensorOnlineLinearMethod"
        and fl is not None
        and fl.__class__.__name__ == "CutlassFP8ScaledMMLinearKernel"
    )


def fused_lfm25_silu_fp8_linear(gate_up, linear):
    if not supports_fused_lfm25_silu_fp8_linear(linear):
        raise ValueError("linear does not support LFM2.5 SiLU+FP8 fusion")
    q, s = fused_lfm25_silu_fp8_quant(gate_up)
    fl = linear.quant_method.fp8_linear
    return fl.apply_scaled_mm(
        A=q,
        B=linear.weight,
        out_dtype=gate_up.dtype,
        As=s,
        Bs=linear.weight_scale,
        bias=None,
        output_shape=[*gate_up.shape[:-1], linear.weight.shape[1]],
    )
