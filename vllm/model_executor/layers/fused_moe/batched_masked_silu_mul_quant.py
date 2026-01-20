# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fused SiLU + Mul + FP8 Quantization kernel.

This module provides a fused kernel that combines:
1. SiLU activation on the first half of the hidden dimension
2. Element-wise multiplication with the second half (gated)
3. FP8 quantization with per-group scales

The kernel is used by both BatchedDeepGemmExperts and BatchedTritonExperts
for efficient activation and quantization in MoE layers.
"""

import torch

from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.utils.deep_gemm import DeepGemmQuantScaleFMT
from vllm.utils.math_utils import cdiv


def scales_shape_stride_dtype(
    E: int, T: int, G: int, quant_scale_fmt: DeepGemmQuantScaleFMT
) -> tuple[tuple[int, ...], tuple[int, ...], torch.dtype]:
    """Compute shape, strides, and dtype for quantization scales tensor.

    Args:
        E: Number of experts
        T: Max tokens per expert
        G: Number of groups (H // group_size)
        quant_scale_fmt: Scale format (FLOAT32, FLOAT32_CEIL_UE8M0, or UE8M0)

    Returns:
        Tuple of (shape, strides, dtype) for the scales tensor
    """
    shape = (E, T, G)
    strides = (T * G, 1, T)
    if quant_scale_fmt in [
        DeepGemmQuantScaleFMT.FLOAT32,
        DeepGemmQuantScaleFMT.FLOAT32_CEIL_UE8M0,
    ]:
        return shape, strides, torch.float32

    assert quant_scale_fmt == DeepGemmQuantScaleFMT.UE8M0
    shape = (E, T, cdiv(G, 4))
    strides = (T * cdiv(G, 4), 1, T)
    return shape, strides, torch.int32


@triton.jit
def _silu_mul_fp8_quant_kernel(
    # Pointers ------------------------------------------------------------
    input_ptr,  # 16-bit activations (E, T, 2*H)
    y_q_ptr,  # fp8 quantized activations (E, T, H)
    y_s_ptr,  # 16-bit scales (E, T, G)
    counts_ptr,  # int32 num tokens per expert (E)
    # Sizes ---------------------------------------------------------------
    H: tl.constexpr,  # hidden dimension (per output)
    GROUP_SIZE: tl.constexpr,  # elements per group (usually 128)
    # Strides for input (elements) ---------------------------------------
    stride_i_e,
    stride_i_t,
    stride_i_h,
    # Strides for y_q (elements) -----------------------------------------
    stride_yq_e,
    stride_yq_t,
    stride_yq_h,
    # Strides for y_s (elements) -----------------------------------------
    stride_ys_e,
    stride_ys_t,
    stride_ys_g,
    # Stride for counts (elements)
    stride_counts_e,
    # Numeric params ------------------------------------------------------
    eps: tl.constexpr,
    fp8_min: tl.constexpr,
    fp8_max: tl.constexpr,
    ceil_ue8m0: tl.constexpr,
    # Meta ---------------------------------------------------------------
    BLOCK: tl.constexpr,
    NUM_STAGES: tl.constexpr,
):
    """Triton kernel for fused SiLU + mul + FP8 quantization.

    For each expert and group, this kernel:
    1. Loads gate and up projections from input (shape: E, T, 2*H)
    2. Applies SiLU: gate = gate * sigmoid(gate)
    3. Applies gated multiplication: y = gate * up
    4. Quantizes to FP8 with per-group scales
    """
    G = H // GROUP_SIZE

    # map program id -> (e, g)
    pid = tl.program_id(0)
    e = pid // G
    g = pid % G

    e = e.to(tl.int64)
    g = g.to(tl.int64)

    # number of valid tokens for this expert
    n_tokens = tl.load(counts_ptr + e * stride_counts_e).to(tl.int64)

    cols = tl.arange(0, BLOCK).to(tl.int64)
    mask = cols < BLOCK

    base_input_offset = e * stride_i_e + g * GROUP_SIZE * stride_i_h
    base_gate_offset = base_input_offset + cols * stride_i_h
    base_up_offset = base_input_offset + H * stride_i_h + cols * stride_i_h
    base_yq_offset = e * stride_yq_e + g * GROUP_SIZE * stride_yq_h + cols * stride_yq_h
    base_ys_offset = e * stride_ys_e + g * stride_ys_g

    for t in tl.range(0, n_tokens, num_stages=NUM_STAGES):
        gate = tl.load(
            input_ptr + base_gate_offset + t * stride_i_t, mask=mask, other=0.0
        ).to(tl.float32)
        up = tl.load(input_ptr + base_up_offset + t * stride_i_t, mask=mask, other=0.0)

        # SiLU activation: gate * sigmoid(gate)
        gate = gate * (1.0 / (1.0 + tl.exp(-gate)))
        # Gated multiplication
        y = gate * up

        # Compute per-group scale
        y_s = tl.maximum(tl.max(tl.abs(y)), eps) / fp8_max
        if ceil_ue8m0:
            # Round scale to power of 2 for UE8M0 format
            y_s = tl.exp2(tl.ceil(tl.log2(y_s)))

        # Quantize to FP8
        y_q = tl.clamp(y / y_s, fp8_min, fp8_max).to(y_q_ptr.dtype.element_ty)

        tl.store(y_q_ptr + base_yq_offset + t * stride_yq_t, y_q, mask=mask)
        tl.store(y_s_ptr + base_ys_offset + t * stride_ys_t, y_s)


def silu_mul_fp8_quant(
    y: torch.Tensor,  # (E, T, 2*H)
    tokens_per_expert: torch.Tensor,  # (E,) number of valid tokens per expert
    group_size: int = 128,
    quant_scale_fmt: DeepGemmQuantScaleFMT = DeepGemmQuantScaleFMT.FLOAT32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused SiLU activation + gated multiplication + FP8 quantization.

    Computes: quantize(silu(y[..., :H]) * y[..., H:])

    y has shape (E, T, 2*H). The first half of the last dimension is
    silu-activated, multiplied by the second half, then quantized into FP8.

    On CUDA SM80+ devices, uses an optimized CUDA kernel.
    On other platforms (including ROCm), uses a Triton kernel.

    Args:
        y: Input tensor of shape (E, T, 2*H) where:
           - E = number of experts
           - T = max tokens per expert
           - 2*H = gate and up projections concatenated
        tokens_per_expert: Number of valid tokens per expert, shape (E,)
        group_size: Quantization group size (default 128)
        quant_scale_fmt: Scale format for quantization

    Returns:
        Tuple of (y_q, y_s) where:
        - y_q: FP8 tensor, shape (E, T, H)
        - y_s: Scales tensor, shape and dtype depend on quant_scale_fmt:
          - FLOAT32: FP32 tensor, shape (E, T, G), strides (T*G, 1, T)
          - FLOAT32_CEIL_UE8M0: FP32 tensor, shape (E, T, G), strides (T*G, 1, T)
          - UE8M0: Int32 tensor, shape (E, T, G//4), strides (T*G//4, 1, T)
    """
    assert y.ndim == 3, "y must be (E, T, 2*H)"
    E, T, H2 = y.shape
    assert H2 % 2 == 0, "last dim of y must be even (2*H)"
    H = H2 // 2
    G = (H + group_size - 1) // group_size
    assert H % 8 == 0, "H must be divisible by 8"
    assert group_size == 128, "group_size must be 128"
    assert tokens_per_expert.ndim == 1 and tokens_per_expert.shape[0] == E

    tokens_per_expert = tokens_per_expert.to(device=y.device, dtype=torch.int32)

    fp8_dtype = torch.float8_e4m3fn
    y_q = torch.empty((E, T, H), dtype=fp8_dtype, device=y.device)

    ys_shape, ys_strides, ys_dtype = scales_shape_stride_dtype(E, T, G, quant_scale_fmt)
    y_s = torch.empty_strided(
        ys_shape,
        ys_strides,
        dtype=ys_dtype,
        device=y.device,
    )

    ceil_ue8m0 = quant_scale_fmt in [
        DeepGemmQuantScaleFMT.FLOAT32_CEIL_UE8M0,
        DeepGemmQuantScaleFMT.UE8M0,
    ]

    cuda_arch = current_platform.get_device_capability(
        device_id=y.device.index
    ).to_int()

    if cuda_arch >= 80:
        # Use optimized CUDA kernel on SM80+ devices
        torch.ops._C.persistent_masked_m_silu_mul_quant(
            y, tokens_per_expert, y_q, y_s, ceil_ue8m0
        )
    else:
        # Use Triton kernel on other platforms (including ROCm)
        stride_cnt_e = tokens_per_expert.stride()[0]

        # Static grid over experts and H-groups.
        # A loop inside the kernel handles the token dim
        grid = (E * G,)
        # strides (elements)
        stride_i_e, stride_i_t, stride_i_h = y.stride()
        stride_yq_e, stride_yq_t, stride_yq_h = y_q.stride()

        f_info = torch.finfo(fp8_dtype)
        fp8_max = f_info.max
        fp8_min = f_info.min
        eps: float = 1e-10
        assert y_s.dtype == torch.float32, (
            f"_silu_mul_fp8_quant_kernel does not support {y_s.dtype} scales. "
            "Only torch.float32 supported."
        )
        _silu_mul_fp8_quant_kernel[grid](
            y,
            y_q,
            y_s,
            tokens_per_expert,
            H,
            group_size,
            stride_i_e,
            stride_i_t,
            stride_i_h,
            stride_yq_e,
            stride_yq_t,
            stride_yq_h,
            ys_strides[0],
            ys_strides[1],
            ys_strides[2],
            stride_cnt_e,
            eps,
            fp8_min,
            fp8_max,
            ceil_ue8m0,
            BLOCK=group_size,
            NUM_STAGES=4,
            num_warps=1,
        )

    return y_q, y_s
