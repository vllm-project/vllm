# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fused SiLU+Mul + per-token-group FP8 quantization for MoE (row-major scales)."""

import torch

from vllm.platforms import current_platform
from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    is_deep_gemm_e8m0_used,
    get_fp8_min_max,
)
from vllm.triton_utils import tl, triton


@triton.jit
def _silu_mul_per_token_group_quant_fp8_rowmajor(
    y_ptr,  # [M, N]  (gate + up concatenated)
    y_q_ptr,  # [M, N // 2]
    y_s_ptr,  # [M, (N // 2) // GROUP_SIZE]  row-major
    M,  # num tokens
    N,  # intermediate size (gate+up)
    y_row_stride,
    y_q_row_stride,
    y_s_row_stride,
    y_s_col_stride,
    eps,
    fp8_min: tl.constexpr,
    fp8_max: tl.constexpr,
    use_ue8m0: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    N_2 = N // 2

    m_offset = pid_m * BLOCK_M
    n_offset = pid_n * BLOCK_N

    offs_n = tl.arange(0, BLOCK_N).to(tl.int64)
    offs_m = tl.arange(0, BLOCK_M).to(tl.int64)
    m_mask = (m_offset + offs_m) < M

    # Load gate and up halves
    base_y_ptr = y_ptr + m_offset * y_row_stride + n_offset
    gate_ptrs = base_y_ptr + offs_m[:, None] * y_row_stride + offs_n[None, :]
    gate = tl.load(gate_ptrs, mask=m_mask[:, None] & (offs_n[None, :] < N_2), other=0.0)
    up = tl.load(gate_ptrs + N_2, mask=m_mask[:, None] & (offs_n[None, :] < N_2), other=0.0)

    # silu & mul
    gate_f32 = gate.to(tl.float32)
    one_f32 = tl.cast(1, tl.float32)
    silu_out = (gate_f32 / (one_f32 + tl.exp(-gate_f32))).to(y_ptr.dtype.element_ty)
    y = (silu_out * up).to(tl.float32)

    # per-token-group quantize
    _absmax = tl.maximum(tl.max(tl.abs(y), axis=1), eps)
    scale_raw = _absmax * (1.0 / fp8_max)
    y_s = tl.math.exp2(tl.ceil(tl.log2(scale_raw))) if use_ue8m0 else scale_raw
    y_s_2d = tl.reshape(y_s, (BLOCK_M, 1))
    y_q = tl.clamp(y / y_s_2d, fp8_min, fp8_max).to(y_q_ptr.dtype.element_ty)

    # store quantized output
    base_y_q_ptr = y_q_ptr + m_offset * y_q_row_stride + n_offset
    y_q_ptrs = base_y_q_ptr + offs_m[:, None] * y_q_row_stride + offs_n[None, :]
    tl.store(y_q_ptrs, y_q, mask=m_mask[:, None] & (offs_n[None, :] < N_2))

    # store scale (row-major: [M, N_2 // GROUP_SIZE])
    group_id = n_offset // GROUP_SIZE
    base_y_s_ptr = y_s_ptr + m_offset * y_s_row_stride + group_id * y_s_col_stride
    y_s_ptrs = base_y_s_ptr + offs_m * y_s_row_stride
    y_s_1d = tl.reshape(y_s, (BLOCK_M,))
    tl.store(y_s_ptrs, y_s_1d, mask=m_mask)


def silu_mul_per_token_group_quant_fp8_rowmajor(
    input: torch.Tensor,  # [M, N]
    output: torch.Tensor | None = None,  # [M, N // 2]
    group_size: int = 128,
    use_ue8m0: bool | None = None,
    eps: float = 1e-10,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused silu+mul + per-token-group FP8 quant with row-major scales.

    Compatible with Triton MoE kernel's expected scale format.
    """
    if use_ue8m0 is None:
        use_ue8m0 = is_deep_gemm_e8m0_used()

    assert input.ndim == 2
    if output is not None:
        assert output.ndim == 2

    M, N = input.size()
    N_2 = N // 2

    assert N_2 % group_size == 0

    fp8_dtype = current_platform.fp8_dtype()
    fp8_min, fp8_max = get_fp8_min_max()

    if output is None:
        output = torch.empty((M, N_2), dtype=fp8_dtype, device=input.device)

    num_groups = N_2 // group_size
    output_scales = torch.empty(
        (M, num_groups), dtype=torch.float32, device=input.device
    )

    BLOCK_M = 8
    BLOCK_N = group_size

    grid = (triton.cdiv(M, BLOCK_M), num_groups)

    _silu_mul_per_token_group_quant_fp8_rowmajor[grid](
        input,
        output,
        output_scales,
        M,
        N,
        input.stride(0),
        output.stride(0),
        output_scales.stride(0),
        output_scales.stride(1),
        eps,
        fp8_min=fp8_min,
        fp8_max=fp8_max,
        use_ue8m0=use_ue8m0,
        GROUP_SIZE=group_size,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
    )

    return output, output_scales
