# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""ROCm MoRI EP helpers.

Ported from SGLang ``rocm_moe_utils.py``: dequantize (upscale) mxfp4 activations
that were FP4-dispatched by MoRI back to a float dtype, for AITER fused-MoE
weight/activation dtype pairs that have no native fp4x2 kernel.
"""

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def upscale_fp4x2_block32_kernel(
    A_u8_ptr,  # *uint8  (view from float4_e2m1fn_x2)
    S_u8_ptr,  # *uint8  (view from float8_e8m0fnu), shape (M, N_fp4/32)
    Out_ptr,  # *fp16/fp32/bf16, shape (M, N_fp4)
    N_FP4: tl.constexpr,
    recv_token_num,
    stride_am,
    stride_an,  # A strides (in uint8 elements) for (M, packed_N)
    stride_sm,
    stride_sn,  # S strides (in uint8 elements) for (M, N_FP4/32)
    stride_om,
    stride_on,  # Out strides (in output elements) for (M, N_FP4)
    BLOCK_N: tl.constexpr,
    OUT_DTYPE: tl.constexpr,  # tl.float16 / tl.float32 / tl.bfloat16
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    recv_token_num_val = tl.load(recv_token_num)
    if pid_m >= recv_token_num_val:
        return

    offs = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = offs < N_FP4

    # --------------------------
    # Load packed fp4x2 byte
    # --------------------------
    byte_idx = offs >> 1  # offs // 2
    is_hi = (offs & 1) != 0  # select high nibble?

    a_ptrs = A_u8_ptr + pid_m * stride_am + byte_idx * stride_an
    a_byte = tl.load(a_ptrs, mask=mask, other=0).to(tl.int32)

    lo = a_byte & 0xF
    hi = (a_byte >> 4) & 0xF
    code = tl.where(is_hi, hi, lo).to(tl.int32)  # 0..15

    # --------------------------
    # Decode float4_e2m1fn
    # layout: [sign|exp(2)|mant(1)]
    # bias=1, finite-only
    # --------------------------
    sign = (code >> 3) & 0x1
    exp = (code >> 1) & 0x3
    mant = code & 0x1

    mant_f = mant.to(tl.float32) * 0.5
    is_sub = exp == 0

    # normal: 2^(exp-bias) * (1 + mant/2), bias=1
    e_norm = (exp - 1).to(tl.float32)
    val_norm = tl.exp2(e_norm) * (1.0 + mant_f)

    # subnorm/zero: mant/2 * 2^(1-bias) = mant/2
    val_sub = mant_f

    val = tl.where(is_sub, val_sub, val_norm)
    val = tl.where(sign != 0, -val, val)  # apply sign

    # --------------------------
    # Per-token block32 scale: scale_idx = offs // 32
    # scale dtype: float8_e8m0fnu stored in uint8
    # decode: e==0 -> 0
    #         e in [1..254] -> 2^(e-127)
    #         e==255 -> clamp to 254
    # --------------------------
    scale_idx = offs >> 5  # offs // 32

    s_ptrs = S_u8_ptr + pid_m * stride_sm + scale_idx * stride_sn
    e = tl.load(s_ptrs, mask=mask, other=0).to(tl.int32)

    e = tl.minimum(e, 254)  # clamp 255->254
    is_zero = e == 0
    exp_s = (e - 127).to(tl.float32)
    s = tl.exp2(exp_s)
    s = tl.where(is_zero, 0.0, s)

    out = (val * s).to(OUT_DTYPE)

    out_ptrs = Out_ptr + pid_m * stride_om + offs * stride_on
    tl.store(out_ptrs, out, mask=mask)


def upscale_mxfp4(
    hidden_state: torch.Tensor,
    hidden_state_scale: torch.Tensor,
    recv_token_num: torch.Tensor,
    output_dtype: torch.dtype,
) -> torch.Tensor:
    """Dequantize mxfp4 (fp4x2 + e8m0 block32 scale) activations to a float dtype.

    - hidden_state: (M, packed_N) torch.float4_e2m1fn_x2
    - hidden_state_scale: (M, N_fp4/32) torch.float8_e8m0fnu
    - recv_token_num: 0-d/1-elem int tensor; rows >= it are left untouched
    - output: (M, N_fp4) output_dtype
    """
    assert hidden_state.dtype == torch.float4_e2m1fn_x2, hidden_state.dtype
    assert hidden_state_scale.dtype == torch.float8_e8m0fnu, hidden_state_scale.dtype

    M, packed_N = hidden_state.shape
    N_fp4 = packed_N * 2

    assert hidden_state_scale.shape[0] == M
    assert hidden_state_scale.shape[1] == (N_fp4 // 32), (
        hidden_state_scale.shape,
        N_fp4,
    )

    # Triton doesn't (reliably) accept torch.float4/float8 pointers directly.
    # Use raw uint8 views.
    A_u8 = hidden_state.view(torch.uint8)
    S_u8 = hidden_state_scale.view(torch.uint8)

    Out = torch.empty((M, N_fp4), dtype=output_dtype, device=hidden_state.device)

    BLOCK_N = 256
    grid = (M, triton.cdiv(N_fp4, BLOCK_N))

    OUT_TL = (
        tl.float16
        if output_dtype == torch.float16
        else tl.bfloat16
        if output_dtype == torch.bfloat16
        else tl.float32
    )

    upscale_fp4x2_block32_kernel[grid](
        A_u8,
        S_u8,
        Out,
        N_FP4=N_fp4,
        recv_token_num=recv_token_num,
        stride_am=A_u8.stride(0),
        stride_an=A_u8.stride(1),
        stride_sm=S_u8.stride(0),
        stride_sn=S_u8.stride(1),
        stride_om=Out.stride(0),
        stride_on=Out.stride(1),
        BLOCK_N=BLOCK_N,
        OUT_DTYPE=OUT_TL,
        num_warps=4,
    )
    return Out
