# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Triton fused SiLU-and-Mul + NVFP4 (e2m1) quantization.

Uses PTX inline assembly to match the CUDA kernel's fast-math intrinsics
and e2m1 conversion bitwise-exactly:
  - rcp.approx.ftz.f32  (reciprocal_approximate_ftz)
  - ex2.approx.ftz.f32  (__expf via base-2 fast exp)
  - cvt.rn.satfinite.e2m1x2.f32  (float32 → packed e2m1)
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from vllm._custom_ops import create_fp4_output_tensors

# ── PTX helpers ──────────────────────────────────────────────────────────


@triton.jit
def _rcp_approx_ftz(x):
    """rcp.approx.ftz.f32 — fast reciprocal (~1 mantissa-bit precision)."""
    return tl.inline_asm_elementwise(
        asm="rcp.approx.ftz.f32 $0, $1;",
        constraints="=f,f",
        args=[x],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _ex2_approx_ftz(x):
    """ex2.approx.ftz.f32 — fast 2^x (used to implement __expf)."""
    return tl.inline_asm_elementwise(
        asm="ex2.approx.ftz.f32 $0, $1;",
        constraints="=f,f",
        args=[x],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _cvt_e2m1x2(even, odd):
    """cvt.rn.satfinite.e2m1x2.f32 — pack two f32 into one byte of e2m1."""
    return tl.inline_asm_elementwise(
        asm=(
            "{ .reg .b8 tmp;"
            "  cvt.rn.satfinite.e2m1x2.f32 tmp, $2, $1;"
            "  cvt.u32.u8 $0, tmp; }"
        ),
        constraints="=r,f,f",
        args=[even, odd],
        dtype=tl.int32,
        is_pure=True,
        pack=1,
    )


# ── kernel ───────────────────────────────────────────────────────────────


@triton.jit
def _silu_mul_nvfp4_quant_kernel(
    input_ptr,  # [M, 2*N]  bfloat16
    output_ptr,  # [M, N//2] uint8  (packed e2m1 pairs)
    sf_out_ptr,  # swizzled scale factors (viewed as uint8*)
    sf_scale_ptr,  # global scale (float32 scalar)
    M,
    N,  # output dim (= input last dim / 2)
    stride_in_m,
    stride_out_m,
    num_k_tiles,  # ceil(N / 64)
    PAIRS: tl.constexpr,  # = 8  (16 values → 8 packed bytes)
):
    """One program = one (row, quantisation-group-of-16) pair."""
    LOG2E: tl.constexpr = 1.4426950408889634

    pid_m = tl.program_id(0)
    pid_g = tl.program_id(1)

    col_base = pid_g * (PAIRS * 2)  # = pid_g * 16
    pair_off = tl.arange(0, PAIRS)  # [0..7]
    even_cols = col_base + 2 * pair_off  # [0,2,4,...,14] + col_base
    odd_cols = even_cols + 1  # [1,3,5,...,15] + col_base

    row_off = pid_m * stride_in_m

    # ── load gate / up (BF16 → float32) ──
    gate_e = tl.load(input_ptr + row_off + even_cols).to(tl.float32)
    up_e = tl.load(input_ptr + row_off + N + even_cols).to(tl.float32)
    gate_o = tl.load(input_ptr + row_off + odd_cols).to(tl.float32)
    up_o = tl.load(input_ptr + row_off + N + odd_cols).to(tl.float32)

    # ── fast-math SiLU: __fdividef(x, 1 + __expf(-x)) ──
    # __expf(y) = ex2.approx.ftz(y * log2(e))
    # __fdividef(a, b) = a * rcp.approx.ftz(b)
    exp_e = _ex2_approx_ftz((-gate_e) * LOG2E)
    exp_o = _ex2_approx_ftz((-gate_o) * LOG2E)
    silu_e = gate_e * _rcp_approx_ftz(1.0 + exp_e)
    silu_o = gate_o * _rcp_approx_ftz(1.0 + exp_o)

    res_e = silu_e * up_e
    res_o = silu_o * up_o

    # ── BF16 round-trip (matches compute_silu_mul returning PackedVec<BF16>) ──
    res_e = res_e.to(tl.bfloat16).to(tl.float32)
    res_o = res_o.to(tl.bfloat16).to(tl.float32)

    # ── per-group abs-max (over all 16 values) ──
    amax = tl.maximum(tl.max(tl.abs(res_e)), tl.max(tl.abs(res_o)))

    # ── scale factor: sf_scale * (amax * rcp.approx.ftz(6.0)) ──
    sf_scale_val = tl.load(sf_scale_ptr).to(tl.float32)
    sf_raw = sf_scale_val * (amax * _rcp_approx_ftz(6.0))

    # quantise sf → float8_e4m3fn → float32 (exact round-trip)
    sf_fp8 = sf_raw.to(tl.float8e4nv)
    sf_rounded = sf_fp8.to(tl.float32)

    # ── write scale byte into swizzled layout ──
    sf_col = pid_g
    m_tile = pid_m // 128
    outer_m = pid_m % 32
    inner_m = (pid_m // 32) % 4
    k_tile = sf_col // 4
    inner_k = sf_col % 4
    sf_offset = (
        (m_tile * num_k_tiles + k_tile) * 512 + outer_m * 16 + inner_m * 4 + inner_k
    )
    sf_byte = sf_fp8.to(tl.uint8, bitcast=True)
    tl.store(sf_out_ptr + sf_offset, sf_byte)

    # ── output scale: rcp(sf_rounded * rcp(sf_scale)) ──
    rcp_sf_scale = _rcp_approx_ftz(sf_scale_val)
    out_scale = tl.where(
        sf_rounded != 0.0,
        _rcp_approx_ftz(sf_rounded * rcp_sf_scale),
        0.0,
    )

    # ── scale values ──
    scaled_e = res_e * out_scale
    scaled_o = res_o * out_scale

    # ── PTX e2m1 conversion + pack ──
    packed = _cvt_e2m1x2(scaled_e, scaled_o).to(tl.uint8)

    # ── store 8 packed bytes ──
    out_off = pid_m * stride_out_m + col_base // 2 + pair_off
    tl.store(output_ptr + out_off, packed)


# ── Python entry point ───────────────────────────────────────────────────


def silu_and_mul_nvfp4_quant(
    input: torch.Tensor,
    input_global_scale: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Drop-in replacement for vllm._custom_ops.silu_and_mul_nvfp4_quant."""
    if input.ndim == 1:
        input = input.unsqueeze(0)
    else:
        input = input.reshape(-1, input.shape[-1])

    M, two_N = input.shape
    N = two_N // 2
    assert N % 16 == 0, f"N must be a multiple of 16, got {N}"

    output, output_scale = create_fp4_output_tensors(
        M,
        N,
        input.device,
        is_sf_swizzled_layout=True,
    )

    num_groups = N // 16
    num_k_tiles = (num_groups + 3) // 4  # ceil(N / 64)

    grid = (M, num_groups)
    _silu_mul_nvfp4_quant_kernel[grid](
        input,
        output,
        output_scale.view(torch.uint8),
        input_global_scale,
        M,
        N,
        input.stride(0),
        output.stride(0),
        num_k_tiles,
        PAIRS=8,
    )

    output_scale = output_scale.view(torch.float8_e4m3fn)
    return output, output_scale
