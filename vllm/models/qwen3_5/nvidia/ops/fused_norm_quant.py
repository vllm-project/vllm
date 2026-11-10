# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Gemma (1 + w) add-RMSNorm fused with the consumer linear's input quant.

Replaces ``gemma_fused_add_rmsnorm`` followed by a standalone static FP8
(``scaled_fp8_quant``) or NVFP4 (``scaled_fp4_quant``) kernel. The quantized
values are computed from the bf16-rounded normed activation with the same
scale arithmetic as those kernels, so codes match the unfused chain.
"""

import torch

from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.utils.math_utils import round_up

FP8_DTYPE = torch.float8_e4m3fn
FP8_MAX = tl.constexpr(448.0)


@triton.jit
def _rcp_approx_ftz(x):
    # Matches reciprocal_approximate_ftz in csrc/quantization/fp4/nvfp4_utils.cuh.
    return tl.inline_asm_elementwise(
        "rcp.approx.ftz.f32 $0, $1;",
        "=r,r",
        [x],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _cvt_e2m1x2(lo, hi):
    """Pack fp32 (lo, hi) pairs into E2M1 bytes (lo in the low nibble)."""
    return tl.inline_asm_elementwise(
        """
        {
        .reg .b8 b0, b1, b2, b3;
        cvt.rn.satfinite.e2m1x2.f32 b0, $5, $1;
        cvt.rn.satfinite.e2m1x2.f32 b1, $6, $2;
        cvt.rn.satfinite.e2m1x2.f32 b2, $7, $3;
        cvt.rn.satfinite.e2m1x2.f32 b3, $8, $4;
        mov.b32 $0, {b0, b1, b2, b3};
        }
        """,
        "=r,r,r,r,r,r,r,r,r",
        [lo, hi],
        dtype=tl.uint8,
        is_pure=True,
        pack=4,
    )


@triton.jit
def _gemma_add_rms_norm_quant_kernel(
    x_ptr,
    res_ptr,
    w_ptr,
    scale_ptr,
    q_ptr,
    sf_ptr,
    y_ptr,
    eps,
    H: tl.constexpr,
    HAS_RESIDUAL: tl.constexpr,
    QUANT_NVFP4: tl.constexpr,
    WRITE_Y: tl.constexpr,
    launch_pdl: tl.constexpr,
    TRIGGER_EARLY: tl.constexpr,
):
    # Index the row as [G, 8, 2]: G groups of 16 (one NVFP4 block each) of 8
    # (lo, hi) pairs, so block reductions and fp4 packing need no reshapes.
    G: tl.constexpr = triton.next_power_of_2(H) // 16
    row = tl.program_id(0).to(tl.int64)
    g = tl.arange(0, G)
    pair = tl.arange(0, 8)
    cols = (
        g[:, None, None] * 16 + pair[None, :, None] * 2 + tl.arange(0, 2)[None, None, :]
    )
    mask = cols < H

    if launch_pdl:
        tl.extra.cuda.gdc_wait()
        if TRIGGER_EARLY:
            # Launch dependents before the loads: the consumer (e.g. the
            # weight-loading GEMM) overlaps with this kernel.
            tl.extra.cuda.gdc_launch_dependents()
    x = tl.load(x_ptr + row * H + cols, mask=mask, other=0.0)
    if HAS_RESIDUAL:
        r = tl.load(res_ptr + row * H + cols, mask=mask, other=0.0)
    w = tl.load(w_ptr + cols, mask=mask, other=0.0)
    scale = tl.load(scale_ptr)

    h = x.to(tl.float32)
    if HAS_RESIDUAL:
        h += r.to(tl.float32)
        # The norm uses the fp32 sum, like flashinfer's fused_add_rmsnorm.
        tl.store(res_ptr + row * H + cols, h, mask=mask)
    var = tl.sum(h * h) / H
    rstd = tl.rsqrt(var + eps)
    # Round to the activation dtype, like the unfused norm output.
    y = (h * rstd * (1.0 + w.to(tl.float32))).to(x_ptr.dtype.element_ty)
    if WRITE_Y:
        tl.store(y_ptr + row * H + cols, y, mask=mask)
    y = y.to(tl.float32)

    if QUANT_NVFP4:
        # scaled_fp4_quant numerics; `scale` is the global scale (448 * 6 / amax).
        vmax = tl.max(tl.max(tl.abs(y), axis=2), axis=1)
        sf = scale * (vmax * _rcp_approx_ftz(tl.full((G,), 6.0, tl.float32)))
        sf_fp8 = tl.minimum(sf, FP8_MAX).to(tl.float8e4nv)
        sf32 = sf_fp8.to(tl.float32)
        out_scale = tl.where(
            sf32 != 0.0, _rcp_approx_ftz(sf32 * _rcp_approx_ftz(scale)), 0.0
        )
        lo, hi = tl.split(y * out_scale[:, None, None])
        pcols = g[:, None] * 8 + pair[None, :]
        tl.store(
            q_ptr + row * (H // 2) + pcols, _cvt_e2m1x2(lo, hi), mask=pcols < H // 2
        )

        # Swizzled 128x4 scale layout [m_tile, k_tile, 32, 4, 4].
        num_k_tiles: tl.constexpr = (H // 16 + 3) // 4
        sf_off = (
            ((row >> 7) * num_k_tiles + (g >> 2)) * 512
            + (row & 31) * 16
            + ((row >> 5) & 3) * 4
            + (g & 3)
        )
        tl.store(sf_ptr + sf_off, sf_fp8, mask=g < H // 16)
    else:
        inv_scale = tl.math.div_rn(1.0, scale)
        q = tl.clamp(y * inv_scale, -FP8_MAX, FP8_MAX)
        tl.store(q_ptr + row * H + cols, q, mask=mask)

    if launch_pdl and not TRIGGER_EARLY:
        tl.extra.cuda.gdc_launch_dependents()


def gemma_add_rms_norm_quant(
    x: torch.Tensor,
    residual: torch.Tensor | None,
    weight: torch.Tensor,
    eps: float,
    scale: torch.Tensor,
    nvfp4: bool,
    write_y: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Gemma RMSNorm of ``x (+ residual)`` quantized for the next linear.

    Args:
        x: (num_tokens, hidden) bf16/fp16 activation, contiguous. Read only.
        residual: same shape as x, or None. Updated in place to x + residual.
        weight: (hidden,) Gemma norm weight; the norm scales by (1 + weight).
        eps: RMS norm epsilon.
        scale: FP8: the static per-tensor input scale. NVFP4: the global
            scale (``input_global_scale_inv``).
        nvfp4: quantize to NVFP4 with swizzled 128x4 block scales instead of
            static FP8.
        write_y: also return the bf16 normed activation.

    Returns:
        ``(q, block_scale, y)``: quantized data (fp8, or packed fp4 uint8),
        NVFP4 block scales (fp8, ``(round_up(M, 128), round_up(H // 16, 4))``;
        rows >= M are left unwritten) or ``scale`` for FP8, and the normed
        activation (or None).

    """
    assert x.dim() == 2 and x.is_contiguous()
    assert x.dtype in (torch.bfloat16, torch.float16)
    assert residual is None or (residual.shape == x.shape and residual.is_contiguous())
    assert weight.shape == (x.shape[1],) and weight.is_contiguous()
    assert scale.numel() == 1 and scale.dtype == torch.float32
    M, H = x.shape
    if nvfp4:
        # Whole 4-column scale tiles: padded scale columns are not written.
        assert H % 64 == 0
        q = torch.empty((M, H // 2), dtype=torch.uint8, device=x.device)
        block_scale = torch.empty(
            (round_up(M, 128), round_up(H // 16, 4) // 4),
            dtype=torch.int32,
            device=x.device,
        ).view(FP8_DTYPE)
    else:
        q = torch.empty((M, H), dtype=FP8_DTYPE, device=x.device)
        block_scale = scale
    y = torch.empty_like(x) if write_y else None
    if M == 0:
        return q, block_scale, y

    # Swept on GB300 at H=5120, M=1..8192: within 3% of the best of 2/4/8/16
    # warps at every swept M except NVFP4 M=129 (4%).
    max_m_16_warps = 128 if nvfp4 else 384
    _gemma_add_rms_norm_quant_kernel[(M,)](
        x,
        residual if residual is not None else x,
        weight,
        scale,
        q,
        block_scale,
        y if y is not None else x,
        eps,
        H=H,
        HAS_RESIDUAL=residual is not None,
        QUANT_NVFP4=nvfp4,
        WRITE_Y=write_y,
        num_warps=16 if max_m_16_warps >= M else 4,
        launch_pdl=current_platform.is_arch_support_pdl(),
        # Overlaps the consumer's weight load with this kernel. Swept on
        # GB300: ~2% win at M <= 128; the prelaunched consumer grid steals
        # SMs at M >= 192 (fp8 +4-7%).
        TRIGGER_EARLY=M <= 128,
    )
    return q, block_scale, y
