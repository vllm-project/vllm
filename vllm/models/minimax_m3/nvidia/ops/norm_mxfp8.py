# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Gemma RMSNorm (optionally fused with the residual add) in one Triton pass.

With an ``sf`` scale buffer the output is E4M3 data plus UE8M0 scales in the
128x4 swizzled layout, replacing the BF16 norm followed by a separate MXFP8
quantize (which re-reads the full normed activation). Without it, the output
is the plain BF16 norm. The residual is bitwise the BF16 norm's; the
quantized data and scales match the reference quantize within one E4M3 step
(the rstd reduction order differs).
"""

import torch

from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.utils.math_utils import round_up

MXFP8_BLOCK = 32
SF_ROW_TILE = 128


@triton.jit(do_not_specialize=["M"])
def _gemma_rmsnorm_mxfp8_kernel(
    x_ptr,
    res_ptr,
    w_ptr,
    out_ptr,
    sf_ptr,
    M,
    x_stride0,
    res_stride0,
    eps,
    H: tl.constexpr,
    launch_pdl: tl.constexpr,
):
    SF_ROW_TILE: tl.constexpr = 128
    G: tl.constexpr = H // 32
    GP: tl.constexpr = triton.next_power_of_2(H) // 32
    PADDED_COLS: tl.constexpr = (G + 3) // 4 * 4
    row = tl.program_id(0).to(tl.int64)
    g = tl.arange(0, GP)
    if launch_pdl:
        # All CTAs wait for the predecessor before any global access (the
        # padding-row stores included), then release dependents before the
        # loads: the consumer (the weight-loading GEMM) overlaps.
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()
    if sf_ptr is not None:
        # Swizzled 128x4 scale layout [m_tile, k_tile, 32, 4, 4].
        sf_off = (
            (g % 4)
            + (g // 4) * 512
            + (row % 32) * 16
            + ((row % SF_ROW_TILE) // 32) * 4
            + (row // SF_ROW_TILE) * (SF_ROW_TILE * PADDED_COLS)
        )
        if row >= M:
            # Zero the padding rows of the swizzled scale tensor.
            tl.store(sf_ptr + sf_off, tl.zeros((GP,), tl.uint8), mask=g < G)
            return

    cols = g[:, None] * 32 + tl.arange(0, 32)[None, :]
    mask = cols < H
    x = tl.load(x_ptr + row * x_stride0 + cols, mask=mask, other=0.0)
    w = tl.load(w_ptr + cols, mask=mask, other=0.0)
    if res_ptr is not None:
        r = tl.load(res_ptr + row * res_stride0 + cols, mask=mask, other=0.0)
    h = x.to(tl.float32)
    if res_ptr is not None:
        h += r.to(tl.float32)
        tl.store(res_ptr + row * res_stride0 + cols, h, mask=mask)
    var = tl.sum(h * h) / H
    rstd = tl.rsqrt(var + eps)
    y = (h * rstd * (1.0 + w.to(tl.float32))).to(tl.bfloat16)

    if sf_ptr is not None:
        amax = tl.max(tl.abs(y.to(tl.float32)), axis=1)
        scale = amax * (1.0 / 448.0)
        # UE8M0 scale: ceil to a power of two (exponent bump when the
        # mantissa is nonzero, except tiny subnormals), clamped to [0, 254].
        bits = scale.to(tl.uint32, bitcast=True)
        exp = (bits >> 23) & 255
        mant = bits & 0x7FFFFF
        bump = (mant != 0) & ~((exp == 0) & (mant <= 0x400000))
        e = tl.minimum(exp + bump.to(tl.uint32), 254)
        e = tl.where(scale <= 0.0, 0, e)
        # Inverse scale 2^(127 - e) (0 for e == 0), always a power of two.
        inv_bits = tl.where(e == 0, 0, (254 - e) << 23)
        inv = inv_bits.to(tl.float32, bitcast=True)
        # Bitwise the reference conversion (saturating; NaN -> +448);
        # checked exhaustively over all BF16 values and power-of-two scales.
        q = tl.clamp(y.to(tl.float32) * inv[:, None], -448.0, 448.0)
        tl.store(
            out_ptr + row * H + cols,
            q.to(tl.float8e4nv).to(tl.uint8, bitcast=True),
            mask=mask,
        )
        tl.store(sf_ptr + sf_off, e.to(tl.uint8), mask=g < G)
    else:
        tl.store(out_ptr + row * H + cols, y, mask=mask)


def gemma_rmsnorm_mxfp8(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    residual: torch.Tensor | None = None,
    mxfp8: bool = True,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """GemmaRMSNorm of ``x`` (plus ``residual`` in place when given).

    Args:
        x: (num_tokens, hidden) BF16 activation, contiguous along the row
            (any row stride). Read only.
        weight: (hidden,) norm weight; the norm scales by (1 + weight).
        eps: RMS norm epsilon.
        residual: same shape/dtype/stride(1) as x; updated in place to
            x + residual. None for a plain norm.
        mxfp8: return MXFP8 (E4M3 data + UE8M0 scales) or the plain BF16
            normed activation.

    Returns:
        ``(data, scale)``: with ``mxfp8=True``, E4M3 data (num_tokens,
        hidden) and UE8M0 scales in the 128x4 swizzled layout with zeroed
        padding rows; with ``mxfp8=False``, the BF16 normed activation and
        None.

    """
    assert x.dim() == 2 and x.stride(1) == 1, x.shape
    assert weight.is_contiguous()
    assert residual is None or (residual.shape == x.shape and residual.stride(1) == 1)
    M, H = x.shape
    sf = None
    if mxfp8:
        padded_m = round_up(M, SF_ROW_TILE)
        out = torch.empty((M, H), dtype=torch.uint8, device=x.device)
        sf = torch.empty(
            padded_m * round_up(H // MXFP8_BLOCK, 4), dtype=torch.uint8, device=x.device
        )
        if M == 0:
            return out.view(torch.float8_e4m3fn), sf
    else:
        padded_m = M
        out = torch.empty((M, H), dtype=torch.bfloat16, device=x.device)
        if M == 0:
            return out, None
    _gemma_rmsnorm_mxfp8_kernel[(padded_m,)](
        x,
        residual,
        weight,
        out,
        sf,
        M,
        x.stride(0),
        residual.stride(0) if residual is not None else x.stride(0),
        eps,
        H=H,
        launch_pdl=current_platform.is_arch_support_pdl(),
        num_warps=4,
    )
    if mxfp8:
        return out.view(torch.float8_e4m3fn), sf
    return out, None
