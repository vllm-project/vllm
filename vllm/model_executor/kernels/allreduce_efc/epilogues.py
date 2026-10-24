# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Epilogues for ``LamportAllReduce.bind`` and the building blocks they share.

Each helper runs unchanged in every EFC phase. Rounding is explicit: an
epilogue that should match an unfused path rounds to BF16 wherever that path
returns BF16.
"""

import torch

from .efc import (
    DeviceScalar,
    Group128Scale,
    NVFP4Scale,
    PerToken,
    Row,
    Scalar,
    Streams,
    Weight,
)

FP8_MAX = 448.0


def rms_norm(cfg, x, weight, eps, *, weight_offset: float = 0.0):
    """FlashInfer's LL consumer RMSNorm: ``x * rsqrt(mean(x^2) + eps) * w``."""
    inv_rms = cfg.rsqrt(cfg.row_sum(x * x) / cfg.hidden + eps)
    w = weight.load()
    if weight_offset:
        w = w + weight_offset
    return x * inv_rms * w


def nvfp4_quant(cfg, y, global_scale, q, sf) -> None:
    """``scaled_fp4_quant`` (``cvt_warp_fp16_to_fp4``) of a BF16-valued ``y``."""
    gs = global_scale.load()
    amax = cfg.group_max(cfg.abs(y), 16)
    scale = cfg.round(gs * (amax * cfg.rcp_approx(6.0)), torch.float8_e4m3fn)
    inv_scale = cfg.where(scale != 0.0, cfg.rcp_approx(scale * cfg.rcp_approx(gs)), 0.0)
    sf.store(scale)
    q.store(y * inv_scale)


def fp8_static_quant(cfg, y, scale, q) -> None:
    """``static_scaled_fp8_quant``: ``clamp(y / scale)``."""
    q.store(cfg.clamp(y / scale.load(), -FP8_MAX, FP8_MAX))


def fp8_group_quant(
    cfg, y, q, s, *, group_size: int = 128, eps: float = 1e-10, ue8m0: bool = False
) -> None:
    """``per_token_group_quant_fp8`` with row-major scales."""
    amax = cfg.maximum(cfg.group_max(cfg.abs(y), group_size), eps)
    scale = amax / FP8_MAX
    if ue8m0:
        scale = cfg.ue8m0_ceil(scale)
    s.store(scale)
    q.store(cfg.clamp(y / scale, -FP8_MAX, FP8_MAX))


def _add_residual(cfg, residual):
    return cfg.round(cfg.accum() + residual.load(), torch.bfloat16)


def residual_rmsnorm(
    cfg,
    residual: Row,
    weight: Weight,
    eps: Scalar,
    residual_out: Row,
    out: Row,
):
    """FlashInfer's ``AllReduceRMSNormLLKernel`` (``weight_bias=0``)."""
    h = _add_residual(cfg, residual)
    residual_out.store(h)
    out.store(rms_norm(cfg, h, weight, eps))


def residual_gemma_rmsnorm(
    cfg,
    residual: Row,
    weight: Weight,
    eps: Scalar,
    residual_out: Row,
    out: Row,
):
    """``GemmaRMSNorm``: the weight is stored as ``w - 1``."""
    h = _add_residual(cfg, residual)
    residual_out.store(h)
    out.store(rms_norm(cfg, h, weight, eps, weight_offset=1.0))


def residual_rmsnorm_fp8(
    cfg,
    residual: Row,
    weight: Weight,
    eps: Scalar,
    scale: DeviceScalar,
    residual_out: Row,
    q: Row,
):
    h = _add_residual(cfg, residual)
    residual_out.store(h)
    y = cfg.round(rms_norm(cfg, h, weight, eps), torch.bfloat16)
    fp8_static_quant(cfg, y, scale, q)


def residual_rmsnorm_nvfp4(
    cfg,
    residual: Row,
    weight: Weight,
    eps: Scalar,
    global_scale: DeviceScalar,
    residual_out: Row,
    q: Row,
    sf: NVFP4Scale,
):
    h = _add_residual(cfg, residual)
    residual_out.store(h)
    y = cfg.round(rms_norm(cfg, h, weight, eps), torch.bfloat16)
    nvfp4_quant(cfg, y, global_scale, q, sf)


def residual_rmsnorm_fp8_group(
    cfg,
    residual: Row,
    weight: Weight,
    eps: Scalar,
    residual_out: Row,
    q: Row,
    s: Group128Scale,
):
    h = _add_residual(cfg, residual)
    residual_out.store(h)
    y = cfg.round(rms_norm(cfg, h, weight, eps), torch.bfloat16)
    fp8_group_quant(cfg, y, q, s)


def mhc_post_pre(cfg, residual, post, comb, pre, residual_out):
    """DSV4.1's mHC post-mix of the hc streams (stored) and their pre-mix
    collapse (returned, BF16-valued), in ``_LamportMHCDeviceKernel``'s order."""
    hc = len(residual)
    reduced = cfg.round(cfg.accum(), torch.bfloat16)
    collapse = cfg.zeros()
    for target in range(hc):
        mixed = reduced * post[target]
        for source in range(hc):
            mixed = mixed + residual[source].load() * comb[source, target]
        mixed = cfg.round(mixed, torch.bfloat16)
        residual_out[target].store(mixed)
        collapse = collapse + mixed * pre[target]
    return cfg.round(collapse, torch.bfloat16)


def mhc_rmsnorm(
    cfg,
    residual: Streams,
    post: PerToken,
    comb: PerToken,
    pre: PerToken,
    weight: Weight,
    eps: Scalar,
    residual_out: Streams,
    out: Row,
):
    """``AllReduceMHC``'s epilogue: post-mixed streams and normalized input."""
    prenorm = mhc_post_pre(cfg, residual, post, comb, pre, residual_out)
    out.store(rms_norm(cfg, prenorm, weight, eps))


def mhc_rmsnorm_nvfp4(
    cfg,
    residual: Streams,
    post: PerToken,
    comb: PerToken,
    pre: PerToken,
    weight: Weight,
    eps: Scalar,
    global_scale: DeviceScalar,
    residual_out: Streams,
    q: Row,
    sf: NVFP4Scale,
):
    prenorm = mhc_post_pre(cfg, residual, post, comb, pre, residual_out)
    y = cfg.round(rms_norm(cfg, prenorm, weight, eps), torch.bfloat16)
    nvfp4_quant(cfg, y, global_scale, q, sf)
