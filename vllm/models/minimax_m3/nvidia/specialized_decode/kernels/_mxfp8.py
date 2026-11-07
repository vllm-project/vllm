# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The MXFP8 operands of a loaded vLLM linear, in the layout the kernels read.

Every kernel that reads an MXFP8 weight (front, o_proj, the MoE's shared
expert) takes the e4m3 weight row-major `[N, K]` (N = output features) and
its UE8M0 scales (one per 32 K elements) as the flat F8_128x4-swizzled buffer
of `[N, K / 32]` (rows padded to 128, scale columns to a multiple of 4).

What the stock MXFP8 linear kernels leave after `process_weights_after_loading`:

====================================  ===================  ==================
kernel                                `weight`           `weight_scale`
====================================  ===================  ==================
FlashInferCutlassMxfp8LinearKernel    [N, K] contiguous    F8_128x4, flat
FlashInferCutedslMxfp8LinearKernel    [K, N] view (the     F8_128x4, flat
                                      transpose of a
                                      contiguous [N, K])
FlashInferTrtllmMxfp8LinearKernel     [N', K] shuffled     shuffled, flat
====================================  ===================  ==================

The first two hold the same bytes: the CuTe-DSL view is undone with `.t()`
(no copy). The TRT-LLM shuffle is refused. Nothing is requantized: the e4m3
and UE8M0 bytes are the stock layer's own.
"""

from __future__ import annotations

import torch


class Mxfp8LayoutError(ValueError):
    pass


def mxfp8_linear_nk(
    lin: torch.nn.Module, what: str, shape: tuple[int, int] | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """(e4m3 weight [N, K] contiguous, flat uint8 F8_128x4 UE8M0 scales) of a
    vLLM MXFP8 linear after weight processing. `shape`: the expected
    `(N, K)`; without it, a contiguous weight is taken as `[N, K]`."""
    w = getattr(lin, "weight", None)
    s = getattr(lin, "weight_scale", None)
    if w is None or s is None:
        raise Mxfp8LayoutError(
            f"{what}: no weight / weight_scale (not an MXFP8 linear?)"
        )
    w, s = w.detach(), s.detach()
    if w.dtype != torch.float8_e4m3fn or w.dim() != 2:
        raise Mxfp8LayoutError(
            f"{what}: weight {w.dtype} {tuple(w.shape)}, expected 2-D float8_e4m3fn "
            "(MXFP8)"
        )
    if hasattr(lin, "_mxfp8_trtllm_output_size"):
        raise Mxfp8LayoutError(
            f"{what}: TRT-LLM-shuffled MXFP8 weight; the kernels read the FlashInfer "
            "CUTLASS / CuTe-DSL layout"
        )
    if shape is not None:
        if tuple(w.shape) == tuple(shape):
            pass
        elif tuple(w.shape) == tuple(shape)[::-1]:
            w = w.t()
        else:
            raise Mxfp8LayoutError(
                f"{what}: weight {tuple(w.shape)}, expected [N, K] = {tuple(shape)} "
                "or its [K, N] view"
            )
    elif not w.is_contiguous() and w.t().is_contiguous():
        w = w.t()  # the CuTe-DSL kernel's [K, N] view
    if not w.is_contiguous():
        raise Mxfp8LayoutError(
            f"{what}: weight [N, K] = {tuple(w.shape)} with strides {w.stride()} is "
            "not row-major"
        )
    n, k = w.shape
    if k % 32:
        raise Mxfp8LayoutError(f"{what}: K = {k} is not a multiple of 32")
    want = (n + 127) // 128 * 128 * ((k // 32 + 3) // 4 * 4)
    sf = s.contiguous().view(torch.uint8).reshape(-1)
    if sf.numel() != want:
        raise Mxfp8LayoutError(
            f"{what}: {sf.numel()} scale bytes, expected {want} (F8_128x4-swizzled "
            f"UE8M0 of [{n}, {k // 32}])"
        )
    return w, sf


__all__ = ["Mxfp8LayoutError", "mxfp8_linear_nk"]
