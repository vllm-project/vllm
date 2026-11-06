# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""o_proj of the low-batch attention: FlashInfer's low-latency MXFP8 GEMM.

The kernel class behind ``mm_mxfp8``'s ``cutedsl_low_latency`` backend,
compiled here with an 8-, 16- or 32-row activation tile and a fixed tactic
per tile. Only its activation (B) load warp waits for the previous grid; the
weight (A) load warp fills its stages without waiting, both feeding each
stage's barrier. It reads the weights and the weight scales of the layer's
FlashInfer ``cute-dsl`` MXFP8 kernel in place.
"""

from functools import cache

import torch
from torch import nn

from vllm.model_executor.kernels.linear.mxfp8.flashinfer import (
    FlashInferCutedslMxfp8LinearKernel,
)

# (cta_k, ab_stages, sfb_tmem_stages, split_k) per activation tile.
TACTICS = {8: (256, 4, 1, 2), 16: (256, 4, 1, 2), 32: (128, 10, 1, 1)}


def tile_rows(rows: int) -> int:
    return 8 if rows <= 8 else 16 if rows <= 16 else 32


def mxfp8_weights(linear: nn.Module, n: int, k: int):
    """``linear``'s E4M3 weight [n, k] (K-major) and its 128x4 E8M0 scales,
    when it is a FlashInfer ``cute-dsl`` MXFP8 linear without bias."""
    method = getattr(linear, "scheme", getattr(linear, "quant_method", None))
    kernel = getattr(method, "kernel", None)
    weight = getattr(linear, "weight", None)
    scale = getattr(linear, "weight_scale", None)
    if (
        type(kernel) is not FlashInferCutedslMxfp8LinearKernel
        or getattr(linear, "bias", None) is not None
        or weight is None
        or scale is None
        or weight.dtype != torch.float8_e4m3fn
        or weight.numel() != n * k
        or scale.numel() != n * k // 32
    ):
        return None
    # vLLM keeps the K-major weight as a column-major [K, N] view.
    if weight.shape == (k, n):
        weight = weight.t()
    if weight.shape != (n, k) or not weight.is_contiguous():
        return None
    return weight, scale


def _ptr(dtype, address: int, align: int):
    import cutlass.cute as cute
    from cutlass.cute.runtime import make_ptr

    return make_ptr(dtype, address, cute.AddressSpace.gmem, assumed_align=align)


@cache
def _kernel(tile: int, device_index: int):
    import cutlass
    import cutlass.cute as cute
    from cutlass.cute.runtime import make_fake_stream
    from flashinfer.gemm.kernels.cute_dsl.low_latency_blockscaled_gemm import (
        LowLatencyBlockscaledGemmKernel,
    )

    cta_k, ab_stages, sfb_stages, split_k = TACTICS[tile]
    gemm = LowLatencyBlockscaledGemmKernel(
        acc_dtype=cutlass.Float32,
        mma_tiler_mnk=(128, tile, cta_k),
        num_ab_stage=ab_stages,
        num_sfb_tmem_stage=sfb_stages,
        sf_vec_size=32,
        use_pdl=True,
        split_k=split_k,
    )
    f8, sf, bf16 = cutlass.Float8E4M3FN, cutlass.Float8E8M0FNU, cutlass.BFloat16
    with torch.accelerator.device_index(device_index):
        return cute.compile(
            gemm,
            _ptr(f8, 16, 16),
            _ptr(sf, 32, 32),
            _ptr(f8, 16, 16),
            _ptr(sf, 16, 16),
            _ptr(bf16, 16, 16),
            None,
            None,
            tuple(cutlass.Int32(v) for v in (128, tile, cta_k, 1)),
            make_fake_stream(),
        )


def compile_all(device_index: int) -> None:
    for tile in TACTICS:
        _kernel(tile, device_index)


def o_proj_gemm(weight, weight_scale, act, act_scale, out, rows: int, stream) -> None:
    """``out`` [rows, n] BF16 = ``act`` [rows, k] (MXFP8, 128x4 scales)
    times ``weight`` [n, k]^T, on ``stream``. The kernel computes ``out^T``:
    its M is the weight's N."""
    import cutlass

    n, k = weight.shape
    f8, sf, bf16 = cutlass.Float8E4M3FN, cutlass.Float8E8M0FNU, cutlass.BFloat16
    _kernel(tile_rows(rows), weight.device.index)(
        _ptr(f8, weight.data_ptr(), 16),
        _ptr(sf, weight_scale.data_ptr(), 32),
        _ptr(f8, act.data_ptr(), 16),
        _ptr(sf, act_scale.data_ptr(), 16),
        _ptr(bf16, out.data_ptr(), 16),
        None,
        None,
        tuple(cutlass.Int32(v) for v in (n, rows, k, 1)),
        stream,
    )
