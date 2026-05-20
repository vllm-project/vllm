# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import logging

import torch

logger = logging.getLogger(__name__)

_available: bool | None = None


def is_available() -> bool:
    global _available
    if _available is not None:
        return _available
    try:
        import cutlass.cute  # noqa: F401
        _available = True
    except ImportError:
        _available = False
    return _available


_cache: dict[tuple, object] = {}


def _get_compiled(a_bf16, b_bf16, out, sa_flat, sb_flat):
    import cutlass.cute as cute
    from cuda.bindings.driver import CUstream
    from cutlass.cute.runtime import from_dlpack
    from torch.cuda import current_stream

    from ._ll_fp8_block_kernels import LLFp8BlockGemm

    cache_key = ("fp8_block",)
    if cache_key in _cache:
        return _cache[cache_key]

    div = 8
    mA = (from_dlpack(a_bf16, assumed_align=16, enable_tvm_ffi=True)
          .mark_layout_dynamic(leading_dim=1)
          .mark_compact_shape_dynamic(mode=1, stride_order=(0, 1),
                                      divisibility=div))
    mB = (from_dlpack(b_bf16, assumed_align=16, enable_tvm_ffi=True)
          .mark_layout_dynamic(leading_dim=1)
          .mark_compact_shape_dynamic(mode=1, stride_order=(0, 1),
                                      divisibility=div))
    mC = (from_dlpack(out, assumed_align=16, enable_tvm_ffi=True)
          .mark_layout_dynamic(leading_dim=1)
          .mark_compact_shape_dynamic(mode=1, stride_order=(0, 1),
                                      divisibility=div))
    mSA = from_dlpack(sa_flat, assumed_align=4,
                      enable_tvm_ffi=True).mark_layout_dynamic()
    mSB = from_dlpack(sb_flat, assumed_align=4,
                      enable_tvm_ffi=True).mark_layout_dynamic()

    gemm = LLFp8BlockGemm(tile_n=16, tile_k=256, num_stages=2,
                           num_dma_warps=4)
    stream = CUstream(current_stream().cuda_stream)
    compiled = cute.compile(gemm, mA, mB, mC, mSA, mSB, stream,
                            options="--enable-tvm-ffi")
    _cache[cache_key] = compiled
    logger.debug("Compiled ll_fp8_block_gemm")
    return compiled


def ll_fp8_block_gemm(
    q_input: torch.Tensor,
    input_scale: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    output: torch.Tensor,
) -> None:
    from cuda.bindings.driver import CUstream
    from torch.cuda import current_stream

    M, K_fp8 = q_input.shape
    N = weight.shape[0]

    # View FP8 as BF16 (2 fp8 = 1 bf16)
    a_bf16 = q_input.view(torch.bfloat16)
    b_bf16 = weight.view(torch.bfloat16)
    
    # Use just view instead?
    sa_flat = input_scale.as_strided((input_scale.numel(),), (1,))
    sb_flat = weight_scale.as_strided((weight_scale.numel(),), (1,))

    stream = CUstream(current_stream().cuda_stream)

    compiled = _get_compiled(a_bf16, b_bf16, output, sa_flat, sb_flat)
    compiled(a_bf16, b_bf16, output, sa_flat, sb_flat, stream)
