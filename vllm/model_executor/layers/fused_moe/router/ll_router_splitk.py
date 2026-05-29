# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def is_available() -> bool:
    try:
        import cutlass.cute  # noqa: F401

        return True
    except ImportError:
        return False


_splitk_cache: dict = {}


def _get_compiled_splitk(a, b, c, split_k: int, num_stages: int = 0):
    """Compile split-K kernel variant."""
    import cutlass.cute as cute
    from cuda.bindings.driver import CUstream
    from cutlass.cute.runtime import from_dlpack
    from torch.cuda import current_stream

    from ._ll_router_splitk_kernels import LLRouterSplitK

    K = a.shape[1]
    tiles = K // 256
    ns = num_stages if num_stages > 0 else min(12, tiles // split_k)
    cache_key = (split_k, ns)
    if cache_key in _splitk_cache:
        return _splitk_cache[cache_key]

    div = 8

    mA = (
        from_dlpack(a, assumed_align=16, enable_tvm_ffi=True)
        .mark_layout_dynamic(leading_dim=1)
        .mark_compact_shape_dynamic(mode=1, stride_order=(0, 1), divisibility=div)
    )
    mB = (
        from_dlpack(b, assumed_align=16, enable_tvm_ffi=True)
        .mark_layout_dynamic(leading_dim=1)
        .mark_compact_shape_dynamic(mode=1, stride_order=(0, 1), divisibility=div)
    )
    mC = (
        from_dlpack(c, assumed_align=16, enable_tvm_ffi=True)
        .mark_layout_dynamic(leading_dim=1)
        .mark_compact_shape_dynamic(mode=1, stride_order=(0, 1), divisibility=div)
    )

    gemm = LLRouterSplitK(
        tile_n=16, tile_k=256, num_stages=ns, num_dma_warps=4, split_k=split_k
    )
    stream = CUstream(current_stream().cuda_stream)
    compiled = cute.compile(
        gemm.call_splitk, mA, mB, mC, stream, options="--enable-tvm-ffi"
    )
    _splitk_cache[cache_key] = compiled
    logger.debug("Compiled ll_router_splitk: sk=%d ns=%d", split_k, ns)
    return compiled
