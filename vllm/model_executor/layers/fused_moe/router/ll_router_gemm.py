# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations
import logging
import torch

logger = logging.getLogger(__name__)

# Called once per process.
_cutedsl_available: bool | None = None
def is_available() -> bool:
    global _cutedsl_available
    if _cutedsl_available is not None:
        return _cutedsl_available
    try:
        import cutlass  # noqa: F401
        import cutlass.cute  # noqa: F401

        _cutedsl_available = True
    except ImportError:
        _cutedsl_available = False
        logger.info("cuteDSL (CUTLASS Python) not available, ll_router_gemm disabled")
    return _cutedsl_available

# Two separate caches because the two kernels have different specialization axes
# Dot-prod: keyed on (M, K) -> both are Constexpr in the kernel -> each unique pair needs 
# its own binary
_compiled_cache: dict[tuple[int, int], object] = {}
# Split-K: keyed on (split_k, num_stages) -> compiled callable, fully shape-dynamic.
_splitk_cache: dict = {}

# lazy import helper - deferred until firt actual kernel call.
_cute_ctx = None
def _cute():
    global _cute_ctx
    if _cute_ctx is not None:
        return _cute_ctx
    import cutlass.cute as cute
    from cuda.bindings.driver import CUstream
    from cutlass.cute.runtime import from_dlpack
    from torch.cuda import current_stream
    _cute_ctx = (cute, from_dlpack, CUstream, current_stream)
    return _cute_ctx

# Takes flattened 1D tensors. The dot-product kernel uses raw pointer 
# arithmetic, not cute's tiled layout system.
def _get_compiled_dotprod(M: int, K: int, N: int, a_flat, b_flat, c_flat):
    cute, from_dlpack, CUstream, current_stream = _cute()

    # N is not in the key.
    # the kernel handles any N at runtime (one CTA per expert, grid size = N).
    key = (M, K)
    if key in _compiled_cache:
        return _compiled_cache[key]

    # cache check before any expensive work. 
    from ._ll_router_gemm_kernels import make_host_bf16
    host_fn = make_host_bf16(K) # creates a new kernel closure with K baked 
                                # into all loop bounds as Constexpr

    a_c = from_dlpack(
        a_flat, assumed_align=32, enable_tvm_ffi=True
    ).mark_layout_dynamic() # shape/stride can change between calls. This is 
    # what lets the cache work — one binary for all tensor sizes with the same (M, K).
    b_c = from_dlpack(
        b_flat, assumed_align=32, enable_tvm_ffi=True
    ).mark_layout_dynamic()
    c_c = from_dlpack(
        c_flat, assumed_align=32, enable_tvm_ffi=True
    ).mark_layout_dynamic()

    K_eff = K
    stream = CUstream(current_stream().cuda_stream)

    compiled = cute.compile(
        host_fn,
        a_c,
        b_c,
        c_c,
        M,
        K_eff,
        N,
        stream,
        options="--enable-tvm-ffi --ptxas-options -maxrregcount=64", # caps register usage. Found empirically.
    )
    _compiled_cache[key] = compiled
    logger.debug("Compiled ll_router_gemm: M=%d, K=%d", M, K)
    return compiled


def _get_compiled_splitk(a, b, c, split_k: int, num_stages: int = 0):
    cute, from_dlpack, CUstream, current_stream = _cute()
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


def ll_router_gemm(
    hidden_states: torch.Tensor,
    router_weight: torch.Tensor,
    output_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    _, _, CUstream, current_stream = _cute()

    M, K = hidden_states.shape
    N = router_weight.shape[0]
    stream = CUstream(current_stream().cuda_stream)
    output = torch.empty(M, N, dtype=output_dtype, device=hidden_states.device)

    if M > 4 and K >= 2048:
        compiled = _get_compiled_splitk(
            hidden_states,
            router_weight,
            output,
            split_k=8,
            num_stages=2,
        )
        compiled(hidden_states, router_weight, output, stream, 1.0)
    else:
        a_flat = hidden_states.reshape(-1)
        b_flat = router_weight.reshape(-1)
        c_flat = output.reshape(-1)
        compiled = _get_compiled_dotprod(M, K, N, a_flat, b_flat, c_flat)
        compiled(a_flat, b_flat, c_flat, N, stream)

    return output
