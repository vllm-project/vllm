# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import logging

import torch

logger = logging.getLogger(__name__)

_cutedsl_available: bool | None = None


# Called once per process.
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
        logger.info("cuteDSL (CUTLASS Python) not available, ll_fp32w_gemm disabled")
    return _cutedsl_available

# Cache: (M, K, a_dtype) -> compiled callable
_compiled_cache: dict[tuple, object] = {}

# lazy import helper - deferred until first actual kernel call.
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


def _stream():
    _, _, CUstream, current_stream = _cute()
    return CUstream(current_stream().cuda_stream)

# Takes flattened 1D tensors. The dot-product kernel uses raw pointer
# arithmetic, not cute's tiled layout system.
def _get_compiled_dotprod(M: int, K: int, N: int, a_flat, b_flat, c_flat, a_dtype):
    cute, from_dlpack, CUstream, current_stream = _cute()

    # N is not in the key.
    # the kernel handles any N at runtime (one CTA per expert, grid size = N).
    bs = 128 if (K % 2048 != 0 and K % 1024 == 0) else 256
    key = (M, K, a_dtype)
    if key in _compiled_cache:
        return _compiled_cache[key]

    # cache check before any expensive work.
    from ._ll_fp32w_dotprod import make_host_fp32w

    # Use BS=128 when K doesn't divide by VPT(8)*256=2048 but does by VPT(8)*128=1024
    host_fn = make_host_fp32w(K) # creates a new kernel closure with K baked
    # into all loop bounds as Constexpr

    a_c = from_dlpack(
        a_flat, assumed_align=32, enable_tvm_ffi=True
    ).mark_layout_dynamic()
    b_c = from_dlpack(
        b_flat, assumed_align=32, enable_tvm_ffi=True
    ).mark_layout_dynamic()
    c_c = from_dlpack(
        c_flat, assumed_align=32, enable_tvm_ffi=True
    ).mark_layout_dynamic()

    stream = _stream()

    compiled = cute.compile(
        host_fn,
        a_c,
        b_c,
        c_c,
        M,
        K,
        N,
        stream,
        options="--enable-tvm-ffi --ptxas-options -maxrregcount=64",
    )
    _compiled_cache[key] = compiled
    logger.debug("Compiled ll_fp32w_dotprod: M=%d, K=%d", M, K)
    return compiled


def ll_fp32w_gemm(
    hidden_states: torch.Tensor,  # [M, K] bf16/fp16/fp32
    router_weight: torch.Tensor,  # [N, K] fp32
    output_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:                # [M, N] fp32
    M, K = hidden_states.shape
    N = router_weight.shape[0]
    stream = _stream()
    output = torch.empty(M, N, dtype=output_dtype, device=hidden_states.device)

    a_flat = hidden_states.reshape(-1)
    b_flat = router_weight.reshape(-1)
    c_flat = output.reshape(-1)

    compiled = _get_compiled_dotprod(M, K, N, a_flat, b_flat, c_flat, hidden_states.dtype)
    compiled(a_flat, b_flat, c_flat, N, stream)

    return output
