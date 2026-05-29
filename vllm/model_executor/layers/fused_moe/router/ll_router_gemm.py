# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import logging

import torch

logger = logging.getLogger(__name__)

_cutedsl_available: bool | None = None


def is_available() -> bool:
    """Check if cuteDSL backend is available."""
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


# Cache: (M, K) -> compiled callable
_compiled_cache: dict[tuple[int, int], object] = {}


def _get_compiled(M: int, K: int, N: int, a_flat, b_flat, c_flat):
    """Get or compile a dot-product kernel for the given (M, K) combination."""
    import cutlass.cute as cute
    from cuda.bindings.driver import CUstream
    from cutlass.cute.runtime import from_dlpack
    from torch.cuda import current_stream

    key = (M, K)
    if key in _compiled_cache:
        return _compiled_cache[key]

    from ._ll_router_gemm_kernels import make_host_bf16

    host_fn = make_host_bf16(K)

    a_c = from_dlpack(
        a_flat, assumed_align=32, enable_tvm_ffi=True
    ).mark_layout_dynamic()
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
        options="--enable-tvm-ffi --ptxas-options -maxrregcount=64",
    )
    _compiled_cache[key] = compiled
    logger.debug("Compiled ll_router_gemm: M=%d, K=%d", M, K)
    return compiled


def ll_router_gemm(
    hidden_states: torch.Tensor,
    router_weight: torch.Tensor,
    output_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    from cuda.bindings.driver import CUstream
    from torch.cuda import current_stream

    M, K = hidden_states.shape
    N = router_weight.shape[0]
    stream = CUstream(current_stream().cuda_stream)
    output = torch.empty(M, N, dtype=output_dtype, device=hidden_states.device)

    if M > 4 and K >= 2048:
        from .ll_router_splitk import _get_compiled_splitk

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
        compiled = _get_compiled(M, K, N, a_flat, b_flat, c_flat)
        compiled(a_flat, b_flat, c_flat, N, stream)

    return output
