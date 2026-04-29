# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Low-latency router GEMM via cuteDSL.

Generalized router GEMM kernel. Supports arbitrary N (num_experts)
and K (hidden_dim) with bf16 and fp8_e4m3fn inputs,
M <= 16 tokens, fp32 output.
"""

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


# Cache: (M, is_fp8) -> compiled callable
_compiled_cache: dict[tuple[int, bool], object] = {}


def _get_compiled(M: int, is_fp8: bool, K: int, N: int, a_flat, b_flat, c_flat):
    """Get or compile a kernel for the given (M, is_fp8) combination."""
    import cutlass.cute as cute
    from cuda.bindings.driver import CUstream
    from cutlass.cute.runtime import from_dlpack
    from torch.cuda import current_stream

    from ._ll_router_gemm_kernels import host_bf16, host_fp8

    key = (M, is_fp8)
    if key in _compiled_cache:
        return _compiled_cache[key]

    host_fn = host_fp8 if is_fp8 else host_bf16

    a_c = from_dlpack(a_flat, assumed_align=32,
                      enable_tvm_ffi=True).mark_layout_dynamic()
    b_c = from_dlpack(b_flat, assumed_align=32,
                      enable_tvm_ffi=True).mark_layout_dynamic()
    c_c = from_dlpack(c_flat, assumed_align=32,
                      enable_tvm_ffi=True).mark_layout_dynamic()

    K_eff = K // 2 if is_fp8 else K
    stream = CUstream(current_stream().cuda_stream)

    compiled = cute.compile(host_fn, a_c, b_c, c_c, M, K_eff, N, stream,
                            options="--enable-tvm-ffi")
    _compiled_cache[key] = compiled
    logger.debug("Compiled ll_router_gemm: M=%d, is_fp8=%s", M, is_fp8)
    return compiled


def ll_router_gemm(
    hidden_states: torch.Tensor,
    router_weight: torch.Tensor,
    output_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Low-latency router GEMM: C[M,N] = A[M,K] @ B[N,K]^T.

    Args:
        hidden_states: [M, K] input tensor (bf16 or fp8_e4m3fn), M <= 16.
        router_weight: [N, K] weight tensor (same dtype as input).
        output_dtype: Output dtype (default float32).

    Returns:
        [M, N] output tensor.
    """
    from cuda.bindings.driver import CUstream
    from torch.cuda import current_stream

    M, K = hidden_states.shape
    N = router_weight.shape[0]
    is_fp8 = hidden_states.dtype == torch.float8_e4m3fn

    output = torch.empty(M, N, dtype=output_dtype, device=hidden_states.device)

    if is_fp8:
        a_flat = hidden_states.view(torch.int16).reshape(-1)
        b_flat = router_weight.view(torch.int16).reshape(-1)
    else:
        a_flat = hidden_states.reshape(-1)
        b_flat = router_weight.reshape(-1)
    c_flat = output.reshape(-1)

    compiled = _get_compiled(M, is_fp8, K, N, a_flat, b_flat, c_flat)

    # TVM FFI: pass torch tensors directly (no from_dlpack on hot path)
    K_eff = K // 2 if is_fp8 else K
    stream = CUstream(current_stream().cuda_stream)
    compiled(a_flat, b_flat, c_flat, K_eff, N, stream)

    return output
