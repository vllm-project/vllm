# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import logging

import torch

logger = logging.getLogger(__name__)


def is_available() -> bool:
    try:
        import cutlass.cute  # noqa: F401
        return True
    except ImportError:
        return False


# Cache: (is_fp8, swapped) -> compiled callable
_compiled_cache: dict[tuple, object] = {}


def _get_compiled(is_fp8: bool, swapped: bool, a, b, c):
    """Get or compile an A GEMM kernel."""
    import cutlass.cute as cute
    from cuda.bindings.driver import CUstream
    from cutlass.cute.runtime import from_dlpack
    from torch.cuda import current_stream

    from ._ll_a_gemm_kernels import LLAGemm

    cache_key = (is_fp8, swapped)
    if cache_key in _compiled_cache:
        return _compiled_cache[cache_key]

    div = 8
    # For swapped path, output C=[N,M] has small M — relax divisibility
    b_div = div  # B (activations) K dim always divisible by 8
    c_div = 1 if swapped else div  # C mode 1 = M (can be 1-8)

    mA = (from_dlpack(a, assumed_align=16, enable_tvm_ffi=True)
          .mark_layout_dynamic(leading_dim=1)
          .mark_compact_shape_dynamic(mode=1, stride_order=(0, 1),
                                      divisibility=div))
    mB = (from_dlpack(b, assumed_align=16, enable_tvm_ffi=True)
          .mark_layout_dynamic(leading_dim=1)
          .mark_compact_shape_dynamic(mode=1, stride_order=(0, 1),
                                      divisibility=b_div))
    mC = (from_dlpack(c, assumed_align=16, enable_tvm_ffi=True)
          .mark_layout_dynamic(leading_dim=1)
          .mark_compact_shape_dynamic(mode=1, stride_order=(0, 1),
                                      divisibility=c_div))

    tk = 256
    tn = 8 if swapped else 16
    ns = 4
    gemm = LLAGemm(tile_n=tn, tile_k=tk, num_stages=ns,
                    num_dma_warps=4, is_fp8=is_fp8)
    stream = CUstream(current_stream().cuda_stream)
    compiled = cute.compile(gemm, mA, mB, mC, stream,
                            options="--enable-tvm-ffi")
    _compiled_cache[cache_key] = compiled
    logger.debug("Compiled ll_a_gemm: is_fp8=%s swapped=%s tile_n=%d",
                 is_fp8, swapped, tn)
    return compiled


def ll_a_gemm(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    is_fp8: bool = False,
) -> torch.Tensor:
    from cuda.bindings.driver import CUstream
    from torch.cuda import current_stream

    M = hidden_states.shape[0]
    N = weight.shape[0]

    if M <= 8:
        out_NM = torch.empty(N, M, dtype=torch.bfloat16,
                             device=hidden_states.device)
        compiled = _get_compiled(is_fp8, True, weight, hidden_states, out_NM)
        stream = CUstream(current_stream().cuda_stream)
        compiled(weight, hidden_states, out_NM, stream)
        return out_NM.T 
    else:
        output = torch.empty(M, N, dtype=torch.bfloat16,
                             device=hidden_states.device)
        compiled = _get_compiled(is_fp8, False, hidden_states, weight, output)
        stream = CUstream(current_stream().cuda_stream)
        compiled(hidden_states, weight, output, stream)
        return output
