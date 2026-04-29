# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""API wrapper for TMA pipeline A GEMM."""

from __future__ import annotations

import logging

import torch

logger = logging.getLogger(__name__)

_compiled_cache: dict[tuple, object] = {}
_desc_cache: dict[tuple, torch.Tensor] = {}


def _make_desc(tensor, box_rows, box_cols):
    from ._ll_a_gemm_tma_prefetch import create_tma_descriptor
    desc_bytes = create_tma_descriptor(tensor, box_rows=box_rows,
                                       box_cols=box_cols)
    return torch.frombuffer(desc_bytes, dtype=torch.uint8).cuda()


def _get_compiled(is_fp8, a, b, c, descA_dev, descB_dev):
    import cutlass.cute as cute
    from cuda.bindings.driver import CUstream
    from cutlass.cute.runtime import from_dlpack
    from torch.cuda import current_stream
    from ._ll_a_gemm_tma_prefetch import LLAGemmTmaPrefetch, TMA_BOX_K

    K, N = a.shape[1], b.shape[0]
    cache_key = (is_fp8, K, N)
    if cache_key in _compiled_cache:
        return _compiled_cache[cache_key]

    div = 8
    mA = (from_dlpack(a, assumed_align=16, enable_tvm_ffi=True)
          .mark_layout_dynamic(leading_dim=1)
          .mark_compact_shape_dynamic(mode=1, stride_order=(0, 1),
                                      divisibility=div))
    mB = (from_dlpack(b, assumed_align=16, enable_tvm_ffi=True)
          .mark_layout_dynamic(leading_dim=1)
          .mark_compact_shape_dynamic(mode=1, stride_order=(0, 1),
                                      divisibility=div))
    mC = (from_dlpack(c, assumed_align=16, enable_tvm_ffi=True)
          .mark_layout_dynamic(leading_dim=1)
          .mark_compact_shape_dynamic(mode=1, stride_order=(0, 1),
                                      divisibility=div))
    dA = (from_dlpack(descA_dev, assumed_align=64, enable_tvm_ffi=True)
          .mark_layout_dynamic())
    dB = (from_dlpack(descB_dev, assumed_align=64, enable_tvm_ffi=True)
          .mark_layout_dynamic())

    K_eff = K
    gemm = LLAGemmTmaPrefetch(tile_k=256, num_stages=8,
                               K_eff=K_eff, is_fp8=is_fp8)
    stream = CUstream(current_stream().cuda_stream)
    compiled = cute.compile(gemm, mA, mB, mC, dA, dB, stream,
                            options="--enable-tvm-ffi")
    _compiled_cache[cache_key] = compiled
    logger.debug("Compiled ll_a_gemm_tma: is_fp8=%s K=%d N=%d", is_fp8, K, N)
    return compiled


def ll_a_gemm_tma(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    is_fp8: bool = False,
) -> torch.Tensor:
    from cuda.bindings.driver import CUstream
    from torch.cuda import current_stream
    from ._ll_a_gemm_tma_prefetch import TMA_BOX_K

    M, K = hidden_states.shape
    N = weight.shape[0]
    output = torch.empty(M, N, dtype=torch.bfloat16,
                         device=hidden_states.device)

    desc_key_a = (hidden_states.data_ptr(), M, K)
    desc_key_b = (weight.data_ptr(), N, K)
    if desc_key_a not in _desc_cache:
        _desc_cache[desc_key_a] = _make_desc(hidden_states, 16, TMA_BOX_K)
    if desc_key_b not in _desc_cache:
        _desc_cache[desc_key_b] = _make_desc(weight, 16, TMA_BOX_K)

    compiled = _get_compiled(is_fp8, hidden_states, weight, output,
                              _desc_cache[desc_key_a], _desc_cache[desc_key_b])

    stream = CUstream(current_stream().cuda_stream)
    compiled(hidden_states, weight, output,
             _desc_cache[desc_key_a], _desc_cache[desc_key_b], stream)
    return output
