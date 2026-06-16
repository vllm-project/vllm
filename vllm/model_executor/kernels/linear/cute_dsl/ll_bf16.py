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
        logger.info("cuteDSL (CUTLASS Python) not available, ll_bf16_gemm disabled")
    return _cutedsl_available


# Two separate caches because the two kernels have different specialization axes
# Dot-prod: keyed on (M, K, bs) -> M and K are Constexpr in the kernel.
_compiled_cache: dict[tuple[int, int, int], object] = {}
# Split-K: keyed on (split_k, num_stages) -> compiled callable, fully shape-dynamic.
_splitk_cache: dict = {}

# Per-model tuned configs: (K, N) -> {M: ("dotprod", bs) | ("splitk", sk, stages)}
_TUNED_CONFIGS: dict[tuple[int, int], dict[int, tuple]] = {
    (7168, 384): {M: ("splitk", 4, 3) for M in range(5, 16)},
}

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


# Takes full 2D row-major tensors.
def _get_compiled_dotprod(M: int, K: int, N: int, a, b, c, bs: int = 128):
    cute, from_dlpack, CUstream, current_stream = _cute()

    key = (M, K, bs)
    if key in _compiled_cache:
        return _compiled_cache[key]

    # cache check before any expensive work.
    from ._ll_bf16_dotprod import make_host_bf16

    host_fn = make_host_bf16(K, bs=bs)  # creates a new kernel closure with K baked
    # into all loop bounds as Constexpr

    a_c = from_dlpack(a, assumed_align=32, enable_tvm_ffi=True).mark_layout_dynamic(
        leading_dim=1
    )
    b_c = from_dlpack(b, assumed_align=32, enable_tvm_ffi=True).mark_layout_dynamic(
        leading_dim=1
    )
    c_c = from_dlpack(c, assumed_align=32, enable_tvm_ffi=True).mark_layout_dynamic(
        leading_dim=1
    )

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
        options="--enable-tvm-ffi --ptxas-options -maxrregcount=64",  # cap regs
    )
    _compiled_cache[key] = compiled
    logger.debug("Compiled ll_bf16_dotprod: M=%d, K=%d", M, K)
    return compiled


# Takes full 2D tensors (not flattened) as opposed to _get_compiled_dotprod.
def _get_compiled_splitk(a, b, c, split_k: int, num_stages: int):
    cute, from_dlpack, CUstream, current_stream = _cute()
    from ._ll_bf16_splitk import LLBf16SplitK

    # shape-dynamic: one binary (same split_k and num_stages) works for all shapes.
    cache_key = (split_k, num_stages)
    if cache_key in _splitk_cache:
        return _splitk_cache[cache_key]

    div = 8
    mA = (
        from_dlpack(a, assumed_align=16, enable_tvm_ffi=True)
        .mark_layout_dynamic(leading_dim=1)  # dimension 1 (K) has a dynamic stride
        .mark_compact_shape_dynamic(
            mode=1,  # mode 1 (K) is dynamic but guaranteed divisible by div=8
            stride_order=(0, 1),  # row-major
            divisibility=div,
        )
        # Helps address math when K alignment is known.
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

    # TODO (roberto): add tile_n, tile_k and num_dma_warps to the tuning space.
    gemm = LLBf16SplitK(
        tile_n=16, tile_k=256, num_stages=num_stages, num_dma_warps=4, split_k=split_k
    )
    compiled = cute.compile(
        gemm.call_splitk, mA, mB, mC, _stream(), options="--enable-tvm-ffi"
    )
    _splitk_cache[cache_key] = compiled
    logger.debug("Compiled ll_bf16_splitk: sk=%d ns=%d", split_k, num_stages)
    return compiled


def _get_config(M: int, K: int, N: int) -> tuple:
    """Look up tuned config, fall back to default dispatch."""
    # TODO (roberto): increase search space - autotuning system
    model_configs = _TUNED_CONFIGS.get((K, N))
    if model_configs and M in model_configs:
        return model_configs[M]
    if M > 4 and K >= 2048:
        return ("splitk", 8, 2)
    return ("dotprod", 128)


def ll_bf16_gemm(
    hidden_states: torch.Tensor,  # [M, K] bf16
    router_weight: torch.Tensor,  # [N, K] bf16
    output_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:  # [M, N] fp32
    M, K = hidden_states.shape
    N = router_weight.shape[0]
    stream = _stream()
    output = torch.empty(M, N, dtype=output_dtype, device=hidden_states.device)

    config = _get_config(M, K, N)

    if config[0] == "splitk":
        _, split_k, num_stages = config
        compiled = _get_compiled_splitk(
            hidden_states, router_weight, output, split_k, num_stages
        )
        compiled(hidden_states, router_weight, output, stream, 1.0)
    else:
        _, bs = config
        compiled = _get_compiled_dotprod(
            M, K, N, hidden_states, router_weight, output, bs
        )
        compiled(hidden_states, router_weight, output, N, stream)

    return output
